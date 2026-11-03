#!/usr/bin/env python3
"""G4 prefix decomposition: split the ~368ms prefix into SigLIP image encoding vs Gemma 2B prefill.

The prefix phase in sample_actions (pi0.py:217) is two sub-calls:
  prefix_tokens, ... = self.embed_prefix(observation)               # SigLIP(images) + token-embed(text)
  _, kv_cache       = self.PaliGemma.llm([prefix_tokens, None], ...) # Gemma 2B prefill (fills KV cache)

embed_prefix itself (pi0.py:106) calls PaliGemma.img per image (SigLIP) + llm(prompt, method="embed")
(token embedding). We JIT each phase (nnx split/merge, same pattern as nnx_utils.module_jit) and
time them separately. We also time preprocess_observation (the per-call obs prep inside sample_actions)
to reconcile the measured model-compute against the 368ms prefix derived from the num_steps sweep
(prefix = L(1) - per_step includes preprocess + noise + loop control, not just the model forwards).
NOTE: sub-phases timed individually may differ slightly from the fused sample_actions (XLA fusion).
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.training import config as _config
from openpi.policies import policy_config, libero_policy
from openpi.shared import download
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

print(f"jax devices: {jax.devices()} | backend: {jax.default_backend()}", flush=True)

config = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
t0 = time.perf_counter()
policy = policy_config.create_trained_policy(config, ckpt)
print(f"policy load: {time.perf_counter() - t0:.1f}s", flush=True)
model = policy._model

# Build the observation exactly as policy.infer + sample_actions do.
example = libero_policy.make_libero_example()
inputs = jax.tree.map(lambda x: x, example)
inputs = policy._input_transform(inputs)
inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
raw_observation = _model.Observation.from_dict(inputs)
observation = _model.preprocess_observation(None, raw_observation, train=False)

# JIT each phase with the nnx split/merge pattern (mirrors nnx_utils.module_jit).
graphdef, state = nnx.split(model)
preprocess_jit = jax.jit(lambda ro: _model.preprocess_observation(None, ro, train=False))


def make_jitted(fn):
    def wrapped(state, obs):
        m = nnx.merge(graphdef, state)
        return fn(m, obs)

    jitted = jax.jit(wrapped)
    return lambda obs: jitted(state, obs)


def siglip_phase(m, obs):
    out = []
    for name in obs.images:
        image_tokens, _ = m.PaliGemma.img(obs.images[name], train=False)
        out.append(image_tokens)
    return out


def embed_prefix_phase(m, obs):
    return m.embed_prefix(obs)


def full_prefix_phase(m, obs):
    prefix_tokens, prefix_mask, prefix_ar_mask = m.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = m.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return kv_cache


siglip_jit = make_jitted(siglip_phase)
embed_prefix_jit = make_jitted(embed_prefix_phase)
full_prefix_jit = make_jitted(full_prefix_phase)


def time_phase(fn, inp, label, warmup=3, repeat=15):
    for _ in range(warmup):
        jax.block_until_ready(fn(inp))
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(inp))
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    print(
        f"{label:42s} mean={ts.mean():7.1f}ms  median={np.median(ts):7.1f}ms  "
        f"min={ts.min():7.1f}ms  std={ts.std():5.1f}ms",
        flush=True,
    )
    return float(ts.mean())


preprocess_ms = time_phase(preprocess_jit, raw_observation, "preprocess_observation")
siglip_ms = time_phase(siglip_jit, observation, "SigLIP image encoding (2 imgs)")
embed_prefix_ms = time_phase(embed_prefix_jit, observation, "embed_prefix (SigLIP + token-embed)")
full_prefix_ms = time_phase(full_prefix_jit, observation, "full prefix (embed_prefix + Gemma2B)")

token_embed_ms = embed_prefix_ms - siglip_ms
gemma2b_prefill_ms = full_prefix_ms - embed_prefix_ms
accounted = preprocess_ms + full_prefix_ms

print("\n========== PREFIX DECOMPOSITION (boosted, in-process) ==========", flush=True)
print(f"preprocess_observation       : {preprocess_ms:7.1f} ms", flush=True)
print(f"SigLIP image encoding (2 img): {siglip_ms:7.1f} ms", flush=True)
print(f"token embedding              : {token_embed_ms:7.1f} ms", flush=True)
print(f"Gemma 2B prefill             : {gemma2b_prefill_ms:7.1f} ms", flush=True)
print(f"prefix model compute (SigLIP+Gemma2B+embed) = {full_prefix_ms:.1f} ms", flush=True)
print(f"accounted (preprocess + prefix compute)     = {accounted:.1f} ms", flush=True)
print(f"reference prefix (num_steps sweep L1-per_step) = 368 ms", flush=True)
print(f"residual (noise gen + while_loop control + sep-JIT vs fused) = {368 - accounted:.1f} ms", flush=True)
print(
    f"shares of prefix model compute: SigLIP {100*siglip_ms/full_prefix_ms:.0f}% / Gemma2B {100*gemma2b_prefill_ms/full_prefix_ms:.0f}%",
    flush=True,
)
print("=================================================================", flush=True)
