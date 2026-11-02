#!/usr/bin/env python3
"""G4 performance anatomy: sweep the flow-matching denoising step count (num_steps) to
decompose pi0.5's L1 latency on Orin.

sample_actions (pi0.py:217) runs two phases:
  - prefix ONCE: embed_prefix + PaliGemma.llm prefill = SigLIP image encode + Gemma 2B, fills KV cache
  - denoising loop num_steps TIMES: PaliGemma.llm on the suffix = Gemma 300M action expert
    (attends to the cached prefix KV), integrating the flow-matching ODE from t=1 to t=0.

num_steps is a keyword arg (default 10); Policy.infer forwards self._sample_kwargs to it
(policy.py:82,94), so we sweep it by setting policy._sample_kwargs -- no source change.
num_steps is a static int to the JIT, so each value recompiles; we warm up each before timing.

L(N) ~= prefix + N * per_step, hence per_step = (L(10)-L(1))/9 and prefix = L(1) - per_step.
This splits the ~504ms into the action-expert part (cuttable by reducing num_steps) vs the
backbone prefix (not cuttable that way) -- the key input to the latency/quality Pareto and to
whether faster inference could break the prefetch wall (REPORT.md table 3).
"""
import time

import numpy as np

import jax

print(f"jax devices: {jax.devices()} | backend: {jax.default_backend()}", flush=True)

from openpi.training import config as _config
from openpi.policies import policy_config, libero_policy
from openpi.shared import download

config = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
t0 = time.perf_counter()
policy = policy_config.create_trained_policy(config, ckpt)
print(f"policy load: {time.perf_counter() - t0:.1f}s", flush=True)
example = libero_policy.make_libero_example()

NUM_STEPS = [10, 5, 3, 2, 1]
WARMUP = 3
REPEAT = 15

print(f"\n{'num_steps':>9} | {'mean':>7} | {'median':>7} | {'min':>7} | {'std':>6} | {'Hz':>5}", flush=True)
res = {}
for ns in NUM_STEPS:
    policy._sample_kwargs = {**policy._sample_kwargs, "num_steps": ns}
    for _ in range(WARMUP):  # absorb JIT recompile for this num_steps value
        policy.infer(example)
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        policy.infer(example)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    res[ns] = float(ts.mean())
    print(
        f"{ns:>9} | {ts.mean():>7.1f} | {np.median(ts):>7.1f} | {ts.min():>7.1f} | "
        f"{ts.std():>6.1f} | {1000 / ts.mean():>5.2f}",
        flush=True,
    )

L10, L1 = res[10], res[1]
per_step = (L10 - L1) / 9.0
prefix = L1 - per_step
print("\n========== G4 LATENCY DECOMPOSITION (pi05_libero on Orin) ==========", flush=True)
print(f"L(1) = {L1:.1f} ms   L(10) = {L10:.1f} ms", flush=True)
print(f"per denoising step (Gemma 300M action expert) = (L10-L1)/9 = {per_step:.1f} ms", flush=True)
print(f"prefix (SigLIP + Gemma 2B prefill, once)      = L1 - per_step = {prefix:.1f} ms", flush=True)
print(
    f"action expert @ num_steps=10 = 10 x {per_step:.1f} = {10 * per_step:.1f} ms "
    f"= {100 * 10 * per_step / L10:.0f}% of L10",
    flush=True,
)
print(f"prefix share of L10 = {100 * prefix / L10:.0f}%", flush=True)
print("\nlinearity check, implied per_step = (L(N) - prefix) / N (should be ~constant):", flush=True)
for ns in sorted(res):
    print(f"  N={ns}: implied per_step = {(res[ns] - prefix) / ns:.1f} ms", flush=True)
print("====================================================================", flush=True)
