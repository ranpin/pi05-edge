#!/usr/bin/env python3
"""G4 roofline: measured device peaks + per-phase FLOPs/bytes => bound type, utilization,
and what quantization can actually move (weight-only INT4 vs W8A8 INT8).

All device peaks are MEASURED on-device (achievable peaks; absorbs power-mode effects —
nvpmodel is not locked, disclosed). Model FLOPs/bytes come from measured token shapes +
config constants (models/gemma.py get_config gemma_2b/gemma_300m, siglip.py So400m/14,
pi0_config.py; vocab 257152 = PALIGEMMA_VOCAB_SIZE).

Phases timed separately (each its own JIT, same run => same thermal state):
  siglip       : 3 images through PaliGemma.img (as embed_prefix does)
  embed_prefix : SigLIP + text token embedding
  full prefix  : embed_prefix + Gemma 2B prefill (fills KV cache)
  expert step  : one denoising step (embed_suffix + expert llm w/ KV cache + out proj)
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops

from openpi.training import config as _config
from openpi.policies import policy_config, libero_policy
from openpi.shared import download
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

print(f"jax devices: {jax.devices()} | backend: {jax.default_backend()}", flush=True)


def gpu_clock_mhz():
    try:
        with open("/sys/class/devfreq/17000000.gpu/cur_freq") as f:
            return int(f.read().strip()) / 1e6
    except Exception:
        return None


# ================= A. device achievable peaks =================
# Benchmarks are SUSTAINED (~1-2s measurement window each) so DVFS reaches the same
# boosted state as the model phases; GPU clock sampled right after each window.
# (Run 1 used short bursts: clock sampled 306-408MHz during benchmarks while phases ran
#  at 1300.5MHz => peaks underestimated, utilization overstated. This version pins state.)
def bench_gemm_cube(n, iters, warmup=5):
    k = jax.random.PRNGKey(0)
    a = jax.random.normal(k, (n, n), jnp.bfloat16)
    b = jax.random.normal(k, (n, n), jnp.bfloat16)
    f = jax.jit(lambda a, b: a @ b)
    for _ in range(warmup):
        f(a, b).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        f(a, b).block_until_ready()
    dt = (time.perf_counter() - t0) / iters
    clk = gpu_clock_mhz()
    del a, b
    return 2.0 * n**3 / dt / 1e12, clk


def bench_gemm_shape(m, n, k, iters, warmup=20):
    a = jax.random.normal(jax.random.PRNGKey(0), (m, k), jnp.bfloat16)
    b = jax.random.normal(jax.random.PRNGKey(1), (k, n), jnp.bfloat16)
    f = jax.jit(lambda a, b: a @ b)
    for _ in range(warmup):
        f(a, b).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        f(a, b).block_until_ready()
    dt = (time.perf_counter() - t0) / iters
    clk = gpu_clock_mhz()
    del a, b
    return 2.0 * m * n * k / dt / 1e12, clk


def bench_gemm_int8(n, iters, warmup=5):
    try:
        k = jax.random.PRNGKey(0)
        a = jax.random.randint(k, (n, n), -127, 127, jnp.int8)
        b = jax.random.randint(k, (n, n), -127, 127, jnp.int8)
        f = jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.int32))
        for _ in range(warmup):
            f(a, b).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            f(a, b).block_until_ready()
        dt = (time.perf_counter() - t0) / iters
        clk = gpu_clock_mhz()
        del a, b
        return 2.0 * n**3 / dt / 1e12, clk
    except Exception as e:  # noqa: BLE001
        print(f"  int8 GEMM unavailable: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None, None


def bench_bw(mbytes, iters, warmup=20):
    n = mbytes * 1024 * 1024 // 2  # bf16 elements
    x = jax.random.normal(jax.random.PRNGKey(0), (n,), jnp.bfloat16)
    c = jnp.asarray(0.5, jnp.bfloat16)
    f = jax.jit(lambda x: x + c)  # read+write, negligible FLOPs
    for _ in range(warmup):
        f(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        f(x).block_until_ready()
    dt_copy = (time.perf_counter() - t0) / iters
    gbs_copy = 2.0 * mbytes / 1024 / dt_copy
    s = jax.jit(lambda x: jnp.sum(x))  # read-only
    for _ in range(warmup):
        s(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        s(x).block_until_ready()
    dt_read = (time.perf_counter() - t0) / iters
    gbs_read = mbytes / 1024 / dt_read
    clk = gpu_clock_mhz()
    del x
    return gbs_copy, gbs_read, clk


print("=== A. DEVICE ACHIEVABLE PEAKS (measured, sustained load, clock-pinned) ===", flush=True)
tf_8k, clk_8k = bench_gemm_cube(8192, 40)
tf_4k, clk_4k = bench_gemm_cube(4096, 200)
tf_ffn, clk_ffn = bench_gemm_shape(1024, 16384, 2048, 250)  # FFN-like shape at T~1024
tf_qkv, clk_qkv = bench_gemm_shape(968, 2048, 2048, 500)  # prefix-GEMM-like (T~968)
int8_tops, clk_i8 = bench_gemm_int8(8192, 40)
gbs_copy, gbs_read, clk_bw = bench_bw(256, 200)
peak_tflops = max(tf_8k, tf_4k, tf_ffn, tf_qkv)
peak_gbs = max(gbs_copy, gbs_read)
ridge = peak_tflops * 1e12 / (peak_gbs * 1e9)
print(f"  bf16 GEMM 8192^3          : {tf_8k:6.1f} TFLOPS  (clk {clk_8k} MHz)", flush=True)
print(f"  bf16 GEMM 4096^3          : {tf_4k:6.1f} TFLOPS  (clk {clk_4k} MHz)", flush=True)
print(f"  bf16 GEMM 1024x16384x2048 : {tf_ffn:6.1f} TFLOPS  (clk {clk_ffn} MHz)", flush=True)
print(f"  bf16 GEMM 968x2048x2048   : {tf_qkv:6.1f} TFLOPS  (clk {clk_qkv} MHz)", flush=True)
print(f"  int8 GEMM 8192^3          : {int8_tops if int8_tops else 'n/a'} TOPS  (clk {clk_i8} MHz)", flush=True)
print(f"  mem BW copy / read-only   : {gbs_copy:.1f} / {gbs_read:.1f} GB/s  (clk {clk_bw} MHz)", flush=True)
print(f"  PEAK bf16 = {peak_tflops:.1f} TFLOPS | PEAK BW = {peak_gbs:.1f} GB/s | ridge = {ridge:.0f} FLOP/B", flush=True)

# ================= B. model characterization =================
print("\n=== B. MODEL CHARACTERIZATION ===", flush=True)
config = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
t0 = time.perf_counter()
policy = policy_config.create_trained_policy(config, ckpt)
print(f"policy load: {time.perf_counter() - t0:.1f}s", flush=True)
model = policy._model

counts = {"siglip": 0, "gemma2b": 0, "expert": 0, "action_io": 0, "other": 0}
sample_paths = {k: [] for k in counts}
for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(model)):
    if not hasattr(leaf, "size"):
        continue
    p = "/".join(str(getattr(part, "key", getattr(part, "attr", part))) for part in path)
    n = leaf.size
    if "/img/" in f"/{p}/" or p.startswith("PaliGemma/img"):
        key = "siglip"
    elif "action_in_proj" in p or "action_out_proj" in p or "time_mlp" in p:
        key = "action_io"
    elif "llm" in p and "_1" in p:
        key = "expert"
    elif "llm" in p:
        key = "gemma2b"
    else:
        key = "other"
    counts[key] += n
    if len(sample_paths[key]) < 3:
        sample_paths[key].append(p)
total_params = sum(counts.values())
for k, v in counts.items():
    print(f"  params {k:9s}: {v/1e6:8.1f} M   e.g. {sample_paths[k][:2]}", flush=True)
print(f"  params TOTAL     : {total_params/1e6:8.1f} M", flush=True)

# config constants (source: models/gemma.py get_config, siglip.py So400m/14, pi0_config.py)
VOCAB = 257_152
G2B = dict(width=2048, depth=18, mlp=16_384, heads=8, kv_heads=1, head_dim=256)
EXP = dict(width=1024, depth=18, mlp=4096, heads=8, kv_heads=1, head_dim=256)


def gemma_linear_params(cfg):
    w, d, mlp, L = cfg["width"], cfg["heads"] * cfg["head_dim"], cfg["mlp"], cfg["depth"]
    kv = cfg["kv_heads"] * cfg["head_dim"]
    return L * (w * d + 2 * w * kv + d * w + 3 * w * mlp)


P_G2B = gemma_linear_params(G2B)
P_EXP = gemma_linear_params(EXP)
print(f"  config check: gemma_2b linear = {P_G2B/1e6:.1f} M | expert linear = {P_EXP/1e6:.1f} M", flush=True)

# observation (same path as policy.infer + sample_actions)
example = libero_policy.make_libero_example()
inputs = policy._input_transform(jax.tree.map(lambda x: x, example))
inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
observation = _model.Observation.from_dict(inputs)
observation = _model.preprocess_observation(None, observation, train=False)

graphdef, state = nnx.split(model)


def jit_phase(fn):
    def wrapped(state, obs):
        m = nnx.merge(graphdef, state)
        return fn(m, obs)

    return jax.jit(wrapped)


def siglip_phase(m, obs):
    return [m.PaliGemma.img(obs.images[name], train=False)[0] for name in obs.images]


def embed_prefix_phase(m, obs):
    return m.embed_prefix(obs)


def prefill_phase(m, obs):
    prefix_tokens, prefix_mask, prefix_ar_mask = m.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = m.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return kv_cache, prefix_mask


prefix_tokens, prefix_mask, prefix_ar_mask = jit_phase(embed_prefix_phase)(state, observation)
T_prefix = int(prefix_tokens.shape[1])
kv_cache, prefix_mask_m = jit_phase(prefill_phase)(state, observation)
kv_bytes = int(sum(l.size * l.dtype.itemsize for l in jax.tree_util.tree_leaves(kv_cache)))
n_prompt = int(jnp.sum(observation.tokenized_prompt_mask))
n_images = len(observation.images)
print(f"  n_images = {n_images} | prompt tokens = {n_prompt} | T_prefix = {T_prefix}", flush=True)
print(f"  KV cache bytes (bf16) = {kv_bytes/1e6:.1f} MB", flush=True)

x_t = jnp.zeros((1, model.action_horizon, model.action_dim), jnp.float32)
time_val = jnp.ones((1,), jnp.float32)


def step_phase(m, obs):
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = m.embed_suffix(obs, x_t, time_val)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask_m, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask_m, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out), _ = m.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    return m.action_out_proj(suffix_out[:, -model.action_horizon :])


# pi05: suffix = action tokens only (state goes into prefix as discrete language tokens)
T_suffix = int(model.action_horizon)
print(f"  T_suffix = {T_suffix} (action_horizon; pi05 suffix = action tokens only)", flush=True)

# ================= C. phase timing (same run) =================
print("\n=== C. PHASE TIMING (same run, boosted back-to-back) ===", flush=True)
siglip_jit = jit_phase(siglip_phase)
embed_prefix_jit = jit_phase(embed_prefix_phase)
prefill_jit = jit_phase(prefill_phase)
step_jit = jit_phase(step_phase)


def time_phase(fn, label, warmup=3, repeat=15):
    for _ in range(warmup):
        jax.block_until_ready(fn(state, observation))
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(state, observation))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    print(
        f"  {label:38s} mean={ts.mean():7.1f}ms  median={np.median(ts):7.1f}  "
        f"min={ts.min():7.1f}  std={ts.std():5.1f}",
        flush=True,
    )
    return float(ts.mean())


t_siglip = time_phase(siglip_jit, "SigLIP (3 images)")
t_embed = time_phase(embed_prefix_jit, "embed_prefix (SigLIP+text embed)")
t_prefill = time_phase(prefill_jit, "full prefix (embed+G2B prefill)")
t_step = time_phase(step_jit, "one denoising step (expert)")
clk2 = gpu_clock_mhz()
print(f"  GPU clock after phases: {clk2} MHz", flush=True)

# ================= D. roofline =================
print("\n=== D. ROOFLINE (FLOPs/bytes vs measured peaks) ===", flush=True)
P_SIGLIP = counts["siglip"]
# SigLIP So400m/14: patch 14 -> (h/14)^2 tokens per image (h from actual obs image shape)
_img_h = int(next(iter(observation.images.values())).shape[-3])
T_i = (_img_h // 14) ** 2
T_img = n_images * T_i
prompt_slots = T_prefix - T_img
print(
    f"  prompt: {n_prompt} real tokens padded to {prompt_slots} slots "
    f"=> {(prompt_slots - n_prompt) / T_prefix * 100:.0f}% of prefill tokens are padding",
    flush=True,
)
d_g = G2B["heads"] * G2B["head_dim"]
d_e = EXP["heads"] * EXP["head_dim"]

# SigLIP: n_images separate batch-1 passes; attn full (non-causal) per image
flop_siglip = n_images * (2 * P_SIGLIP * T_i + 4 * 27 * T_i * T_i * 1152)
byte_siglip = n_images * 2 * P_SIGLIP  # weights re-read per image call
# Gemma 2B prefill: linear over T_prefix + bidirectional attention; embedder excluded (gather only)
flop_g2b = 2 * P_G2B * T_prefix + 4 * G2B["depth"] * T_prefix * T_prefix * d_g
byte_g2b = 2 * P_G2B + kv_bytes  # weights + KV write
# expert step: linear over T_suffix + cross-attn to (prefix+suffix) KV
T_kv = T_prefix + T_suffix
# measured expert params include adaRMS conditioning weights (read EVERY step, ~0 FLOPs)
P_EXP_MEAS = counts["expert"]
flop_exp = 2 * P_EXP * T_suffix + 4 * EXP["depth"] * T_suffix * T_kv * d_e
byte_exp = 2 * P_EXP_MEAS + kv_bytes  # measured expert weights (incl adarms) + prefix KV read

peak_f = peak_tflops * 1e12
peak_b = peak_gbs * 1e9


def roofline(label, flop, byte, t_meas):
    ai = flop / byte
    t_compute = flop / peak_f * 1e3
    t_memory = byte / peak_b * 1e3
    t_roof = max(t_compute, t_memory)
    bound = "COMPUTE" if t_compute >= t_memory else "MEMORY"
    util = t_roof / t_meas * 100
    eff = flop / (t_meas / 1e3) / 1e12
    print(
        f"  {label:22s} FLOPs={flop/1e9:8.2f}G  bytes={byte/1e9:6.3f}G  AI={ai:6.0f} "
        f"(ridge {ridge:.0f})  roof={t_roof:7.1f}ms [{bound:7s}]  meas={t_meas:7.1f}ms  "
        f"util={util:5.1f}%  eff={eff:5.1f}TF",
        flush=True,
    )
    return t_roof, bound


roofline("SigLIP (3 imgs)", flop_siglip, byte_siglip, t_siglip)
roofline("Gemma2B prefill", flop_g2b, byte_g2b, t_prefill - t_embed)
roofline("expert step", flop_exp, byte_exp, t_step)

# SigLIP what-if: batch 3 images in one call (weights read once)
roofline("SigLIP batched(whatif)", 3 * (2 * P_SIGLIP * T_i + 4 * 27 * T_i * T_i * 1152), 2 * P_SIGLIP, t_siglip)

# quantization what-if (roof times only)
print("\n  quant what-if (roof times):", flush=True)
int8_scale = (int8_tops / peak_tflops) if int8_tops else 2.0
for name, flop, byte_w in [
    ("Gemma2B prefill", flop_g2b, 2 * P_G2B),
    ("expert step", flop_exp, 2 * P_EXP_MEAS),
    ("SigLIP", flop_siglip, n_images * 2 * P_SIGLIP),
]:
    t_int4wo = max(flop / peak_f, (byte_w / 4) / peak_b) * 1e3  # weight-only INT4: weights/4, compute unchanged
    t_w8a8 = max(flop / (peak_f * int8_scale), (byte_w / 2) / peak_b) * 1e3  # W8A8: compute x scale, weights/2
    print(
        f"  {name:16s} bf16 roof={max(flop/peak_f, byte_w/peak_b)*1e3:7.1f}ms | "
        f"INT4 weight-only={t_int4wo:7.1f}ms | W8A8(x{int8_scale:.2f} compute)={t_w8a8:7.1f}ms",
        flush=True,
    )

print("\n=== SUMMARY ===", flush=True)
print(f"  prefix total (measured, embed+prefill) = {t_prefill:.1f} ms", flush=True)
print(f"  expert step (measured)                 = {t_step:.1f} ms", flush=True)
print(f"  implied L(num_steps) = {t_prefill:.0f} + N x {t_step:.1f} ms", flush=True)
print("=====================================================================", flush=True)
