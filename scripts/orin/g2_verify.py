#!/usr/bin/env python3
"""G2 verification: pi0.5 (pi05_libero) single-inference + L1 latency on Jetson Orin.

L1 caliber (pre-registered in the project plan) = PURE in-process inference wall time,
NO websocket / NO network round-trip. This is the engine-floor number; the end-to-end
control cycle (L2) and LIBERO required control freq (L3) are measured separately.

Uses openpi's own make_libero_example() for a dummy observation (model compute is
input-independent, so latency is valid; action correctness is G3's closed-loop job).
"""
import time
import numpy as np

print("=== [0/5] JAX devices (confirm GPU, not CPU fallback) ===", flush=True)
import jax
print(f"jax devices : {jax.devices()}", flush=True)
print(f"jax backend : {jax.default_backend()}", flush=True)

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy
from openpi.shared import download

print("=== [1/5] resolve config + download checkpoint (gs://openpi-assets/checkpoints/pi05_libero) ===", flush=True)
t0 = time.perf_counter()
config = _config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
print(f"checkpoint_dir = {checkpoint_dir}", flush=True)
print(f"download+config time = {time.perf_counter()-t0:.1f}s", flush=True)

print("=== [2/5] create_trained_policy (load model into memory) ===", flush=True)
t0 = time.perf_counter()
policy = policy_config.create_trained_policy(config, checkpoint_dir)
load_time = time.perf_counter() - t0
print(f"policy load time = {load_time:.1f}s", flush=True)

print("=== [3/5] dummy LIBERO observation (openpi make_libero_example) ===", flush=True)
example = libero_policy.make_libero_example()
for k, v in example.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}", flush=True)
    else:
        print(f"  {k}: {v!r}", flush=True)

print("=== [4/5] warmup (absorb JAX jit compilation) ===", flush=True)
WARMUP = 3
for i in range(WARMUP):
    t0 = time.perf_counter()
    result = policy.infer(example)
    print(f"  warmup {i}: {time.perf_counter()-t0:.3f}s", flush=True)

actions = np.asarray(result["actions"])
print(f"action_chunk shape={actions.shape} dtype={actions.dtype}", flush=True)

print("=== [5/5] L1 latency measurement ===", flush=True)
N = 20
times = []
for _ in range(N):
    t0 = time.perf_counter()
    result = policy.infer(example)
    times.append(time.perf_counter() - t0)
times = np.array(times) * 1000.0  # ms

H = actions.shape[0] if actions.ndim > 1 else 1
print("", flush=True)
print("========== G2 L1 LATENCY RESULT (pi05_libero on Jetson Orin) ==========", flush=True)
print(f"N = {N} inferences (after {WARMUP} warmup)", flush=True)
print(f"L1 latency : mean={times.mean():.1f}ms  median={np.median(times):.1f}ms  "
      f"min={times.min():.1f}ms  max={times.max():.1f}ms  std={times.std():.1f}ms", flush=True)
print(f"L1 throughput (policy.infer calls/sec) = {1000.0/times.mean():.2f} Hz", flush=True)
print(f"action chunk H = {H}  =>  per-action latency = {times.mean()/H:.1f}ms  "
      f"=> open-loop action rate = {H*1000.0/times.mean():.1f} Hz", flush=True)
print(f"model load time = {load_time:.1f}s", flush=True)
print(f"jax backend = {jax.default_backend()}  devices = {jax.devices()}", flush=True)
print("========================================================================", flush=True)
