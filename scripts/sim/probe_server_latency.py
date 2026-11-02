#!/usr/bin/env python3
"""Probe the policy server's steady-state inference latency under different inter-call gaps.

Why this exists: the G3 closed-loop smoke test measured server-side inference at ~587ms,
~20% above the G2 in-process record (~490ms). The suspect is Orin GPU DVFS — in the closed
loop the GPU idles between inferences (sim stepping + network round trip), so its clock drops
and each inference starts un-boosted; g2_verify's back-to-back loop keeps the clock boosted.
This probe isolates that one variable by sending N inferences with a controllable idle gap:

  --gap 0    back-to-back, GPU stays busy   -> expect the boosted number (~490ms)
  --gap 0.3  GPU idles 300ms between calls  -> expect the closed-loop number (~587ms) if DVFS is the cause

Reports server-side infer_ms (= L1, the policy.infer wall time measured ON the Orin) and the
client-side round trip (= L2). Run from this directory so the vendored openpi_client resolves.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from openpi_client import websocket_client_policy as _wcp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="policy server host (the Orin box)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--gap", type=float, default=0.0, help="idle seconds between inferences (simulates closed-loop GPU idle)")
    args = p.parse_args()

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    # Fixed dummy obs, same shape/dtype as real LIBERO obs; values are irrelevant to latency.
    obs = {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "prompt": "pick up the black bowl and place it on the plate",
    }

    for _ in range(args.warmup):
        client.infer(obs)
        if args.gap > 0:
            time.sleep(args.gap)

    l1, l2 = [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = client.infer(obs)
        l2.append((time.perf_counter() - t0) * 1000.0)
        l1.append(r["server_timing"]["infer_ms"])
        if args.gap > 0:
            time.sleep(args.gap)

    l1, l2 = np.array(l1), np.array(l2)
    print(f"gap={args.gap}s  N={args.n} (after {args.warmup} warmup)")
    print(f"  server L1 (infer_ms) : mean={l1.mean():.1f} median={np.median(l1):.1f} "
          f"min={l1.min():.1f} max={l1.max():.1f} std={l1.std():.1f}")
    print(f"  client L2 (roundtrip): mean={l2.mean():.1f} median={np.median(l2):.1f}")


if __name__ == "__main__":
    main()
