#!/usr/bin/env python3
"""G4 quality axis: serve pi05_libero with a configurable flow-matching denoising step
count (num_steps). Based on openpi scripts/serve_policy.py; after create_trained_policy it
injects num_steps into policy._sample_kwargs (forwarded to sample_actions, pi0.py:217) so
the closed-loop success rate can be swept against the latency decomposition (REPORT.md
table 4). num_steps is a JIT static, so the value is fixed for the server's lifetime —
restart the server to change it.
"""
import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Flow-matching denoising steps (sample_actions default is 10).
    num_steps: int = 10
    # Port to serve on.
    port: int = 8000
    # Training config name.
    config: str = "pi05_libero"
    # Checkpoint (gs:// URI or local dir); create_trained_policy downloads if needed.
    ckpt: str = "gs://openpi-assets/checkpoints/pi05_libero"


def main(args: Args) -> None:
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt)
    policy._sample_kwargs = {**policy._sample_kwargs, "num_steps": args.num_steps}
    logging.info(
        "serving %s with num_steps=%d (_sample_kwargs=%s)", args.config, args.num_steps, policy._sample_kwargs
    )
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
