#!/usr/bin/env python3
"""G3 bridge client: LIBERO closed-loop eval against a remote openpi policy server.

Adapted from openpi examples/libero/main.py (the official LIBERO eval). Differences:

- torch-free task enumeration. libero's benchmark/__init__.py does `import torch` but only
  ever calls torch.load (to read each task's .pruned_init initial states). Those files are
  torch.save ZIP archives whose payload is a pure-numpy pickle, so we install a minimal torch
  stub whose load() reads the zip directly. This avoids installing torch OR openpi-client,
  either of which forces numpy>=1.26 and would break this machine's pinned
  numpy==1.22.4 / opencv 4.6 / numba 0.56.4 LIBERO stack.
- measures L2 (client-side end-to-end obs->action wall time) per inference, and records the
  server's self-reported inference time (response["server_timing"]["infer_ms"] = L1), so the
  network+serialization overhead = L2 - L1 falls out directly.
- host/port are CLI args (nothing hardcoded); per-episode results stream to a JSONL file.

Run from this directory so the vendored `openpi_client` package is importable:
    python eval_libero_bridge.py --host <ORIN_IP> --max_tasks 1 --num_trials_per_task 2
"""
import argparse
import collections
import json
import logging
import math
import os
import pathlib
import pickle
import sys
import time
import types
import zipfile

# Make the vendored openpi_client importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --- torch stub: MUST be installed before `from libero.libero import benchmark` ---
def _torch_load(path, *args, **kwargs):
    # .pruned_init is a torch.save ZIP; payload is a pure-numpy pickle at archive/data.pkl.
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = "archive/data.pkl" if "archive/data.pkl" in names else next(n for n in names if n.endswith(".pkl"))
        return pickle.loads(z.read(pkl))


_torch_stub = types.ModuleType("torch")
_torch_stub.load = _torch_load
sys.modules.setdefault("torch", _torch_stub)

import numpy as np
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="policy server host (the Orin box running serve_policy.py)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize_size", type=int, default=224)
    p.add_argument("--replan_steps", type=int, default=5, help="actions executed per policy query (official default 5)")
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_steps_wait", type=int, default=10, help="dummy-action steps to let objects settle")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--max_tasks", type=int, default=0, help="0 = all tasks in suite; >0 limits (smoke test)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--results_out", default="g3_results.jsonl")
    p.add_argument("--record_video", action="store_true")
    p.add_argument("--video_out_path", default="videos")
    return p.parse_args()


def eval_libero(args) -> None:
    np.random.seed(args.seed)

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS[args.task_suite_name]

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
    logging.info(f"Task suite: {args.task_suite_name} | evaluating {num_tasks}/{task_suite.n_tasks} tasks")

    imageio = None
    if args.record_video:
        import imageio  # optional dep, only imported when recording

        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected. Server metadata: {client.get_server_metadata()}")

    results_f = open(args.results_out, "a")
    total_episodes, total_successes = 0, 0
    all_l2_ms, all_server_ms = [], []
    chunk_shape_logged = False

    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), leave=False):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            ep_l2_ms, ep_server_ms = [], []

            while t < max_steps + args.num_steps_wait:
                try:
                    # Do nothing for the first few steps: the sim drops objects and we wait for them to fall.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match train preprocessing.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.record_video:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        # L2 = client-side end-to-end obs->action wall time (network + server inference).
                        t0 = time.perf_counter()
                        result = client.infer(element)
                        l2_ms = (time.perf_counter() - t0) * 1000.0
                        action_chunk = result["actions"]
                        server_ms = result.get("server_timing", {}).get("infer_ms")
                        ep_l2_ms.append(l2_ms)
                        if server_ms is not None:
                            ep_server_ms.append(server_ms)
                        if not chunk_shape_logged:
                            logging.info(f"action_chunk shape={np.asarray(action_chunk).shape}")
                            chunk_shape_logged = True
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan every {args.replan_steps} steps but policy predicts {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            all_l2_ms.extend(ep_l2_ms)
            all_server_ms.extend(ep_server_ms)

            rec = {
                "task_id": task_id,
                "task": task.name,
                "language": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "steps": t,
                "l2_mean_ms": round(float(np.mean(ep_l2_ms)), 2) if ep_l2_ms else None,
                "server_infer_mean_ms": round(float(np.mean(ep_server_ms)), 2) if ep_server_ms else None,
            }
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results_f.flush()

            if args.record_video and replay_images:
                suffix = "success" if done else "failure"
                seg = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{seg}_{suffix}_t{task_id}e{episode_idx}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(
                f"[{task_description}] ep{episode_idx} success={done} steps={t} "
                f"L2={rec['l2_mean_ms']}ms server_L1={rec['server_infer_mean_ms']}ms"
            )

        logging.info(f"Task {task_id} ({task_description}) success: {task_successes}/{task_episodes}")

    summary = {
        "SUMMARY": {
            "task_suite": args.task_suite_name,
            "replan_steps": args.replan_steps,
            "tasks_evaluated": num_tasks,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": round(total_successes / total_episodes, 4) if total_episodes else None,
            "l2_mean_ms": round(float(np.mean(all_l2_ms)), 2) if all_l2_ms else None,
            "l2_median_ms": round(float(np.median(all_l2_ms)), 2) if all_l2_ms else None,
            "server_infer_mean_ms": round(float(np.mean(all_server_ms)), 2) if all_server_ms else None,
        }
    }
    results_f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    results_f.close()
    logging.info(json.dumps(summary, indent=2, ensure_ascii=False))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils (identical to openpi examples/libero/main.py)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(parse_args())
