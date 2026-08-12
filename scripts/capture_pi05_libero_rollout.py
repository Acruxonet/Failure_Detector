#!/usr/bin/env python3
"""Capture one successful frozen pi0.5 rollout with every MuJoCo state.

This script deliberately stops after collecting a clean successful trajectory.
It does not perturb states, render novel camera views, or compute disagreement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from libero.libero import benchmark

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_CHECKPOINT = Path(
    "/mnt/data/nas/hufangchi/cache/huggingface/hub/"
    "models--lerobot--pi05_libero_finetuned/snapshots/"
    "dbf8a3f794a9c4297b44f40b752712f50073d945"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def batch_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: batch_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value


def prepare_env_observation(
    observation: dict[str, Any],
    task_description: str,
    env_preprocessor: Any,
) -> dict[str, Any]:
    batch = preprocess_observation(batch_observation(observation))
    batch["task"] = [task_description]
    return env_preprocessor(batch)


def capture_sim_state(env: LiberoEnv, proprio: torch.Tensor) -> dict[str, np.ndarray]:
    if env._env is None:
        raise RuntimeError("Underlying LIBERO environment has not been created")
    sim = env._env.sim
    return {
        "sim_state": np.asarray(env._env.get_sim_state(), dtype=np.float64).copy(),
        "qpos": np.asarray(sim.data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(sim.data.qvel, dtype=np.float64).copy(),
        "proprio": proprio.detach().cpu().numpy()[0].astype(np.float32, copy=True),
    }


def evenly_spaced_middle_indices(num_states: int, count: int = 10) -> list[int]:
    if num_states < count:
        raise ValueError(f"Need at least {count} states, got {num_states}")
    upper = num_states - 1
    indices = np.rint(np.linspace(0.1 * upper, 0.9 * upper, count)).astype(int)
    if len(np.unique(indices)) != count:
        indices = np.rint(np.linspace(0, upper, count)).astype(int)
    return indices.tolist()


def save_successful_rollout(
    output_dir: Path,
    records: list[dict[str, np.ndarray]],
    actions: list[np.ndarray],
    rewards: list[float],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_indices = evenly_spaced_middle_indices(len(records))
    metadata["num_states"] = len(records)
    metadata["num_actions"] = len(actions)
    metadata["snapshot_indices"] = snapshot_indices
    metadata["snapshot_sampling"] = "10 evenly spaced indices from 10% through 90% of the trajectory"

    np.savez_compressed(
        output_dir / "trajectory.npz",
        sim_states=np.stack([record["sim_state"] for record in records]),
        qpos=np.stack([record["qpos"] for record in records]),
        qvel=np.stack([record["qvel"] for record in records]),
        proprio=np.stack([record["proprio"] for record in records]),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        snapshot_indices=np.asarray(snapshot_indices, dtype=np.int64),
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading checkpoint once: {checkpoint}", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = "cuda"
    policy_cfg.n_action_steps = 10
    policy_cfg.num_inference_steps = 10
    policy_cfg.gradient_checkpointing = False
    policy_cfg.compile_model = False

    env_cfg = LiberoEnvConfig(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=args.image_size,
        observation_width=args.image_size,
        control_mode="relative",
        init_states=True,
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    print("Checkpoint loaded.", flush=True)

    suite_class = benchmark.get_benchmark_dict()[args.suite]
    task_suite = suite_class()
    task = task_suite.get_task(args.task_id)
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        episode_length=args.max_steps,
        observation_width=args.image_size,
        observation_height=args.image_size,
        obs_type="pixels_agent_pos",
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )

    try:
        for attempt in range(args.max_attempts):
            episode_seed = args.seed + attempt
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
            policy.reset()
            observation, _ = env.reset(seed=episode_seed)
            init_state_index = env.init_state_id - 1
            print(
                f"Attempt {attempt + 1}/{args.max_attempts}: init_state={init_state_index}, "
                f"seed={episode_seed}",
                flush=True,
            )

            records: list[dict[str, np.ndarray]] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            succeeded = False

            for step in range(args.max_steps):
                env_batch = prepare_env_observation(
                    observation,
                    env.task_description,
                    env_preprocessor,
                )
                records.append(capture_sim_state(env, env_batch[OBS_STATE]))

                policy_batch = preprocessor(env_batch)
                with torch.inference_mode():
                    action = policy.select_action(policy_batch)
                    action = postprocessor(action)
                    action_transition = env_postprocessor({ACTION: action})
                    action = action_transition[ACTION]

                action_numpy = action.detach().cpu().numpy()[0].astype(np.float32)
                if not np.isfinite(action_numpy).all():
                    raise RuntimeError(f"Non-finite policy action at step {step}: {action_numpy}")

                if env._env is None:
                    raise RuntimeError("Underlying LIBERO environment unexpectedly missing")
                raw_observation, reward, done, _ = env._env.step(action_numpy)
                observation = env._format_raw_obs(raw_observation)
                actions.append(action_numpy.copy())
                rewards.append(float(reward))

                if (step + 1) % 10 == 0 or reward > 0:
                    print(
                        f"  step={step + 1:03d} reward={float(reward):.1f} "
                        f"action_norm={float(np.linalg.norm(action_numpy)):.4f}",
                        flush=True,
                    )

                if reward > 0:
                    final_batch = prepare_env_observation(
                        observation,
                        env.task_description,
                        env_preprocessor,
                    )
                    records.append(capture_sim_state(env, final_batch[OBS_STATE]))
                    succeeded = True
                    break
                if done:
                    break

            if not succeeded:
                print(f"Attempt {attempt + 1} did not succeed; discarding it.", flush=True)
                continue

            metadata = {
                "format_version": 1,
                "checkpoint": str(checkpoint),
                "suite": args.suite,
                "task_id": args.task_id,
                "task_name": task.name,
                "language": task.language,
                "bddl_file": str(env._task_bddl_file),
                "seed": episode_seed,
                "attempt": attempt + 1,
                "init_state_index": init_state_index,
                "success_source": "LIBERO sparse reward > 0",
                "success_reward": rewards[-1],
                "control_mode": "relative",
                "n_action_steps": policy_cfg.n_action_steps,
                "num_inference_steps": policy_cfg.num_inference_steps,
                "image_size": args.image_size,
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
            }
            save_successful_rollout(args.output_dir, records, actions, rewards, metadata)
            print(f"SUCCESS: saved trajectory to {args.output_dir}", flush=True)
            print(f"states={len(records)} actions={len(actions)}", flush=True)
            print(
                f"snapshot_indices={evenly_spaced_middle_indices(len(records))}",
                flush=True,
            )
            return

        raise RuntimeError(f"No successful episode found in {args.max_attempts} attempts")
    finally:
        env.close()


if __name__ == "__main__":
    main()
