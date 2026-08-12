#!/usr/bin/env python3
"""Capture multiple successful pi0.5 rollouts for held-out LIBERO tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from capture_pi05_libero_rollout import (
    DEFAULT_CHECKPOINT,
    capture_sim_state,
    prepare_env_observation,
    save_successful_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 6])
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=31000)
    parser.add_argument("--max-attempts-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    output_root.mkdir(parents=True)

    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg = LiberoEnvConfig(
        task=args.suite,
        task_ids=args.task_ids,
        observation_height=args.image_size,
        observation_width=args.image_size,
        control_mode="relative",
        init_states=True,
    )
    print(f"Loading checkpoint once: {checkpoint}", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = "cuda"
    policy_cfg.n_action_steps = 10
    policy_cfg.num_inference_steps = 10
    policy_cfg.gradient_checkpointing = False
    policy_cfg.compile_model = False
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    policy.requires_grad_(False)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    print("Checkpoint loaded and frozen.", flush=True)

    suite_class = benchmark.get_benchmark_dict()[args.suite]
    task_suite = suite_class()
    captured: list[dict[str, object]] = []

    for task_id in args.task_ids:
        task = task_suite.get_task(task_id)
        env = LiberoEnv(
            task_suite=task_suite,
            task_id=task_id,
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
        successes = 0
        try:
            for attempt in range(args.max_attempts_per_task):
                if successes >= args.episodes_per_task:
                    break
                episode_seed = args.base_seed + task_id * 100 + attempt
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                torch.cuda.manual_seed_all(episode_seed)
                policy.reset()
                observation, _ = env.reset(seed=episode_seed)
                init_state_index = env.init_state_id - 1
                print(
                    f"Task {task_id} attempt {attempt + 1}/{args.max_attempts_per_task}: "
                    f"init_state={init_state_index} seed={episode_seed}",
                    flush=True,
                )

                records: list[dict[str, np.ndarray]] = []
                actions: list[np.ndarray] = []
                rewards: list[float] = []
                succeeded = False
                for step in range(args.max_steps):
                    env_batch = prepare_env_observation(
                        observation, env.task_description, env_preprocessor
                    )
                    records.append(capture_sim_state(env, env_batch[OBS_STATE]))
                    policy_batch = preprocessor(env_batch)
                    with torch.inference_mode():
                        action = policy.select_action(policy_batch)
                        action = postprocessor(action)
                        action = env_postprocessor({ACTION: action})[ACTION]
                    action_numpy = action.detach().cpu().numpy()[0].astype(np.float32)
                    if not np.isfinite(action_numpy).all():
                        raise RuntimeError(f"Non-finite action at task {task_id}, step {step}")
                    if env._env is None:
                        raise RuntimeError("Underlying LIBERO environment is missing")
                    raw_observation, reward, done, _ = env._env.step(action_numpy)
                    observation = env._format_raw_obs(raw_observation)
                    actions.append(action_numpy.copy())
                    rewards.append(float(reward))
                    if (step + 1) % 25 == 0 or reward > 0:
                        print(
                            f"  step={step + 1:03d} reward={float(reward):.1f}",
                            flush=True,
                        )
                    if reward > 0:
                        final_batch = prepare_env_observation(
                            observation, env.task_description, env_preprocessor
                        )
                        records.append(capture_sim_state(env, final_batch[OBS_STATE]))
                        succeeded = True
                        break
                    if done:
                        break

                if not succeeded:
                    print(f"  Task {task_id} attempt failed; discarded.", flush=True)
                    continue

                episode_id = successes
                episode_dir = output_root / f"task_{task_id:02d}" / f"episode_{episode_id:02d}"
                metadata = {
                    "format_version": 1,
                    "validation_split": True,
                    "checkpoint": str(checkpoint),
                    "suite": args.suite,
                    "task_id": task_id,
                    "task_name": task.name,
                    "language": task.language,
                    "bddl_file": str(env._task_bddl_file),
                    "episode_id": episode_id,
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
                save_successful_rollout(episode_dir, records, actions, rewards, metadata)
                captured.append(
                    {
                        "task_id": task_id,
                        "task_name": task.name,
                        "episode_id": episode_id,
                        "seed": episode_seed,
                        "init_state_index": init_state_index,
                        "num_states": len(records),
                        "relative_path": str(episode_dir.relative_to(output_root)),
                    }
                )
                successes += 1
                print(
                    f"  SUCCESS {successes}/{args.episodes_per_task}: {episode_dir}",
                    flush=True,
                )

            if successes != args.episodes_per_task:
                raise RuntimeError(
                    f"Task {task_id}: only {successes}/{args.episodes_per_task} successful episodes "
                    f"after {args.max_attempts_per_task} attempts"
                )
        finally:
            env.close()

    manifest = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "suite": args.suite,
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes_per_task,
        "base_seed": args.base_seed,
        "num_captured_episodes": len(captured),
        "episodes": captured,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(captured)} successful validation episodes to {output_root}", flush=True)


if __name__ == "__main__":
    main()
