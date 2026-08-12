#!/usr/bin/env python3
"""Aggregate per-episode LIBERO multiview samples into one validation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    episode_dirs = sorted(path.parent for path in input_root.glob("task_*/episode_*/manifest.json"))
    if len(episode_dirs) != 6:
        raise ValueError(f"Expected 6 episode directories, got {len(episode_dirs)}")

    policy_parts: dict[str, list[np.ndarray]] = {}
    state_parts: dict[str, list[np.ndarray]] = {}
    optional_state_keys = (
        "clean_sim_states",
        "perturbed_sim_states",
        "clean_qpos",
        "perturbed_qpos",
        "qvel",
        "clean_object_qpos",
        "perturbed_object_qpos",
        "trajectory_state_indices",
    )
    episode_records: list[dict[str, object]] = []
    global_pair_offset = 0

    for episode_index, episode_dir in enumerate(episode_dirs):
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        task_id = int(manifest["task_id"])
        local_episode_id = int(
            json.loads(
                (
                    input_root.parent
                    / "pi05_libero_spatial_validation_t1_t2_t6_rollouts"
                    / episode_dir.relative_to(input_root)
                    / "metadata.json"
                ).read_text(encoding="utf-8")
            )["episode_id"]
        )
        episode_key = f"task_{task_id:02d}_episode_{local_episode_id:02d}"
        with np.load(episode_dir / "policy_samples.npz", allow_pickle=False) as policy:
            num_samples = len(policy["pair_id"])
            local_pairs = np.unique(policy["pair_id"])
            if num_samples != 500 or len(local_pairs) != 50:
                raise ValueError(f"Unexpected sample counts in {episode_dir}")
            for key in (
                "observation.images.image",
                "observation.images.image2",
                "observation.state",
                "task",
                "trajectory_state_index",
                "variant",
                "view_label",
                "camera_yaw_degrees",
            ):
                policy_parts.setdefault(key, []).append(policy[key].copy())
            policy_parts.setdefault("pair_id", []).append(
                policy["pair_id"].astype(np.int64) + global_pair_offset
            )
            policy_parts.setdefault("episode_pair_id", []).append(policy["pair_id"].astype(np.int64))
            policy_parts.setdefault("sample_id", []).append(
                np.asarray([f"{episode_key}_{value}" for value in policy["sample_id"].astype(str)])
            )
            policy_parts.setdefault("task_id", []).append(
                np.full(num_samples, task_id, dtype=np.int64)
            )
            policy_parts.setdefault("episode_id", []).append(
                np.full(num_samples, local_episode_id, dtype=np.int64)
            )
            policy_parts.setdefault("episode_key", []).append(
                np.full(num_samples, episode_key)
            )

        with np.load(episode_dir / "paired_states.npz", allow_pickle=False) as states:
            for key in optional_state_keys:
                state_parts.setdefault(key, []).append(states[key].copy())
            state_parts.setdefault("target_object_position_world", []).append(
                np.repeat(states["target_object_position_world"][None], 50, axis=0)
            )
            state_parts.setdefault("orbit_target", []).append(
                np.repeat(states["orbit_target"][None], 50, axis=0)
            )
            state_parts.setdefault("camera_positions", []).append(
                np.repeat(states["camera_positions"][None], 50, axis=0)
            )
            state_parts.setdefault("camera_quaternions", []).append(
                np.repeat(states["camera_quaternions"][None], 50, axis=0)
            )
            state_parts.setdefault("task_id", []).append(np.full(50, task_id, dtype=np.int64))
            state_parts.setdefault("episode_id", []).append(
                np.full(50, local_episode_id, dtype=np.int64)
            )
            state_parts.setdefault("episode_key", []).append(np.full(50, episode_key))
            state_parts.setdefault("episode_pair_id", []).append(np.arange(50, dtype=np.int64))
            state_parts.setdefault("global_pair_id", []).append(
                np.arange(global_pair_offset, global_pair_offset + 50, dtype=np.int64)
            )

        episode_records.append(
            {
                "episode_index": episode_index,
                "episode_key": episode_key,
                "task_id": task_id,
                "episode_id": local_episode_id,
                "source": str(episode_dir),
                "global_pair_start": global_pair_offset,
                "global_pair_end_exclusive": global_pair_offset + 50,
            }
        )
        global_pair_offset += 50
        print(f"Loaded {episode_key}: 50 pairs / 500 samples", flush=True)

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "policy_samples.npz",
        **{key: np.concatenate(parts, axis=0) for key, parts in policy_parts.items()},
    )
    np.savez_compressed(
        output_dir / "paired_states.npz",
        **{key: np.concatenate(parts, axis=0) for key, parts in state_parts.items()},
    )
    aggregate_manifest = {
        "format_version": 1,
        "suite": "libero_spatial",
        "task_id": 1,
        "task_ids": [1, 2, 6],
        "num_tasks": 3,
        "episodes_per_task": 2,
        "num_episodes": 6,
        "snapshot_pairs_per_episode": 50,
        "num_snapshot_pairs": 300,
        "num_policy_samples": 3000,
        "episodes": episode_records,
        "note": "task_id=1 is only used to instantiate identical LIBERO policy feature shapes",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(aggregate_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved aggregate validation data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
