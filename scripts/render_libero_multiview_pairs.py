#!/usr/bin/env python3
"""Create paired clean / object-perturbed LIBERO observations.

The input is a previously captured successful rollout. For each saved snapshot
index, this script restores the exact MuJoCo state, copies it, perturbs only the
task object's free joint, and renders five agent-view camera poses. It saves both
policy-ready array samples and human-viewable PNG/contact sheets.

This script does not load or run a policy and does not call check_success().
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero import benchmark
from PIL import Image, ImageDraw

from lerobot.envs.libero import LiberoEnv


VIEW_SPECS = (
    ("nominal", 0.0),
    ("yaw_m030", -30.0),
    ("yaw_m015", -15.0),
    ("yaw_p015", 15.0),
    ("yaw_p030", 30.0),
)
TASK_OBJECT = "akita_black_bowl_1"
TARGET_OBJECT = "plate_1"
TASK_OBJECT_JOINT = "akita_black_bowl_1_joint0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--snapshot-count", type=int, default=10)
    parser.add_argument("--translation-cm", type=float, default=4.0)
    parser.add_argument("--object-yaw-deg", type=float, default=15.0)
    return parser.parse_args()


def evenly_spaced_middle_indices(num_states: int, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError(f"snapshot-count must be positive, got {count}")
    if num_states < count:
        raise ValueError(f"Need at least {count} trajectory states, got {num_states}")
    upper = num_states - 1
    indices = np.rint(np.linspace(0.1 * upper, 0.9 * upper, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError(
            f"Cannot select {count} unique states from the trajectory's 10%-90% interval"
        )
    return indices


def rotate_about_z(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ vector


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 active rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s,
             (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                 (matrix[0, 1] + matrix[1, 0]) / s,
                 (matrix[0, 2] + matrix[2, 0]) / s]
            )
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / s,
                 (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                 (matrix[1, 2] + matrix[2, 1]) / s]
            )
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / s,
                 (matrix[0, 2] + matrix[2, 0]) / s,
                 (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
            )
    return quaternion / np.linalg.norm(quaternion)


def look_at_quaternion(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera quaternion whose -z axis looks at target."""
    backward = camera_position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    return rotation_matrix_to_wxyz(rotation)


def multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def policy_oriented_rgb(sim: Any, camera_name: str, size: int) -> np.ndarray:
    """Render uint8 HWC RGB and apply LIBERO's required 180-degree flip."""
    raw = sim.render(width=size, height=size, camera_name=camera_name)
    image = np.asarray(raw, dtype=np.uint8)
    if image.shape != (size, size, 3):
        raise RuntimeError(f"Unexpected {camera_name} render shape: {image.shape}")
    return np.flip(image, axis=(0, 1)).copy()


def object_root_body(inner_env: Any, object_name: str) -> str:
    objects = getattr(inner_env, "objects_dict", {})
    if object_name in objects:
        return str(objects[object_name].root_body)
    candidates = [name for name in inner_env.sim.model.body_names if str(name).startswith(object_name)]
    if not candidates:
        raise KeyError(f"Could not find a body for object {object_name!r}")
    return str(candidates[0])


def make_contact_sheet(
    output_path: Path,
    pair_id: int,
    trajectory_index: int,
    images: dict[str, dict[str, np.ndarray]],
) -> None:
    tile_size = next(iter(images["clean"].values())).shape[0]
    label_height = 28
    margin = 8
    header_height = 34
    width = margin * 2 + tile_size * len(VIEW_SPECS)
    height = header_height + 2 * (label_height + tile_size) + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 8),
        f"pair {pair_id:02d} | trajectory state {trajectory_index}",
        fill="black",
    )
    for row, variant in enumerate(("clean", "perturbed")):
        y_label = header_height + row * (label_height + tile_size)
        draw.text((margin, y_label + 6), variant, fill="black")
        y_image = y_label + label_height
        for column, (view_label, yaw_degrees) in enumerate(VIEW_SPECS):
            x = margin + column * tile_size
            tile = Image.fromarray(images[variant][view_label])
            sheet.paste(tile, (x, y_image))
            draw.rectangle((x, y_image, x + 88, y_image + 17), fill=(255, 255, 255))
            draw.text((x + 2, y_image + 2), f"{view_label} ({yaw_degrees:+.0f})", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def restore_state_without_success_check(base_env: Any, state: np.ndarray) -> None:
    base_env.set_state(state)
    base_env.sim.forward()
    base_env._post_process()
    base_env._update_observables(force=True)


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    trajectory_path = rollout_dir / "trajectory.npz"
    metadata_path = rollout_dir / "metadata.json"
    if not trajectory_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing trajectory.npz or metadata.json in {rollout_dir}")

    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trajectory = np.load(trajectory_path, allow_pickle=False)
    saved_snapshot_indices = trajectory["snapshot_indices"].astype(np.int64)
    if args.snapshot_count == len(saved_snapshot_indices):
        snapshot_indices = saved_snapshot_indices
        snapshot_sampling = "saved source-rollout snapshot indices"
    else:
        snapshot_indices = evenly_spaced_middle_indices(
            len(trajectory["sim_states"]), args.snapshot_count
        )
        snapshot_sampling = (
            f"{args.snapshot_count} evenly spaced unique indices from 10% through 90% "
            "of the same successful trajectory"
        )

    suite_name = str(source_metadata["suite"])
    task_id = int(source_metadata["task_id"])
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_length=280,
        observation_width=args.image_size,
        observation_height=args.image_size,
        obs_type="pixels_agent_pos",
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )

    output_dir.mkdir(parents=True)
    samples_dir = output_dir / "images"
    contact_sheets_dir = output_dir / "contact_sheets"

    policy_scene_images: list[np.ndarray] = []
    policy_wrist_images: list[np.ndarray] = []
    policy_states: list[np.ndarray] = []
    sample_tasks: list[str] = []
    sample_ids: list[str] = []
    sample_pair_ids: list[int] = []
    sample_trajectory_indices: list[int] = []
    sample_variants: list[str] = []
    sample_view_labels: list[str] = []
    sample_yaw_degrees: list[float] = []
    sample_records: list[dict[str, Any]] = []

    clean_sim_states: list[np.ndarray] = []
    perturbed_sim_states: list[np.ndarray] = []
    clean_qpos: list[np.ndarray] = []
    perturbed_qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    clean_object_qpos: list[np.ndarray] = []
    perturbed_object_qpos: list[np.ndarray] = []

    try:
        env.reset(seed=int(source_metadata["seed"]))
        if env._env is None:
            raise RuntimeError("LIBERO environment was not initialized")
        base_env = env._env
        sim = base_env.sim
        inner_env = base_env.env

        camera_id = int(sim.model.camera_name2id("agentview"))
        if int(sim.model.cam_bodyid[camera_id]) != 0:
            raise RuntimeError("agentview is not attached to the world body")
        nominal_camera_position = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy()
        nominal_camera_quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()

        reference_state = trajectory["sim_states"][int(snapshot_indices[0])]
        restore_state_without_success_check(base_env, reference_state)
        task_body = object_root_body(inner_env, TASK_OBJECT)
        target_body = object_root_body(inner_env, TARGET_OBJECT)
        task_position = np.asarray(sim.data.get_body_xpos(task_body), dtype=np.float64).copy()
        target_position = np.asarray(sim.data.get_body_xpos(target_body), dtype=np.float64).copy()
        orbit_target = 0.5 * (task_position + target_position)

        camera_positions: list[np.ndarray] = []
        camera_quaternions: list[np.ndarray] = []
        nominal_offset = nominal_camera_position - orbit_target
        for _, yaw_degrees in VIEW_SPECS:
            if yaw_degrees == 0.0:
                position = nominal_camera_position.copy()
                quaternion = nominal_camera_quaternion.copy()
            else:
                position = orbit_target + rotate_about_z(nominal_offset, yaw_degrees)
                quaternion = look_at_quaternion(position, orbit_target)
            camera_positions.append(position)
            camera_quaternions.append(quaternion)

        translation = np.array([args.translation_cm / 100.0, 0.0, 0.0], dtype=np.float64)
        yaw_radians = math.radians(args.object_yaw_deg)
        yaw_quaternion = np.array(
            [math.cos(yaw_radians / 2.0), 0.0, 0.0, math.sin(yaw_radians / 2.0)],
            dtype=np.float64,
        )

        for pair_id, trajectory_index in enumerate(snapshot_indices.tolist()):
            source_state = trajectory["sim_states"][trajectory_index].astype(np.float64, copy=True)
            source_proprio = trajectory["proprio"][trajectory_index].astype(np.float32, copy=True)
            restore_state_without_success_check(base_env, source_state)
            clean_object = np.asarray(sim.data.get_joint_qpos(TASK_OBJECT_JOINT), dtype=np.float64).copy()
            if clean_object.shape != (7,):
                raise RuntimeError(
                    f"Expected free-joint qpos shape (7,), got {clean_object.shape}"
                )

            variant_states: dict[str, np.ndarray] = {"clean": source_state.copy()}
            perturbed_object = clean_object.copy()
            perturbed_object[:3] += translation
            perturbed_object[3:7] = multiply_quaternions_wxyz(
                yaw_quaternion, clean_object[3:7]
            )
            sim.data.set_joint_qpos(TASK_OBJECT_JOINT, perturbed_object)
            sim.forward()
            variant_states["perturbed"] = np.asarray(base_env.get_sim_state(), dtype=np.float64).copy()

            clean_sim_states.append(variant_states["clean"].copy())
            perturbed_sim_states.append(variant_states["perturbed"].copy())
            clean_qpos.append(trajectory["qpos"][trajectory_index].astype(np.float64, copy=True))
            perturbed_qpos.append(np.asarray(sim.data.qpos, dtype=np.float64).copy())
            qvel.append(trajectory["qvel"][trajectory_index].astype(np.float64, copy=True))
            clean_object_qpos.append(clean_object)
            perturbed_object_qpos.append(perturbed_object)

            pair_images: dict[str, dict[str, np.ndarray]] = {"clean": {}, "perturbed": {}}
            for variant in ("clean", "perturbed"):
                restore_state_without_success_check(base_env, variant_states[variant])
                variant_dir = samples_dir / f"pair_{pair_id:02d}" / variant
                variant_dir.mkdir(parents=True, exist_ok=True)

                wrist_image = policy_oriented_rgb(sim, "robot0_eye_in_hand", args.image_size)
                Image.fromarray(wrist_image).save(variant_dir / "wrist.png")

                for view_index, (view_label, yaw_degrees) in enumerate(VIEW_SPECS):
                    sim.model.cam_pos[camera_id] = camera_positions[view_index]
                    sim.model.cam_quat[camera_id] = camera_quaternions[view_index]
                    sim.forward()
                    scene_image = policy_oriented_rgb(sim, "agentview", args.image_size)
                    Image.fromarray(scene_image).save(variant_dir / f"{view_label}.png")
                    pair_images[variant][view_label] = scene_image

                    sample_id = f"pair_{pair_id:02d}_{variant}_{view_label}"
                    policy_scene_images.append(np.transpose(scene_image, (2, 0, 1)))
                    policy_wrist_images.append(np.transpose(wrist_image, (2, 0, 1)))
                    policy_states.append(source_proprio.copy())
                    sample_tasks.append(str(task.language))
                    sample_ids.append(sample_id)
                    sample_pair_ids.append(pair_id)
                    sample_trajectory_indices.append(trajectory_index)
                    sample_variants.append(variant)
                    sample_view_labels.append(view_label)
                    sample_yaw_degrees.append(yaw_degrees)
                    sample_records.append(
                        {
                            "sample_index": len(sample_ids) - 1,
                            "sample_id": sample_id,
                            "pair_id": pair_id,
                            "trajectory_state_index": trajectory_index,
                            "variant": variant,
                            "view_label": view_label,
                            "camera_yaw_degrees": yaw_degrees,
                            "scene_png": str(
                                Path("images") / f"pair_{pair_id:02d}" / variant / f"{view_label}.png"
                            ),
                            "wrist_png": str(
                                Path("images") / f"pair_{pair_id:02d}" / variant / "wrist.png"
                            ),
                        }
                    )

            make_contact_sheet(
                contact_sheets_dir / f"pair_{pair_id:02d}.png",
                pair_id,
                trajectory_index,
                pair_images,
            )
            print(
                f"Rendered pair {pair_id + 1:02d}/{len(snapshot_indices)} "
                f"from trajectory state {trajectory_index}",
                flush=True,
            )

        sim.model.cam_pos[camera_id] = nominal_camera_position
        sim.model.cam_quat[camera_id] = nominal_camera_quaternion
        sim.forward()

        np.savez_compressed(
            output_dir / "policy_samples.npz",
            **{
                "observation.images.image": np.stack(policy_scene_images).astype(np.uint8),
                "observation.images.image2": np.stack(policy_wrist_images).astype(np.uint8),
                "observation.state": np.stack(policy_states).astype(np.float32),
                "task": np.asarray(sample_tasks),
                "sample_id": np.asarray(sample_ids),
                "pair_id": np.asarray(sample_pair_ids, dtype=np.int64),
                "trajectory_state_index": np.asarray(sample_trajectory_indices, dtype=np.int64),
                "variant": np.asarray(sample_variants),
                "view_label": np.asarray(sample_view_labels),
                "camera_yaw_degrees": np.asarray(sample_yaw_degrees, dtype=np.float32),
            },
        )
        np.savez_compressed(
            output_dir / "paired_states.npz",
            clean_sim_states=np.stack(clean_sim_states),
            perturbed_sim_states=np.stack(perturbed_sim_states),
            clean_qpos=np.stack(clean_qpos),
            perturbed_qpos=np.stack(perturbed_qpos),
            qvel=np.stack(qvel),
            clean_object_qpos=np.stack(clean_object_qpos),
            perturbed_object_qpos=np.stack(perturbed_object_qpos),
            trajectory_state_indices=snapshot_indices,
            camera_positions=np.stack(camera_positions),
            camera_quaternions=np.stack(camera_quaternions),
            camera_yaw_degrees=np.asarray([yaw for _, yaw in VIEW_SPECS], dtype=np.float64),
            perturb_translation_xyz=translation,
            perturb_yaw_degrees=np.asarray(args.object_yaw_deg, dtype=np.float64),
            orbit_target=orbit_target,
            reference_task_object_position_world=task_position,
            target_object_position_world=target_position,
        )

        manifest = {
            "format_version": 1,
            "source_rollout": str(rollout_dir),
            "suite": suite_name,
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "snapshot_indices": snapshot_indices.tolist(),
            "snapshot_sampling": snapshot_sampling,
            "task_object": TASK_OBJECT,
            "task_object_joint": TASK_OBJECT_JOINT,
            "target_object": TARGET_OBJECT,
            "perturbation": {
                "translation_xyz_metres": translation.tolist(),
                "world_z_yaw_degrees": args.object_yaw_deg,
                "physics_stepped_after_perturbation": False,
            },
            "views": [
                {"label": label, "yaw_degrees": yaw} for label, yaw in VIEW_SPECS
            ],
            "camera_orbit_target_world_xyz": orbit_target.tolist(),
            "image_size": [args.image_size, args.image_size],
            "image_layout": "CHW uint8 in policy_samples.npz; RGB HWC in PNG files",
            "policy_image_range": "divide uint8 arrays by 255 to obtain float32 [0, 1]",
            "policy_keys": [
                "observation.images.image",
                "observation.images.image2",
                "observation.state",
                "task",
            ],
            "num_pairs": len(snapshot_indices),
            "num_policy_samples": len(sample_ids),
            "num_scene_pngs": len(sample_ids),
            "num_wrist_pngs": 2 * len(snapshot_indices),
            "num_contact_sheets": len(snapshot_indices),
            "success_check_used": False,
            "policy_loaded_or_run": False,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "sample_index.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in sample_records),
            encoding="utf-8",
        )
        (output_dir / "README.md").write_text(
            """# LIBERO clean/perturbed multiview samples

`policy_samples.npz` contains unbatched LeRobot-style samples. Images are
compact `uint8` CHW arrays. Convert them to float tensors in `[0, 1]` before
passing them to the frozen policy preprocessor:

```python
import numpy as np
import torch

data = np.load("policy_samples.npz")
i = 0
sample = {
    "observation.images.image": torch.from_numpy(data["observation.images.image"][i]).float() / 255,
    "observation.images.image2": torch.from_numpy(data["observation.images.image2"][i]).float() / 255,
    "observation.state": torch.from_numpy(data["observation.state"][i]),
    "task": str(data["task"][i]),
}
```

`paired_states.npz` contains exact clean/perturbed MuJoCo states, object joint
poses, and camera extrinsics. `images/` contains individual RGB PNGs and
`contact_sheets/` contains clean-vs-perturbed 2x5 comparisons.
""",
            encoding="utf-8",
        )
    finally:
        env.close()

    print(f"Saved {len(sample_ids)} policy samples to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
