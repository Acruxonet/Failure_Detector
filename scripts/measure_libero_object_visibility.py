#!/usr/bin/env python3
"""Measure task-object visibility for saved LIBERO clean/perturbed views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from libero.libero import benchmark

from lerobot.envs.libero import LiberoEnv

from render_libero_multiview_pairs import TASK_OBJECT, restore_state_without_success_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def visible_geom_pixels(sim, camera_name: str, size: int, geom_ids: np.ndarray) -> int:
    # This mirrors robosuite's segmentation readback, except that RGB channels
    # are explicitly widened before multiplying by 256 / 65536. NumPy 2 raises
    # on the original uint8 overflow expression in robosuite 1.4.x.
    context = sim._render_context_offscreen
    camera_id = int(sim.model.camera_name2id(camera_name))
    context.render(width=size, height=size, camera_id=camera_id, segmentation=True)
    viewport = mujoco.MjrRect(0, 0, size, size)
    encoded = np.empty((size, size, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb=encoded, depth=None, viewport=viewport, con=context.con)
    widened = encoded.astype(np.int32)
    segmentation_ids = (
        widened[:, :, 0] + widened[:, :, 1] * (2**8) + widened[:, :, 2] * (2**16)
    )
    segmentation_ids[segmentation_ids >= (context.scn.ngeom + 1)] = 0
    id_lookup = np.full((context.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
    for index in range(context.scn.ngeom):
        geom = context.scn.geoms[index]
        if geom.segid != -1:
            id_lookup[geom.segid + 1, 0] = geom.objtype
            id_lookup[geom.segid + 1, 1] = geom.objid
    segmentation = id_lookup[segmentation_ids]
    if segmentation.shape != (size, size, 2):
        raise RuntimeError(f"Unexpected segmentation shape: {segmentation.shape}")
    return int(np.count_nonzero(np.isin(segmentation[..., 1], geom_ids)))


def main() -> None:
    args = parse_args()
    pairs_dir = args.pairs_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    manifest = json.loads((pairs_dir / "manifest.json").read_text(encoding="utf-8"))
    states = np.load(pairs_dir / "paired_states.npz", allow_pickle=False)
    num_pairs = len(states["trajectory_state_indices"])

    suite_name = str(manifest["suite"])
    task_id = int(manifest["task_id"])
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
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

    scene_pixels = np.zeros((num_pairs, 2, 5), dtype=np.int64)
    wrist_pixels = np.zeros((num_pairs, 2), dtype=np.int64)
    try:
        env.reset(seed=1000)
        if env._env is None:
            raise RuntimeError("LIBERO environment was not initialized")
        base_env = env._env
        sim = base_env.sim
        inner_env = base_env.env
        target_geom_ids = np.asarray(
            sorted(
                geom_id
                for geom_id, instance in inner_env.model.geom_ids_to_instances.items()
                if instance == TASK_OBJECT
            ),
            dtype=np.int64,
        )
        if not len(target_geom_ids):
            raise RuntimeError(f"No geom IDs found for {TASK_OBJECT}")

        camera_id = int(sim.model.camera_name2id("agentview"))
        camera_positions = states["camera_positions"]
        camera_quaternions = states["camera_quaternions"]
        for pair_id in range(num_pairs):
            for variant_index, state_key in enumerate(
                ("clean_sim_states", "perturbed_sim_states")
            ):
                restore_state_without_success_check(base_env, states[state_key][pair_id])
                wrist_pixels[pair_id, variant_index] = visible_geom_pixels(
                    sim, "robot0_eye_in_hand", args.image_size, target_geom_ids
                )
                for view_index in range(5):
                    sim.model.cam_pos[camera_id] = camera_positions[view_index]
                    sim.model.cam_quat[camera_id] = camera_quaternions[view_index]
                    sim.forward()
                    scene_pixels[pair_id, variant_index, view_index] = visible_geom_pixels(
                        sim, "agentview", args.image_size, target_geom_ids
                    )
            if (pair_id + 1) % 10 == 0 or pair_id + 1 == num_pairs:
                print(f"Visibility {pair_id + 1:02d}/{num_pairs}", flush=True)
    finally:
        env.close()

    if not np.any(scene_pixels):
        raise RuntimeError("Target object was absent from every scene segmentation")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scene_visible_pixels=scene_pixels,
        wrist_visible_pixels=wrist_pixels,
        trajectory_state_indices=states["trajectory_state_indices"],
        target_geom_ids=target_geom_ids,
        variant=np.asarray(["clean", "perturbed"]),
        view_label=np.asarray([view["label"] for view in manifest["views"]]),
    )
    print(f"Saved visibility metrics to {output}", flush=True)


if __name__ == "__main__":
    main()
