#!/usr/bin/env python3
"""Run deterministic frozen pi0.5 inference on LIBERO multiview pairs.

Every sample uses the same explicit flow-matching noise tensor. Before every
forward pass, Python, NumPy, Torch CPU/CUDA RNGs and the policy action queue are
reset. Therefore, model inputs differ only in their saved image tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi05_disagreement_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors


DEFAULT_CHECKPOINT = Path(
    "/mnt/data/nas/hufangchi/cache/huggingface/hub/"
    "models--lerobot--pi05_libero_finetuned/snapshots/"
    "dbf8a3f794a9c4297b44f40b752712f50073d945"
)
PRIMARY_METRIC = "exec10_mean_pairwise_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--executed-horizon", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    return parser.parse_args()


def reset_all_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pairwise_distances(actions: np.ndarray) -> np.ndarray:
    """Return [num_view_pairs, horizon] L2 distances for [views, horizon, dims]."""
    pairs = []
    for left, right in itertools.combinations(range(actions.shape[0]), 2):
        pairs.append(np.linalg.norm(actions[left] - actions[right], axis=-1))
    return np.stack(pairs)


def compute_group_metrics(actions: np.ndarray, executed_horizon: int) -> dict[str, float]:
    executed = actions[:, :executed_horizon]
    full_pairwise = pairwise_distances(actions)
    executed_pairwise = pairwise_distances(executed)
    first_pairwise = pairwise_distances(executed[:, :1])
    arm_pairwise = pairwise_distances(executed[:, :, :6])

    gripper_pairs = []
    per_dimension_pairs = []
    for left, right in itertools.combinations(range(executed.shape[0]), 2):
        difference = np.abs(executed[left] - executed[right])
        gripper_pairs.append(difference[:, 6])
        per_dimension_pairs.append(difference)
    gripper_pairwise = np.stack(gripper_pairs)
    per_dimension_pairwise = np.stack(per_dimension_pairs)

    metrics = {
        "first_action_mean_pairwise_l2": float(first_pairwise.mean()),
        "first_action_max_pairwise_l2": float(first_pairwise.max()),
        "exec10_mean_pairwise_l2": float(executed_pairwise.mean()),
        "exec10_max_pairwise_l2": float(executed_pairwise.max()),
        "exec10_centroid_rmse": float(np.sqrt(np.mean((executed - executed.mean(axis=0)) ** 2))),
        "exec10_arm_mean_pairwise_l2": float(arm_pairwise.mean()),
        "exec10_gripper_mean_pairwise_abs": float(gripper_pairwise.mean()),
        "full50_mean_pairwise_l2": float(full_pairwise.mean()),
        "full50_max_pairwise_l2": float(full_pairwise.max()),
    }
    for dimension in range(actions.shape[-1]):
        metrics[f"exec10_dim{dimension}_mean_pairwise_abs"] = float(
            per_dimension_pairwise[:, :, dimension].mean()
        )
    return metrics


def make_sign_flip_matrix(num_pairs: int, seed: int) -> tuple[np.ndarray, str]:
    """Build exact signs for small n, otherwise fixed-seed Monte Carlo signs."""
    if num_pairs <= 20:
        signs = np.asarray(
            list(itertools.product((-1.0, 1.0), repeat=num_pairs)), dtype=np.float64
        )
        return signs, f"exact_all_{len(signs)}_sign_flips"
    num_randomizations = 100_000
    generator = np.random.default_rng(seed)
    signs = generator.choice(
        np.asarray([-1, 1], dtype=np.int8),
        size=(num_randomizations, num_pairs),
        replace=True,
    )
    return signs, f"fixed_seed_monte_carlo_{num_randomizations}_sign_flips"


def paired_sign_flip_pvalues(
    differences: np.ndarray, sign_matrix: np.ndarray
) -> tuple[float, float]:
    """Paired randomization p-values: two-sided and perturbed>clean."""
    observed = float(differences.mean())
    null = (sign_matrix @ differences) / len(differences)
    tolerance = 1e-15
    two_sided = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    greater = float(np.mean(null >= observed - tolerance))
    return two_sided, greater


def validate_controlled_inputs(data: dict[str, np.ndarray]) -> dict[str, Any]:
    pair_ids = data["pair_id"]
    variants = data["variant"]
    states = data["observation.state"]
    tasks = data["task"]
    wrists = data["observation.images.image2"]
    scene_images = data["observation.images.image"]
    view_labels = data["view_label"]
    yaw = data["camera_yaw_degrees"]

    unique_pairs = np.unique(pair_ids)
    if unique_pairs.tolist() != list(range(len(unique_pairs))):
        raise ValueError(f"Expected contiguous pair ids, got {unique_pairs.tolist()}")

    for pair_id in unique_pairs:
        indices = np.flatnonzero(pair_ids == pair_id)
        if len(indices) != 10:
            raise ValueError(f"Pair {pair_id} has {len(indices)} samples, expected 10")
        if not np.array_equal(states[indices], np.repeat(states[indices[:1]], 10, axis=0)):
            raise ValueError(f"State is not invariant within pair {pair_id}")
        if len(set(tasks[indices].tolist())) != 1:
            raise ValueError(f"Task language is not invariant within pair {pair_id}")

        clean = indices[variants[indices] == "clean"]
        perturbed = indices[variants[indices] == "perturbed"]
        if len(clean) != 5 or len(perturbed) != 5:
            raise ValueError(f"Pair {pair_id} does not have 5 clean and 5 perturbed samples")
        if not np.array_equal(view_labels[clean], view_labels[perturbed]):
            raise ValueError(f"View labels are not paired for pair {pair_id}")
        if not np.array_equal(yaw[clean], yaw[perturbed]):
            raise ValueError(f"Camera yaw values are not paired for pair {pair_id}")
        if not np.array_equal(wrists[clean], np.repeat(wrists[clean[:1]], 5, axis=0)):
            raise ValueError(f"Clean wrist image changes across views for pair {pair_id}")
        if not np.array_equal(wrists[perturbed], np.repeat(wrists[perturbed[:1]], 5, axis=0)):
            raise ValueError(f"Perturbed wrist image changes across views for pair {pair_id}")
        if len({sha256_array(scene_images[index]) for index in clean}) != 5:
            raise ValueError(f"Clean scene views are not all distinct for pair {pair_id}")
        if len({sha256_array(scene_images[index]) for index in perturbed}) != 5:
            raise ValueError(f"Perturbed scene views are not all distinct for pair {pair_id}")

    return {
        "num_pairs": int(len(unique_pairs)),
        "samples_per_pair": 10,
        "views_per_variant": 5,
        "state_identical_within_each_pair": True,
        "task_identical_within_each_pair": True,
        "wrist_identical_across_views_within_each_variant": True,
        "view_labels_and_camera_yaws_paired_clean_vs_perturbed": True,
        "all_scene_views_distinct_within_each_variant": True,
    }


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def make_unbatched_sample(data: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "observation.images.image": torch.from_numpy(
            data["observation.images.image"][index].copy()
        ).to(torch.float32).div_(255.0),
        "observation.images.image2": torch.from_numpy(
            data["observation.images.image2"][index].copy()
        ).to(torch.float32).div_(255.0),
        "observation.state": torch.from_numpy(data["observation.state"][index].copy()).to(
            torch.float32
        ),
        "task": str(data["task"][index]),
    }


def run_one_sample(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    sample: dict[str, Any],
    fixed_noise: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    reset_all_rngs(seed)
    policy.reset()
    processed = preprocessor(sample)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(processed, noise=fixed_noise.clone())
        actions = postprocessor(normalized)
    return (
        normalized.detach().to(torch.float32).cpu().numpy()[0].copy(),
        actions.detach().to(torch.float32).cpu().numpy()[0].copy(),
    )


def save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    trajectory_indices: np.ndarray,
    clean_values: np.ndarray,
    perturbed_values: np.ndarray,
) -> None:
    x = np.arange(len(trajectory_indices))
    width = 0.38
    ratios = perturbed_values / np.maximum(clean_values, 1e-12)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(x - width / 2, clean_values, width, label="clean", color="#4C78A8")
    axes[0].bar(x + width / 2, perturbed_values, width, label="perturbed", color="#E45756")
    axes[0].set_ylabel("Mean pairwise action L2")
    axes[0].set_title("Cross-view disagreement over first 10 action steps")
    axes[0].set_xticks(x, [str(value) for value in trajectory_indices])
    axes[0].set_xlabel("Trajectory state index")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#59A14F" if value > 1.0 else "#9C755F" for value in ratios]
    axes[1].bar(x, ratios, color=colors)
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Perturbed / clean")
    axes[1].set_xlabel("Trajectory state index")
    axes[1].set_xticks(x, [str(value) for value in trajectory_indices])
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_paired_scatter(
    path: Path,
    trajectory_indices: np.ndarray,
    clean_values: np.ndarray,
    perturbed_values: np.ndarray,
) -> None:
    lower = float(min(clean_values.min(), perturbed_values.min()))
    upper = float(max(clean_values.max(), perturbed_values.max()))
    padding = max((upper - lower) * 0.08, 0.01)
    limits = (max(0.0, lower - padding), upper + padding)
    counterexamples = perturbed_values < clean_values

    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    scatter = axis.scatter(
        clean_values,
        perturbed_values,
        c=trajectory_indices,
        cmap="viridis",
        s=72,
        edgecolors=np.where(counterexamples, "#D62728", "white"),
        linewidths=np.where(counterexamples, 2.2, 0.8),
        alpha=0.92,
    )
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.4, label="y = x")
    for state_index, clean, perturbed, is_counterexample in zip(
        trajectory_indices, clean_values, perturbed_values, counterexamples, strict=True
    ):
        if is_counterexample:
            axis.annotate(
                str(int(state_index)),
                (clean, perturbed),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                color="#B22222",
            )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Clean disagreement")
    axis.set_ylabel("Perturbed disagreement")
    axis.set_title("Paired cross-view disagreement by snapshot")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Trajectory state index")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Dataset or manifest is missing")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pi0.5 inference")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reset_all_rngs(args.seed)

    data = load_dataset(dataset_path)
    required_keys = {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "task",
        "pair_id",
        "trajectory_state_index",
        "variant",
        "view_label",
        "camera_yaw_degrees",
    }
    missing = required_keys.difference(data)
    if missing:
        raise KeyError(f"Dataset is missing keys: {sorted(missing)}")
    control_checks = validate_controlled_inputs(data)
    num_pairs = int(control_checks["num_pairs"])
    num_samples = len(data["task"])
    if num_samples != num_pairs * 10:
        raise ValueError(f"Expected 10 samples per pair, got {num_samples} over {num_pairs} pairs")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = "cuda"
    policy_cfg.n_action_steps = args.executed_horizon
    policy_cfg.num_inference_steps = args.num_inference_steps
    policy_cfg.gradient_checkpointing = False
    policy_cfg.compile_model = False

    env_cfg = LiberoEnvConfig(
        task=str(source_manifest["suite"]),
        task_ids=[int(source_manifest["task_id"])],
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        init_states=True,
    )

    print(f"Loading frozen checkpoint: {checkpoint}", flush=True)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    policy.requires_grad_(False)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("Policy contains trainable parameters after freezing")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print("Checkpoint loaded and all parameters frozen.", flush=True)

    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(args.seed)
    fixed_noise_cpu = torch.randn(
        (1, policy_cfg.chunk_size, policy_cfg.max_action_dim),
        generator=noise_generator,
        dtype=torch.float32,
        device="cpu",
    )
    fixed_noise = fixed_noise_cpu.to("cuda")

    normalized_action_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    inference_seconds: list[float] = []
    first_repeat_normalized: np.ndarray | None = None
    first_repeat_actions: np.ndarray | None = None
    experiment_start = time.perf_counter()

    for index in range(num_samples):
        sample = make_unbatched_sample(data, index)
        torch.cuda.synchronize()
        start = time.perf_counter()
        normalized, actions = run_one_sample(
            policy,
            preprocessor,
            postprocessor,
            sample,
            fixed_noise,
            args.seed,
        )
        torch.cuda.synchronize()
        inference_seconds.append(time.perf_counter() - start)
        normalized_action_chunks.append(normalized)
        action_chunks.append(actions)
        if index == 0:
            first_repeat_normalized = normalized.copy()
            first_repeat_actions = actions.copy()
        if (index + 1) % 25 == 0 or index + 1 == num_samples:
            print(
                f"Inference {index + 1:03d}/{num_samples} | "
                f"last={inference_seconds[-1]:.3f}s mean={np.mean(inference_seconds):.3f}s",
                flush=True,
            )

    if first_repeat_normalized is None or first_repeat_actions is None:
        raise RuntimeError("No inference output was produced")
    repeated_normalized, repeated_actions = run_one_sample(
        policy,
        preprocessor,
        postprocessor,
        make_unbatched_sample(data, 0),
        fixed_noise,
        args.seed,
    )
    normalized_repeat_max_abs = float(
        np.max(np.abs(first_repeat_normalized - repeated_normalized))
    )
    action_repeat_max_abs = float(np.max(np.abs(first_repeat_actions - repeated_actions)))
    if normalized_repeat_max_abs != 0.0 or action_repeat_max_abs != 0.0:
        raise RuntimeError(
            "Determinism check failed: "
            f"normalized={normalized_repeat_max_abs}, postprocessed={action_repeat_max_abs}"
        )

    normalized_array = np.stack(normalized_action_chunks).astype(np.float32)
    action_array = np.stack(action_chunks).astype(np.float32)
    if normalized_array.shape != (num_samples, policy_cfg.chunk_size, 7):
        raise RuntimeError(f"Unexpected normalized action shape: {normalized_array.shape}")
    if action_array.shape != (num_samples, policy_cfg.chunk_size, 7):
        raise RuntimeError(f"Unexpected postprocessed action shape: {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise RuntimeError("Non-finite values found in action outputs")

    metric_rows: list[dict[str, Any]] = []
    for pair_id in range(num_pairs):
        pair_indices = np.flatnonzero(data["pair_id"] == pair_id)
        for variant in ("clean", "perturbed"):
            indices = pair_indices[data["variant"][pair_indices] == variant]
            metrics = compute_group_metrics(action_array[indices], args.executed_horizon)
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "trajectory_state_index": int(data["trajectory_state_index"][indices[0]]),
                    "variant": variant,
                    **metrics,
                }
            )

    metric_names = [
        key for key in metric_rows[0] if key not in {"pair_id", "trajectory_state_index", "variant"}
    ]
    paired_statistics: dict[str, Any] = {}
    sign_matrix, sign_flip_method = make_sign_flip_matrix(num_pairs, args.seed)
    for metric_name in metric_names:
        clean = np.asarray(
            [row[metric_name] for row in metric_rows if row["variant"] == "clean"],
            dtype=np.float64,
        )
        perturbed = np.asarray(
            [row[metric_name] for row in metric_rows if row["variant"] == "perturbed"],
            dtype=np.float64,
        )
        differences = perturbed - clean
        two_sided_p, greater_p = paired_sign_flip_pvalues(differences, sign_matrix)
        paired_statistics[metric_name] = {
            "clean_mean": float(clean.mean()),
            "clean_std": float(clean.std(ddof=1)),
            "perturbed_mean": float(perturbed.mean()),
            "perturbed_std": float(perturbed.std(ddof=1)),
            "mean_paired_difference_perturbed_minus_clean": float(differences.mean()),
            "mean_ratio_perturbed_over_clean": float(
                np.mean(perturbed / np.maximum(clean, 1e-12))
            ),
            "median_ratio_perturbed_over_clean": float(
                np.median(perturbed / np.maximum(clean, 1e-12))
            ),
            "num_pairs_perturbed_greater": int(np.sum(differences > 0.0)),
            "paired_sign_flip_method": sign_flip_method,
            "paired_sign_flip_two_sided_p": two_sided_p,
            "paired_sign_flip_one_sided_perturbed_greater_p": greater_p,
        }

    primary = paired_statistics[PRIMARY_METRIC]
    if primary["mean_paired_difference_perturbed_minus_clean"] > 0:
        direction = "perturbed_higher"
    elif primary["mean_paired_difference_perturbed_minus_clean"] < 0:
        direction = "perturbed_lower"
    else:
        direction = "equal"

    output_dir.mkdir(parents=True)
    action_output_arrays = {
        "normalized_action_chunks": normalized_array,
        "action_chunks": action_array,
        "executed_action_chunks": action_array[:, : args.executed_horizon],
        "fixed_noise": fixed_noise_cpu.numpy(),
        "sample_id": data["sample_id"],
        "pair_id": data["pair_id"],
        "trajectory_state_index": data["trajectory_state_index"],
        "variant": data["variant"],
        "view_label": data["view_label"],
        "camera_yaw_degrees": data["camera_yaw_degrees"],
    }
    for optional_key in ("task_id", "episode_id", "episode_key", "episode_pair_id"):
        if optional_key in data:
            action_output_arrays[optional_key] = data[optional_key]
    np.savez_compressed(output_dir / "action_outputs.npz", **action_output_arrays)
    save_metrics_csv(output_dir / "per_group_metrics.csv", metric_rows)

    clean_primary = np.asarray(
        [row[PRIMARY_METRIC] for row in metric_rows if row["variant"] == "clean"]
    )
    perturbed_primary = np.asarray(
        [row[PRIMARY_METRIC] for row in metric_rows if row["variant"] == "perturbed"]
    )
    trajectory_indices = np.asarray(
        [row["trajectory_state_index"] for row in metric_rows if row["variant"] == "clean"]
    )
    save_plot(
        output_dir / "cross_view_disagreement.png",
        trajectory_indices,
        clean_primary,
        perturbed_primary,
    )
    save_paired_scatter(
        output_dir / "paired_scatter.png",
        trajectory_indices,
        clean_primary,
        perturbed_primary,
    )

    result = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_snapshot_commit": checkpoint.name,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "source_manifest": str(manifest_path),
        "seed": args.seed,
        "fixed_noise_sha256": sha256_array(fixed_noise_cpu.numpy()),
        "determinism": {
            "python_numpy_torch_cpu_cuda_reset_before_every_sample": True,
            "explicit_identical_noise_passed_to_every_sample": True,
            "policy_reset_before_every_sample": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "repeat_sample_index": 0,
            "normalized_action_repeat_max_abs_difference": normalized_repeat_max_abs,
            "postprocessed_action_repeat_max_abs_difference": action_repeat_max_abs,
            "bitwise_repeat_passed": True,
        },
        "input_control_checks": control_checks,
        "model": {
            "parameters_frozen": True,
            "eval_mode": True,
            "compile_model": False,
            "gradient_checkpointing": False,
            "dtype": policy_cfg.dtype,
            "chunk_size": policy_cfg.chunk_size,
            "executed_horizon": args.executed_horizon,
            "action_dimension": action_array.shape[-1],
            "num_inference_steps": args.num_inference_steps,
        },
        "num_samples": num_samples,
        "num_groups": 2 * num_pairs,
        "num_snapshot_pairs": num_pairs,
        "primary_metric": PRIMARY_METRIC,
        "primary_result_direction": direction,
        "primary_statistics": primary,
        "paired_statistics": paired_statistics,
        "timing": {
            "total_experiment_seconds": float(time.perf_counter() - experiment_start),
            "mean_inference_seconds": float(np.mean(inference_seconds)),
            "median_inference_seconds": float(np.median(inference_seconds)),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Deterministic pi0.5 cross-view disagreement experiment",
                "",
                f"Primary metric: `{PRIMARY_METRIC}`.",
                "It is the mean pairwise 7D action L2 distance across all 10 camera-view",
                "pairs and the first 10 action steps, after checkpoint postprocessing.",
                "",
                f"- Clean mean: {primary['clean_mean']:.8f}",
                f"- Perturbed mean: {primary['perturbed_mean']:.8f}",
                f"- Mean paired difference: {primary['mean_paired_difference_perturbed_minus_clean']:.8f}",
                f"- Mean perturbed/clean ratio: {primary['mean_ratio_perturbed_over_clean']:.6f}",
                f"- Perturbed greater in: {primary['num_pairs_perturbed_greater']}/{num_pairs} snapshots",
                f"- Paired sign-flip method: {primary['paired_sign_flip_method']}",
                f"- Paired sign-flip p (two-sided): {primary['paired_sign_flip_two_sided_p']:.8f}",
                f"- Paired sign-flip p (perturbed > clean): {primary['paired_sign_flip_one_sided_perturbed_greater_p']:.8f}",
                "- Repeated sample max absolute action difference: 0.0",
                "",
                "Raw 50-step and executed 10-step action chunks are in `action_outputs.npz`.",
                "Per-group metrics are in `per_group_metrics.csv`; full settings and all",
                "metric statistics are in `summary.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"primary_metric": PRIMARY_METRIC, **primary}, indent=2), flush=True)
    print(f"Saved final experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
