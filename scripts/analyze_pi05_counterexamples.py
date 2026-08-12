#!/usr/bin/env python3
"""Analyze stage dependence, visibility, and per-view drivers of disagreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi05_counterexample_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRIMARY_METRIC = "exec10_mean_pairwise_l2"
STAGE_ORDER = (
    "far_approach",
    "close_approach",
    "grasp_contact",
    "lift_transport",
    "place_alignment",
)
CRITICAL_STAGES = {"close_approach", "grasp_contact", "lift_transport"}
STAGE_COLORS = {
    "far_approach": "#4C78A8",
    "close_approach": "#F28E2B",
    "grasp_contact": "#E15759",
    "lift_transport": "#59A14F",
    "place_alignment": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--old-experiment-dir", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_stage(
    object_position: np.ndarray,
    eef_position: np.ndarray,
    target_position: np.ndarray,
    table_object_z: float,
) -> tuple[str, float, float, float]:
    eef_object_distance = float(np.linalg.norm(eef_position - object_position))
    lift_height = float(object_position[2] - table_object_z)
    target_xy_distance = float(np.linalg.norm(object_position[:2] - target_position[:2]))
    if eef_object_distance > 0.115:
        stage = "far_approach"
    elif lift_height < 0.003 and eef_object_distance > 0.055:
        stage = "close_approach"
    elif lift_height < 0.003:
        stage = "grasp_contact"
    elif target_xy_distance > 0.06:
        stage = "lift_transport"
    else:
        stage = "place_alignment"
    return stage, eef_object_distance, lift_height, target_xy_distance


def view_distance_to_others(actions: np.ndarray) -> np.ndarray:
    result = np.zeros(actions.shape[0], dtype=np.float64)
    for view in range(actions.shape[0]):
        distances = []
        for other in range(actions.shape[0]):
            if other != view:
                distances.append(np.linalg.norm(actions[view] - actions[other], axis=-1))
        result[view] = float(np.stack(distances).mean())
    return result


def binomial_greater_p(num_positive: int, total: int) -> float:
    return float(sum(math.comb(total, value) for value in range(num_positive, total + 1)) / 2**total)


def fisher_critical_enrichment_p(
    critical_positive: int,
    critical_total: int,
    all_positive: int,
    all_total: int,
) -> float:
    lower = critical_positive
    upper = min(critical_total, all_positive)
    denominator = math.comb(all_total, critical_total)
    return float(
        sum(
            math.comb(all_positive, value)
            * math.comb(all_total - all_positive, critical_total - value)
            for value in range(lower, upper + 1)
        )
        / denominator
    )


def save_stage_scatter(path: Path, rows: list[dict[str, Any]]) -> None:
    clean = np.asarray([row["clean_disagreement"] for row in rows])
    perturbed = np.asarray([row["perturbed_disagreement"] for row in rows])
    lower = float(min(clean.min(), perturbed.min()))
    upper = float(max(clean.max(), perturbed.max()))
    padding = max((upper - lower) * 0.08, 0.01)
    limits = (max(0.0, lower - padding), upper + padding)

    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for stage in STAGE_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        x = np.asarray([row["clean_disagreement"] for row in stage_rows])
        y = np.asarray([row["perturbed_disagreement"] for row in stage_rows])
        counterexample = y < x
        axis.scatter(
            x,
            y,
            label=f"{stage} (n={len(stage_rows)})",
            color=STAGE_COLORS[stage],
            s=70,
            edgecolors=np.where(counterexample, "#B22222", "white"),
            linewidths=np.where(counterexample, 2.0, 0.7),
            alpha=0.92,
        )
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.4, label="y = x")
    for row in rows:
        if row["disagreement_delta"] < -0.04:
            axis.annotate(
                str(row["trajectory_state_index"]),
                (row["clean_disagreement"], row["perturbed_disagreement"]),
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
    axis.set_title("Paired disagreement reveals strong task-stage dependence")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_counterexample_view_profiles(
    path: Path,
    snapshot_rows: list[dict[str, Any]],
    per_view_rows: list[dict[str, Any]],
) -> None:
    worst = sorted(
        (row for row in snapshot_rows if row["counterexample"]),
        key=lambda row: row["disagreement_delta"],
    )[:6]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    x = np.arange(5)
    width = 0.38
    for axis, snapshot in zip(axes.flat, worst, strict=True):
        state_index = snapshot["trajectory_state_index"]
        rows = [
            row for row in per_view_rows if row["trajectory_state_index"] == state_index
        ]
        clean = np.asarray([row["clean_distance_to_other_views"] for row in rows])
        perturbed = np.asarray([row["perturbed_distance_to_other_views"] for row in rows])
        pixel_gain = np.asarray([row["visible_object_pixel_gain"] for row in rows])
        axis.bar(x - width / 2, clean, width, label="clean", color="#4C78A8")
        axis.bar(x + width / 2, perturbed, width, label="perturbed", color="#E45756")
        for view_index, gain in enumerate(pixel_gain):
            axis.text(
                view_index,
                max(clean[view_index], perturbed[view_index]) + 0.012,
                f"{gain:+.0f}px",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
        axis.set_title(
            f"state {state_index} | {snapshot['stage']}\n"
            f"paired delta {snapshot['disagreement_delta']:+.3f}"
        )
        axis.set_xticks(x, ["nom", "-30", "-15", "+15", "+30"])
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].set_ylabel("Distance to other-view actions")
    axes[1, 0].set_ylabel("Distance to other-view actions")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Strongest counterexamples: per-view action dispersion\n"
        "labels show perturbed-minus-clean visible bowl pixels",
        fontsize=15,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pairs_dir = args.pairs_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    paired_states = np.load(pairs_dir / "paired_states.npz", allow_pickle=False)
    visibility = np.load(experiment_dir / "object_visibility.npz", allow_pickle=False)
    actions = np.load(experiment_dir / "action_outputs.npz", allow_pickle=False)
    metric_rows = read_csv(experiment_dir / "per_group_metrics.csv")
    metric_lookup = {
        (int(row["pair_id"]), row["variant"]): float(row[PRIMARY_METRIC])
        for row in metric_rows
    }
    with np.load(pairs_dir / "policy_samples.npz", allow_pickle=False) as policy_data:
        policy_states = policy_data["observation.state"][::10].copy()
        scene_images = policy_data["observation.images.image"].copy()

    num_pairs = len(paired_states["trajectory_state_indices"])
    if num_pairs != 50:
        raise ValueError(f"Expected 50 pairs, got {num_pairs}")
    view_labels = visibility["view_label"].astype(str)
    object_positions = paired_states["clean_object_qpos"][:, :3]
    target_position = paired_states["target_object_position_world"]
    table_object_z = float(np.median(object_positions[:8, 2]))
    action_chunks = actions["executed_action_chunks"]

    snapshot_rows: list[dict[str, Any]] = []
    per_view_rows: list[dict[str, Any]] = []
    for pair_id in range(num_pairs):
        indices = np.flatnonzero(actions["pair_id"] == pair_id)
        clean_indices = indices[actions["variant"][indices] == "clean"]
        perturbed_indices = indices[actions["variant"][indices] == "perturbed"]
        clean_actions = action_chunks[clean_indices]
        perturbed_actions = action_chunks[perturbed_indices]
        clean_view_disagreement = view_distance_to_others(clean_actions)
        perturbed_view_disagreement = view_distance_to_others(perturbed_actions)
        matched_action_response = np.linalg.norm(
            perturbed_actions - clean_actions, axis=-1
        ).mean(axis=1)

        clean_scene = scene_images[clean_indices].astype(np.int16)
        perturbed_scene = scene_images[perturbed_indices].astype(np.int16)
        rgb_difference = np.abs(perturbed_scene - clean_scene)
        rgb_change_fraction = np.max(rgb_difference, axis=1).reshape(5, -1).mean(axis=1) / 255.0
        pixel_fraction_over_10 = (
            np.max(rgb_difference, axis=1).reshape(5, -1) > 10
        ).mean(axis=1)
        visible_clean = visibility["scene_visible_pixels"][pair_id, 0].astype(np.float64)
        visible_perturbed = visibility["scene_visible_pixels"][pair_id, 1].astype(np.float64)
        visible_gain = visible_perturbed - visible_clean

        stage, eef_distance, lift_height, target_xy_distance = classify_stage(
            object_positions[pair_id],
            policy_states[pair_id, :3],
            target_position,
            table_object_z,
        )
        clean_disagreement = metric_lookup[(pair_id, "clean")]
        perturbed_disagreement = metric_lookup[(pair_id, "perturbed")]
        delta = perturbed_disagreement - clean_disagreement

        for view_index, view_label in enumerate(view_labels):
            per_view_rows.append(
                {
                    "pair_id": pair_id,
                    "trajectory_state_index": int(
                        paired_states["trajectory_state_indices"][pair_id]
                    ),
                    "stage": stage,
                    "counterexample": delta < 0.0,
                    "view_label": view_label,
                    "clean_distance_to_other_views": clean_view_disagreement[view_index],
                    "perturbed_distance_to_other_views": perturbed_view_disagreement[view_index],
                    "matched_clean_perturbed_action_response": matched_action_response[view_index],
                    "clean_visible_object_pixels": int(visible_clean[view_index]),
                    "perturbed_visible_object_pixels": int(visible_perturbed[view_index]),
                    "visible_object_pixel_gain": int(visible_gain[view_index]),
                    "rgb_mean_absolute_change_fraction": rgb_change_fraction[view_index],
                    "rgb_pixel_fraction_change_over_10": pixel_fraction_over_10[view_index],
                }
            )

        snapshot_rows.append(
            {
                "pair_id": pair_id,
                "trajectory_state_index": int(paired_states["trajectory_state_indices"][pair_id]),
                "stage": stage,
                "manipulation_critical": stage in CRITICAL_STAGES,
                "clean_disagreement": clean_disagreement,
                "perturbed_disagreement": perturbed_disagreement,
                "disagreement_delta": delta,
                "perturbed_over_clean_ratio": perturbed_disagreement / clean_disagreement,
                "counterexample": delta < 0.0,
                "eef_object_distance_m": eef_distance,
                "object_lift_height_m": lift_height,
                "object_target_xy_distance_m": target_xy_distance,
                "clean_visible_pixels_mean": float(visible_clean.mean()),
                "perturbed_visible_pixels_mean": float(visible_perturbed.mean()),
                "visibility_gain_fraction": float(
                    visible_perturbed.mean() / max(visible_clean.mean(), 1.0) - 1.0
                ),
                "clean_min_visibility_view": str(view_labels[int(np.argmin(visible_clean))]),
                "largest_visibility_gain_view": str(view_labels[int(np.argmax(visible_gain))]),
                "largest_rgb_change_view": str(view_labels[int(np.argmax(rgb_change_fraction))]),
                "clean_max_action_disagreement_view": str(
                    view_labels[int(np.argmax(clean_view_disagreement))]
                ),
                "perturbed_max_action_disagreement_view": str(
                    view_labels[int(np.argmax(perturbed_view_disagreement))]
                ),
                "largest_matched_action_response_view": str(
                    view_labels[int(np.argmax(matched_action_response))]
                ),
                "rgb_change_fraction_mean": float(rgb_change_fraction.mean()),
                "rgb_pixel_fraction_change_over_10_mean": float(
                    pixel_fraction_over_10.mean()
                ),
            }
        )

    stage_rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        rows = [row for row in snapshot_rows if row["stage"] == stage]
        deltas = np.asarray([row["disagreement_delta"] for row in rows])
        clean = np.asarray([row["clean_disagreement"] for row in rows])
        perturbed = np.asarray([row["perturbed_disagreement"] for row in rows])
        positive = int(np.sum(deltas > 0))
        stage_rows.append(
            {
                "stage": stage,
                "manipulation_critical": stage in CRITICAL_STAGES,
                "num_snapshots": len(rows),
                "num_perturbed_greater": positive,
                "fraction_perturbed_greater": positive / len(rows),
                "clean_mean": float(clean.mean()),
                "perturbed_mean": float(perturbed.mean()),
                "mean_paired_delta": float(deltas.mean()),
                "mean_visibility_gain_fraction": float(
                    np.mean([row["visibility_gain_fraction"] for row in rows])
                ),
                "one_sided_sign_test_p": binomial_greater_p(positive, len(rows)),
            }
        )

    critical = [row for row in snapshot_rows if row["manipulation_critical"]]
    noncritical = [row for row in snapshot_rows if not row["manipulation_critical"]]
    critical_positive = sum(not row["counterexample"] for row in critical)
    noncritical_positive = sum(not row["counterexample"] for row in noncritical)
    all_positive = critical_positive + noncritical_positive
    fisher_p = fisher_critical_enrichment_p(
        critical_positive, len(critical), all_positive, len(snapshot_rows)
    )

    write_csv(experiment_dir / "snapshot_diagnostics.csv", snapshot_rows)
    write_csv(experiment_dir / "per_view_diagnostics.csv", per_view_rows)
    write_csv(experiment_dir / "stage_summary.csv", stage_rows)
    counterexamples = [row for row in snapshot_rows if row["counterexample"]]
    write_csv(experiment_dir / "counterexamples.csv", counterexamples)
    save_stage_scatter(experiment_dir / "paired_scatter_by_stage.png", snapshot_rows)
    save_counterexample_view_profiles(
        experiment_dir / "counterexample_view_profiles.png",
        snapshot_rows,
        per_view_rows,
    )

    old_counterexample_lines: list[str] = []
    if args.old_experiment_dir:
        old_rows = read_csv(args.old_experiment_dir.resolve() / "per_group_metrics.csv")
        old_lookup = {
            (int(row["trajectory_state_index"]), row["variant"]): float(row[PRIMARY_METRIC])
            for row in old_rows
        }
        old_states = sorted({key[0] for key in old_lookup})
        for state_index in old_states:
            clean = old_lookup[(state_index, "clean")]
            perturbed = old_lookup[(state_index, "perturbed")]
            if perturbed >= clean:
                continue
            exact = next(
                (row for row in snapshot_rows if row["trajectory_state_index"] == state_index),
                None,
            )
            if exact is not None:
                context = (
                    f"dense run stage `{exact['stage']}`, dense delta "
                    f"{exact['disagreement_delta']:+.5f}, visibility gain "
                    f"{100 * exact['visibility_gain_fraction']:+.1f}%"
                )
            else:
                nearest = sorted(
                    snapshot_rows,
                    key=lambda row: abs(row["trajectory_state_index"] - state_index),
                )[:2]
                context = "not resampled; neighboring dense deltas " + ", ".join(
                    f"state {row['trajectory_state_index']}: {row['disagreement_delta']:+.5f}"
                    for row in nearest
                )
            old_counterexample_lines.append(
                f"- State {state_index}: old delta {perturbed - clean:+.5f}; {context}."
            )

    report_lines = [
        "# Counterexample and task-stage analysis",
        "",
        "Stage classification is kinematic and fixed by these thresholds: far approach",
        "uses end-effector/object distance >11.5 cm; close approach ends at 5.5 cm;",
        "grasp contact requires <3 mm object lift; lift/transport continues until the",
        "object is within 6 cm XY of the plate; the remainder is place alignment.",
        "",
        "## Main stage result",
        "",
        f"- All snapshots: {all_positive}/{len(snapshot_rows)} have perturbed > clean.",
        f"- Manipulation-critical stages: {critical_positive}/{len(critical)} have perturbed > clean; "
        f"mean paired delta {np.mean([row['disagreement_delta'] for row in critical]):+.6f}.",
        f"- Noncritical stages: {noncritical_positive}/{len(noncritical)} have perturbed > clean; "
        f"mean paired delta {np.mean([row['disagreement_delta'] for row in noncritical]):+.6f}.",
        f"- Exploratory one-sided Fisher exact p for enrichment in critical stages: {fisher_p:.8g}.",
        "",
        "This stage split is exploratory / post-hoc and should be confirmed on new tasks and",
        "rollouts before being treated as a general detector result.",
        "",
        "## Original 3/10 counterexamples",
        "",
        *old_counterexample_lines,
        "",
        f"## Dense-run counterexamples ({len(counterexamples)}/{len(snapshot_rows)})",
        "",
        "| state | stage | delta | pert/clean | visibility gain | clean min-vis view | max RGB-change view |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in counterexamples:
        report_lines.append(
            f"| {row['trajectory_state_index']} | {row['stage']} | "
            f"{row['disagreement_delta']:+.5f} | {row['perturbed_over_clean_ratio']:.3f} | "
            f"{100 * row['visibility_gain_fraction']:+.1f}% | "
            f"{row['clean_min_visibility_view']} | {row['largest_rgb_change_view']} |"
        )
    report_lines.extend(
        [
            "",
            "The segmentation visibility gain measures how many more target-bowl pixels are",
            "visible after perturbation. Large positive gains in a counterexample indicate",
            "that moving the bowl out from under the robot made the anomaly easier, not harder,",
            "to see; a visually obvious anomaly can produce a more consistent recovery action",
            "and therefore lower cross-view disagreement.",
            "",
        ]
    )
    (experiment_dir / "counterexample_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    summary = {
        "num_snapshots": len(snapshot_rows),
        "num_perturbed_greater": all_positive,
        "num_counterexamples": len(counterexamples),
        "critical_stage_names": sorted(CRITICAL_STAGES),
        "critical_num_snapshots": len(critical),
        "critical_num_perturbed_greater": critical_positive,
        "critical_mean_paired_delta": float(
            np.mean([row["disagreement_delta"] for row in critical])
        ),
        "noncritical_num_snapshots": len(noncritical),
        "noncritical_num_perturbed_greater": noncritical_positive,
        "noncritical_mean_paired_delta": float(
            np.mean([row["disagreement_delta"] for row in noncritical])
        ),
        "exploratory_fisher_one_sided_p": fisher_p,
        "stage_summary": stage_rows,
    }
    (experiment_dir / "counterexample_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
