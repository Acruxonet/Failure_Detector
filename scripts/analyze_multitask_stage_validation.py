#!/usr/bin/env python3
"""Validate the frozen task-stage hypothesis across held-out tasks and episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi05_multitask_stage_mpl")

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
TASK_LABELS = {
    1: "bowl next to ramekin",
    2: "bowl at table center",
    6: "bowl next to cookie box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
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
    """Apply the thresholds frozen after the original task-0 analysis."""
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


def exact_one_sided_sign_p(num_positive: int, total: int) -> float:
    if total == 0:
        return float("nan")
    return float(
        sum(math.comb(total, value) for value in range(num_positive, total + 1))
        / 2**total
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray([row["disagreement_delta"] for row in rows], dtype=np.float64)
    clean = np.asarray([row["clean_disagreement"] for row in rows], dtype=np.float64)
    perturbed = np.asarray(
        [row["perturbed_disagreement"] for row in rows], dtype=np.float64
    )
    positive = int(np.sum(deltas > 0.0))
    return {
        "num_snapshots": len(rows),
        "num_perturbed_greater": positive,
        "fraction_perturbed_greater": positive / len(rows),
        "clean_mean": float(clean.mean()),
        "perturbed_mean": float(perturbed.mean()),
        "mean_paired_delta": float(deltas.mean()),
        "median_paired_delta": float(np.median(deltas)),
    }


def save_task_scatter(path: Path, rows: list[dict[str, Any]]) -> None:
    clean = np.asarray([row["clean_disagreement"] for row in rows])
    perturbed = np.asarray([row["perturbed_disagreement"] for row in rows])
    lower = float(min(clean.min(), perturbed.min()))
    upper = float(max(clean.max(), perturbed.max()))
    padding = max((upper - lower) * 0.06, 0.008)
    limits = (max(0.0, lower - padding), upper + padding)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True)
    for axis, task_id in zip(axes, sorted(TASK_LABELS), strict=True):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        for stage in STAGE_ORDER:
            stage_rows = [row for row in task_rows if row["stage"] == stage]
            if not stage_rows:
                continue
            for episode_id, marker in ((0, "o"), (1, "^")):
                episode_rows = [
                    row for row in stage_rows if row["episode_id"] == episode_id
                ]
                if not episode_rows:
                    continue
                axis.scatter(
                    [row["clean_disagreement"] for row in episode_rows],
                    [row["perturbed_disagreement"] for row in episode_rows],
                    color=STAGE_COLORS[stage],
                    marker=marker,
                    s=44,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.45,
                )
        axis.plot(limits, limits, "--", color="black", linewidth=1.2)
        num_up = sum(row["disagreement_delta"] > 0 for row in task_rows)
        axis.set_title(
            f"Task {task_id}: {TASK_LABELS[task_id]}\n"
            f"perturbed > clean: {num_up}/{len(task_rows)}"
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.set_xlabel("Clean disagreement")
    axes[0].set_ylabel("Perturbed disagreement")

    stage_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            color=STAGE_COLORS[stage],
            label=stage,
            markersize=7,
        )
        for stage in STAGE_ORDER
    ]
    episode_handles = [
        plt.Line2D([], [], linestyle="", marker="o", color="#444", label="episode 0"),
        plt.Line2D([], [], linestyle="", marker="^", color="#444", label="episode 1"),
    ]
    axes[-1].legend(
        handles=stage_handles + episode_handles,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.92,
    )
    figure.suptitle(
        "Held-out LIBERO validation: paired cross-view disagreement (50 snapshots/episode)",
        fontsize=14,
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_episode_effects(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["episode_key"].replace("task_", "t").replace("_episode_", "e") for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    axis.bar(
        x - width / 2,
        [row["critical_mean_paired_delta"] for row in rows],
        width,
        color="#E15759",
        label="manipulation-critical",
    )
    axis.bar(
        x + width / 2,
        [row["noncritical_mean_paired_delta"] for row in rows],
        width,
        color="#4C78A8",
        label="non-critical",
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean paired delta (perturbed - clean)")
    axis.set_title("Stage effect by independent rollout episode")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pairs_dir = args.pairs_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    paired_states = np.load(pairs_dir / "paired_states.npz", allow_pickle=False)
    action_outputs = np.load(experiment_dir / "action_outputs.npz", allow_pickle=False)
    metric_rows = read_csv(experiment_dir / "per_group_metrics.csv")
    metric_lookup = {
        (int(row["pair_id"]), row["variant"]): float(row[PRIMARY_METRIC])
        for row in metric_rows
    }
    with np.load(pairs_dir / "policy_samples.npz", allow_pickle=False) as policy_data:
        policy_states = policy_data["observation.state"][::10].copy()

    num_pairs = len(paired_states["global_pair_id"])
    if num_pairs != 300 or len(policy_states) != num_pairs:
        raise ValueError(
            f"Expected 300 pairs/states, got {num_pairs}/{len(policy_states)}"
        )
    if len(action_outputs["sample_id"]) != 3000:
        raise ValueError("Expected 3000 policy action outputs")

    table_z_by_episode: dict[str, float] = {}
    for episode_key in np.unique(paired_states["episode_key"].astype(str)):
        indices = np.flatnonzero(paired_states["episode_key"].astype(str) == episode_key)
        early = indices[paired_states["episode_pair_id"][indices] < 8]
        table_z_by_episode[episode_key] = float(
            np.median(paired_states["clean_object_qpos"][early, 2])
        )

    snapshot_rows: list[dict[str, Any]] = []
    for pair_index in range(num_pairs):
        pair_id = int(paired_states["global_pair_id"][pair_index])
        task_id = int(paired_states["task_id"][pair_index])
        episode_id = int(paired_states["episode_id"][pair_index])
        episode_key = str(paired_states["episode_key"][pair_index])
        object_position = paired_states["clean_object_qpos"][pair_index, :3]
        target_position = paired_states["target_object_position_world"][pair_index]
        stage, eef_distance, lift_height, target_xy_distance = classify_stage(
            object_position,
            policy_states[pair_index, :3],
            target_position,
            table_z_by_episode[episode_key],
        )
        clean = metric_lookup[(pair_id, "clean")]
        perturbed = metric_lookup[(pair_id, "perturbed")]
        snapshot_rows.append(
            {
                "global_pair_id": pair_id,
                "task_id": task_id,
                "task_label": TASK_LABELS[task_id],
                "episode_id": episode_id,
                "episode_key": episode_key,
                "episode_pair_id": int(paired_states["episode_pair_id"][pair_index]),
                "trajectory_state_index": int(
                    paired_states["trajectory_state_indices"][pair_index]
                ),
                "stage": stage,
                "manipulation_critical": stage in CRITICAL_STAGES,
                "clean_disagreement": clean,
                "perturbed_disagreement": perturbed,
                "disagreement_delta": perturbed - clean,
                "perturbed_greater": perturbed > clean,
                "eef_object_distance_m": eef_distance,
                "object_lift_height_m": lift_height,
                "object_target_xy_distance_m": target_xy_distance,
            }
        )

    stage_rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        selected = [row for row in snapshot_rows if row["stage"] == stage]
        if selected:
            stage_rows.append({"stage": stage, **summarize_rows(selected)})

    episode_rows: list[dict[str, Any]] = []
    for episode_key in sorted({row["episode_key"] for row in snapshot_rows}):
        selected = [row for row in snapshot_rows if row["episode_key"] == episode_key]
        critical = [row for row in selected if row["manipulation_critical"]]
        noncritical = [row for row in selected if not row["manipulation_critical"]]
        critical_summary = summarize_rows(critical)
        noncritical_summary = summarize_rows(noncritical)
        episode_rows.append(
            {
                "episode_key": episode_key,
                "task_id": selected[0]["task_id"],
                "episode_id": selected[0]["episode_id"],
                "num_snapshots": len(selected),
                "critical_num_snapshots": len(critical),
                "critical_num_perturbed_greater": critical_summary[
                    "num_perturbed_greater"
                ],
                "critical_fraction_perturbed_greater": critical_summary[
                    "fraction_perturbed_greater"
                ],
                "critical_mean_paired_delta": critical_summary["mean_paired_delta"],
                "noncritical_num_snapshots": len(noncritical),
                "noncritical_num_perturbed_greater": noncritical_summary[
                    "num_perturbed_greater"
                ],
                "noncritical_fraction_perturbed_greater": noncritical_summary[
                    "fraction_perturbed_greater"
                ],
                "noncritical_mean_paired_delta": noncritical_summary[
                    "mean_paired_delta"
                ],
                "critical_minus_noncritical_mean_delta": critical_summary[
                    "mean_paired_delta"
                ]
                - noncritical_summary["mean_paired_delta"],
            }
        )

    task_rows: list[dict[str, Any]] = []
    for task_id in sorted(TASK_LABELS):
        selected = [row for row in snapshot_rows if row["task_id"] == task_id]
        critical = [row for row in selected if row["manipulation_critical"]]
        noncritical = [row for row in selected if not row["manipulation_critical"]]
        all_summary = summarize_rows(selected)
        critical_summary = summarize_rows(critical)
        noncritical_summary = summarize_rows(noncritical)
        task_rows.append(
            {
                "task_id": task_id,
                "task_label": TASK_LABELS[task_id],
                **all_summary,
                "critical_num_snapshots": len(critical),
                "critical_fraction_perturbed_greater": critical_summary[
                    "fraction_perturbed_greater"
                ],
                "critical_mean_paired_delta": critical_summary["mean_paired_delta"],
                "noncritical_num_snapshots": len(noncritical),
                "noncritical_fraction_perturbed_greater": noncritical_summary[
                    "fraction_perturbed_greater"
                ],
                "noncritical_mean_paired_delta": noncritical_summary[
                    "mean_paired_delta"
                ],
            }
        )

    critical = [row for row in snapshot_rows if row["manipulation_critical"]]
    noncritical = [row for row in snapshot_rows if not row["manipulation_critical"]]
    critical_summary = summarize_rows(critical)
    noncritical_summary = summarize_rows(noncritical)
    critical_episode_positive = sum(
        row["critical_mean_paired_delta"] > 0 for row in episode_rows
    )
    contrast_episode_positive = sum(
        row["critical_minus_noncritical_mean_delta"] > 0 for row in episode_rows
    )
    summary = {
        "validation_design": {
            "held_out_task_ids": sorted(TASK_LABELS),
            "episodes_per_task": 2,
            "snapshots_per_episode": 50,
            "num_snapshot_pairs": num_pairs,
            "primary_metric": PRIMARY_METRIC,
            "stage_rule_status": "frozen from task-0 analysis before this validation",
            "independent_replication_unit": "rollout episode",
        },
        "stage_thresholds_m": {
            "far_approach_eef_object_distance_gt": 0.115,
            "close_approach_eef_object_distance_gt": 0.055,
            "unlifted_object_height_lt": 0.003,
            "lift_transport_target_xy_distance_gt": 0.06,
        },
        "all_snapshots": summarize_rows(snapshot_rows),
        "critical_snapshots": critical_summary,
        "noncritical_snapshots": noncritical_summary,
        "episode_level_confirmation": {
            "num_episodes": len(episode_rows),
            "critical_mean_delta_positive_episodes": critical_episode_positive,
            "critical_mean_delta_positive_exact_one_sided_sign_p": exact_one_sided_sign_p(
                critical_episode_positive, len(episode_rows)
            ),
            "critical_minus_noncritical_positive_episodes": contrast_episode_positive,
            "critical_minus_noncritical_exact_one_sided_sign_p": exact_one_sided_sign_p(
                contrast_episode_positive, len(episode_rows)
            ),
        },
        "per_stage": stage_rows,
        "per_task": task_rows,
        "per_episode": episode_rows,
    }

    write_csv(experiment_dir / "snapshot_stage_validation.csv", snapshot_rows)
    write_csv(experiment_dir / "stage_validation.csv", stage_rows)
    write_csv(experiment_dir / "episode_stage_summary.csv", episode_rows)
    write_csv(experiment_dir / "task_stage_summary.csv", task_rows)
    save_task_scatter(experiment_dir / "paired_scatter_validation.png", snapshot_rows)
    save_episode_effects(experiment_dir / "episode_stage_effects.png", episode_rows)
    (experiment_dir / "stage_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "# 多任务、多 episode 阶段验证",
        "",
        "本次使用 3 个未参与原阶段规律总结的 LIBERO Spatial 任务，每个任务 2 个成功 rollout，",
        "每个 rollout 均匀抽取 50 个 snapshot。每个 snapshot 保持 proprioception、language、",
        "随机种子和推理噪声不变，只比较 clean / 物体轻微扰动后各 5 个相机视角的 action disagreement。",
        "",
        "阶段阈值沿用 task 0 的既有规则，没有根据本批结果重新调整。统计上把 6 个 rollout episode",
        "作为独立重复单位；300 个 snapshot 的比例只作描述，避免把同一轨迹内相关帧当作独立样本。",
        "",
        "## 汇总结果",
        "",
        f"- 全部 snapshot：{summary['all_snapshots']['num_perturbed_greater']}/300 的扰动 disagreement 更大。",
        f"- manipulation-critical：{critical_summary['num_perturbed_greater']}/{critical_summary['num_snapshots']}，平均 paired delta {critical_summary['mean_paired_delta']:+.6f}。",
        f"- non-critical：{noncritical_summary['num_perturbed_greater']}/{noncritical_summary['num_snapshots']}，平均 paired delta {noncritical_summary['mean_paired_delta']:+.6f}。",
        f"- 6 个 episode 中，critical 平均 delta 为正：{critical_episode_positive}/6；critical 高于 non-critical：{contrast_episode_positive}/6。",
        "",
        "详细的逐 snapshot、逐阶段、逐 task 和逐 episode 结果分别保存在同目录 CSV；",
        "`paired_scatter_validation.png` 展示三个任务的 paired scatter，",
        "`episode_stage_effects.png` 展示以 episode 为单位的阶段效应。",
        "",
    ]
    (experiment_dir / "VALIDATION_README.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
