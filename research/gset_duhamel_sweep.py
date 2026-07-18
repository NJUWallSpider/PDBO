#!/usr/bin/env python3
"""Run the per-mode Duhamel diagnostic across multiple Gset instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN_SCRIPT = Path(__file__).with_name("g3_spectral_dynamics.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gset-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dual-init", type=float, default=15.0)
    parser.add_argument(
        "--dual-init-mode",
        choices=("configured", "psd-margin"),
        default="configured",
    )
    parser.add_argument("--dual-psd-margin", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument(
        "--initialization",
        choices=("center-uniform", "v1"),
        default="center-uniform",
    )
    parser.add_argument("--v1-max-abs", type=float, default=0.49)
    parser.add_argument(
        "--v1-dual-mode",
        choices=("configured", "psd-boundary", "clip-threshold"),
        default="clip-threshold",
    )
    parser.add_argument("--v1-clip-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--major-low-modes", type=int, default=8)
    parser.add_argument("--major-energy-modes", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/gset_sweep"),
    )
    parser.add_argument(
        "--save-npz", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--reuse-existing", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def metrics_at_time(rows: list[dict[str, str]], time_index: int) -> dict[str, float | int]:
    selected = [row for row in rows if int(row["time"]) == time_index]
    if not selected:
        raise ValueError(f"no Duhamel rows at time {time_index}")
    epsilon = [float(row["epsilon"]) for row in selected]
    mode_energy = [float(row["mode_energy_fraction"]) for row in selected]
    selected_energy = float(selected[0]["selected_modes_energy_fraction"])
    energy_lt_point_one = sum(
        energy for energy, value in zip(mode_energy, epsilon) if value < 0.1
    )
    energy_lt_one = sum(
        energy for energy, value in zip(mode_energy, epsilon) if value < 1.0
    )
    return {
        "time": time_index,
        "epsilon_lt_0_1": sum(value < 0.1 for value in epsilon),
        "epsilon_lt_1": sum(value < 1.0 for value in epsilon),
        "max_epsilon": max(epsilon),
        "selected_energy_fraction": selected_energy,
        "energy_fraction_epsilon_lt_0_1": energy_lt_point_one,
        "energy_fraction_epsilon_lt_1": energy_lt_one,
        "selected_energy_share_epsilon_lt_0_1": (
            energy_lt_point_one / selected_energy if selected_energy > 0.0 else 0.0
        ),
        "selected_energy_share_epsilon_lt_1": (
            energy_lt_one / selected_energy if selected_energy > 0.0 else 0.0
        ),
    }


def summarize_result(
    summary: dict[str, Any], rows: list[dict[str, str]], scalar_rows: list[dict[str, str]]
) -> dict[str, Any]:
    times = sorted({int(row["time"]) for row in rows})
    first_clip = int(summary.get("first_clip_exact", summary["first_clip"]))
    if first_clip >= 0:
        before_clip_candidates = [time for time in times if time < first_clip]
        preclip_time = before_clip_candidates[-1] if before_clip_candidates else times[0]
        clip_time = min(time for time in times if time >= first_clip)
    else:
        preclip_time = times[-1]
        clip_time = times[-1]

    initial = metrics_at_time(rows, times[0])
    preclip = metrics_at_time(rows, preclip_time)
    clip = metrics_at_time(rows, clip_time)
    final = metrics_at_time(rows, times[-1])
    selected_count = int(summary["duhamel"]["selected_mode_count"])
    scalar_by_time = {int(float(row["time"])): row for row in scalar_rows}

    def archive_cut_at(target_time: int) -> float:
        available = [time for time in scalar_by_time if time <= target_time]
        return float(scalar_by_time[max(available)]["archive_best_cut"])

    final_cut = float(summary["best_threshold_cut"])
    target_cut = 0.99 * final_cut
    target_times = [
        time
        for time, row in scalar_by_time.items()
        if float(row["archive_best_cut"]) >= target_cut
    ]

    result: dict[str, Any] = {
        "gset_id": int(summary["parameters"]["gset_id"]),
        "n": int(summary["n"]),
        "lambda_min": float(summary["lambda_min"]),
        "lambda_max": float(summary["lambda_max"]),
        "initialization": str(summary["parameters"]["initialization"]),
        "effective_initial_dual": float(summary["effective_initial_dual"]),
        "initial_convex": int(summary["initial_convex_condition"]),
        "positive_multiplier_condition": int(summary["positive_multiplier_condition"]),
        "selected_mode_count": selected_count,
        "initial_selected_energy_fraction": initial["selected_energy_fraction"],
        "preclip_time": preclip["time"],
        "preclip_epsilon_lt_0_1": preclip["epsilon_lt_0_1"],
        "preclip_epsilon_lt_1": preclip["epsilon_lt_1"],
        "preclip_max_epsilon": preclip["max_epsilon"],
        "preclip_selected_energy_fraction": preclip["selected_energy_fraction"],
        "preclip_energy_fraction_epsilon_lt_0_1": preclip[
            "energy_fraction_epsilon_lt_0_1"
        ],
        "preclip_energy_fraction_epsilon_lt_1": preclip[
            "energy_fraction_epsilon_lt_1"
        ],
        "preclip_selected_energy_share_epsilon_lt_0_1": preclip[
            "selected_energy_share_epsilon_lt_0_1"
        ],
        "preclip_selected_energy_share_epsilon_lt_1": preclip[
            "selected_energy_share_epsilon_lt_1"
        ],
        "first_clip": first_clip,
        "clip_sample_time": clip["time"],
        "clip_epsilon_lt_0_1": clip["epsilon_lt_0_1"],
        "clip_epsilon_lt_1": clip["epsilon_lt_1"],
        "clip_max_epsilon": clip["max_epsilon"],
        "clip_selected_energy_fraction": clip["selected_energy_fraction"],
        "clip_energy_fraction_epsilon_lt_0_1": clip[
            "energy_fraction_epsilon_lt_0_1"
        ],
        "clip_energy_fraction_epsilon_lt_1": clip[
            "energy_fraction_epsilon_lt_1"
        ],
        "first_prediction_error_above_10pct": int(
            summary["first_prediction_error_above_10pct"]
        ),
        "first_forcing_relative_above_1pct": int(
            summary["first_forcing_relative_above_1pct"]
        ),
        "final_time": final["time"],
        "final_epsilon_lt_0_1": final["epsilon_lt_0_1"],
        "final_epsilon_lt_1": final["epsilon_lt_1"],
        "final_max_epsilon": final["max_epsilon"],
        "final_selected_energy_fraction": final["selected_energy_fraction"],
        "archive_cut_at_100": archive_cut_at(100),
        "archive_cut_at_500": archive_cut_at(500),
        "archive_cut_at_1000": archive_cut_at(1000),
        "time_to_99pct_final_cut": min(target_times) if target_times else -1,
        "final_energy_fraction_epsilon_lt_0_1": final[
            "energy_fraction_epsilon_lt_0_1"
        ],
        "final_energy_fraction_epsilon_lt_1": final[
            "energy_fraction_epsilon_lt_1"
        ],
        "best_threshold_cut": float(summary["best_threshold_cut"]),
        "spectral_upper": float(summary["spectral_upper"]),
        "spectral_certified_ratio": float(summary["spectral_certified_ratio"]),
        "max_relative_identity_residual": float(
            summary["duhamel"]["max_relative_identity_residual"]
        ),
        "max_forcing_projection_discrepancy": float(
            summary["duhamel"]["max_forcing_projection_discrepancy"]
        ),
        "max_bound_violation": float(summary["duhamel"]["max_bound_violation"]),
        "max_relative_bound_violation": float(
            summary["duhamel"].get("max_relative_bound_violation", 0.0)
        ),
    }
    return result


def run_instance(args: argparse.Namespace, gset_id: int) -> tuple[Path, Path, Path]:
    prefix = args.output_dir / f"g{gset_id}_filtering"
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.parent / f"{prefix.name}-duhamel.csv"
    scalar_path = prefix.parent / f"{prefix.name}-scalars.csv"
    if (
        args.reuse_existing
        and json_path.exists()
        and csv_path.exists()
        and scalar_path.exists()
    ):
        print(f"G{gset_id}: reused existing artifacts")
        return json_path, csv_path, scalar_path
    command = [
        sys.executable,
        str(SINGLE_RUN_SCRIPT),
        "--gset-id",
        str(gset_id),
        "--steps",
        str(args.steps),
        "--sample-every",
        str(args.sample_every),
        "--batch",
        str(args.batch),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--dual-init",
        str(args.dual_init),
        "--dual-init-mode",
        args.dual_init_mode,
        "--dual-psd-margin",
        str(args.dual_psd_margin),
        "--rho",
        str(args.rho),
        "--initialization",
        args.initialization,
        "--v1-max-abs",
        str(args.v1_max_abs),
        "--v1-dual-mode",
        args.v1_dual_mode,
        "--v1-clip-steps",
        str(args.v1_clip_steps),
        "--seed",
        str(args.seed),
        "--bins",
        str(args.bins),
        "--trajectory-index",
        str(args.trajectory_index),
        "--major-low-modes",
        str(args.major_low_modes),
        "--major-energy-modes",
        str(args.major_energy_modes),
        "--output",
        str(prefix),
        "--no-plot",
        "--save-npz" if args.save_npz else "--no-save-npz",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"Gset G{gset_id} diagnostic failed")
    print(f"G{gset_id}: completed")
    return json_path, csv_path, scalar_path


def main() -> None:
    args = parse_args()
    if not args.gset_ids or any(gset_id < 1 for gset_id in args.gset_ids):
        raise ValueError("gset-ids must contain positive instance IDs")
    args.gset_ids = list(dict.fromkeys(args.gset_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for gset_id in args.gset_ids:
        json_path, csv_path, scalar_path = run_instance(args, gset_id)
        summary = json.loads(json_path.read_text())
        with csv_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        with scalar_path.open(newline="") as csv_file:
            scalar_rows = list(csv.DictReader(csv_file))
        summaries.append(summarize_result(summary, rows, scalar_rows))

    summary_path = args.output_dir / "gset-duhamel-sweep.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
