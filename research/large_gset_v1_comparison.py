#!/usr/bin/env python3
"""Compare PSD-margin and minimum-eigenvector starts on large Gset graphs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.sparse import linalg as sparse_linalg


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem_parser import generate_Max_cut as generate_max_cut, parse_gset
from src.solver import PDBO_CPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gset-ids", nargs="+", type=int, default=[67, 70, 72, 77, 81])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--psd-margin", type=float, default=0.1)
    parser.add_argument("--v1-max-abs", type=float, default=0.49)
    parser.add_argument("--v1-clip-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/large_gset_v1"),
    )
    return parser.parse_args()


def build_solver(data: dict[str, Any], args: argparse.Namespace) -> PDBO_CPU:
    return PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=1,
        primal_lr=args.alpha,
        dual_lr=args.beta,
        dual_init=1.0,
        max_iters=0,
        primal_init="center_uniform",
        rho=args.rho,
        seed=args.seed,
        verbose=False,
    )


def smallest_eigenpair(
    solver: PDBO_CPU, seed: int
) -> tuple[float, np.ndarray, float, float]:
    matrix = solver.W.astype(np.float64)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    values, vectors = sparse_linalg.eigsh(
        matrix,
        k=1,
        which="SA",
        v0=rng.normal(size=solver.n),
        tol=1e-10,
        return_eigenvectors=True,
    )
    elapsed = time.perf_counter() - started
    eigenvalue = float(values[0])
    eigenvector = vectors[:, 0]
    residual = float(np.linalg.norm(matrix @ eigenvector - eigenvalue * eigenvector))
    return eigenvalue, eigenvector, residual, elapsed


def configure_initial_state(
    solver: PDBO_CPU,
    initialization: str,
    lambda_min: float,
    v1: np.ndarray,
    args: argparse.Namespace,
) -> float:
    initial_dual = -lambda_min + args.psd_margin
    if initialization != "center_uniform":
        scale = args.v1_max_abs / float(np.max(np.abs(v1)))
        solver.primal[0] = (0.5 + scale * v1).astype(np.float32)
    if initialization == "v1_clip_threshold":
        multiplier = (0.5 / args.v1_max_abs) ** (1.0 / args.v1_clip_steps)
        initial_dual = (1.0 - multiplier) / (2.0 * args.alpha) - lambda_min
    solver.dual.fill(initial_dual)
    return initial_dual


def sample_state(
    solver: PDBO_CPU,
    v1: np.ndarray,
    gset_id: int,
    initialization: str,
    initial_dual: float,
    lambda_min: float,
    eigen_residual: float,
    step: int,
) -> dict[str, float | int | str]:
    z = np.asarray(solver.primal[0], dtype=np.float64) - 0.5
    norm_squared = float(z @ z)
    if norm_squared > 0.0:
        wz = solver.W @ z
        rayleigh = float(z @ wz) / norm_squared
        v1_energy = float(z @ v1) ** 2 / norm_squared
    else:
        rayleigh = 0.0
        v1_energy = 0.0
    dual = np.asarray(solver.dual[0], dtype=np.float64)
    return {
        "gset_id": gset_id,
        "initialization": initialization,
        "n": solver.n,
        "step": step,
        "lambda_min": lambda_min,
        "eigen_residual": eigen_residual,
        "initial_dual": initial_dual,
        "archive_cut": -float(solver.objective),
        "fractionality": 1.0 - 4.0 * norm_squared / solver.n,
        "rayleigh": rayleigh,
        "v1_energy_fraction": v1_energy,
        "mean_dual": float(np.mean(dual)),
        "dual_anisotropy_rms": float(np.std(dual)),
        "saturated_fraction": float(
            np.mean((solver.primal[0] <= 0.0) | (solver.primal[0] >= 1.0))
        ),
    }


def run_trajectory(
    data: dict[str, Any],
    args: argparse.Namespace,
    gset_id: int,
    initialization: str,
    lambda_min: float,
    v1: np.ndarray,
    eigen_residual: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    solver = build_solver(data, args)
    initial_dual = configure_initial_state(
        solver, initialization, lambda_min, v1, args
    )
    solver._update_incumbent()
    rows = [
        sample_state(
            solver,
            v1,
            gset_id,
            initialization,
            initial_dual,
            lambda_min,
            eigen_residual,
            0,
        )
    ]
    first_saturation = -1
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        solver._paper_step()
        solver._update_incumbent()
        saturated = np.any((solver.primal[0] <= 0.0) | (solver.primal[0] >= 1.0))
        if first_saturation < 0 and saturated:
            first_saturation = step
        if step % args.sample_every == 0 or step == args.steps:
            rows.append(
                sample_state(
                    solver,
                    v1,
                    gset_id,
                    initialization,
                    initial_dual,
                    lambda_min,
                    eigen_residual,
                    step,
                )
            )
    runtime = time.perf_counter() - started
    final_cut = float(rows[-1]["archive_cut"])
    target = 0.99 * final_cut
    time_to_target = min(
        int(row["step"]) for row in rows if float(row["archive_cut"]) >= target
    )
    by_step = {int(row["step"]): row for row in rows}

    def cut_at(target_step: int) -> float:
        step = max(item for item in by_step if item <= target_step)
        return float(by_step[step]["archive_cut"])

    summary: dict[str, float | int | str] = {
        "gset_id": gset_id,
        "initialization": initialization,
        "n": solver.n,
        "lambda_min": lambda_min,
        "eigen_residual": eigen_residual,
        "initial_dual": initial_dual,
        "initial_convex": int(initial_dual + lambda_min >= -1e-10),
        "first_saturation": first_saturation,
        "cut_at_100": cut_at(100),
        "cut_at_500": cut_at(500),
        "cut_at_1000": cut_at(1000),
        "final_cut": final_cut,
        "time_to_99pct_final_cut": time_to_target,
        "final_v1_energy_fraction": float(rows[-1]["v1_energy_fraction"]),
        "final_rayleigh": float(rows[-1]["rayleigh"]),
        "runtime_seconds": runtime,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.sample_every < 1:
        raise ValueError("steps and sample-every must be positive")
    if args.psd_margin < 0.0:
        raise ValueError("psd-margin must be non-negative")
    if not 0.0 < args.v1_max_abs < 0.5 or args.v1_clip_steps < 1:
        raise ValueError("invalid v1 clipping target")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for gset_id in dict.fromkeys(args.gset_ids):
        graph = parse_gset(str(gset_id))
        data = generate_max_cut(graph)
        prototype = build_solver(data, args)
        lambda_min, v1, residual, eigen_seconds = smallest_eigenpair(
            prototype, args.seed
        )
        for initialization in (
            "center_uniform",
            "v1_psd_margin",
            "v1_clip_threshold",
        ):
            rows, summary = run_trajectory(
                data,
                args,
                gset_id,
                initialization,
                lambda_min,
                v1,
                residual,
            )
            summary["eigen_seconds"] = eigen_seconds
            all_rows.extend(rows)
            summaries.append(summary)
        print(f"G{gset_id}: completed")

    timeseries_path = args.output_dir / "large-gset-v1-timeseries.csv"
    with timeseries_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.output_dir / "large-gset-v1-summary.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"saved {timeseries_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
