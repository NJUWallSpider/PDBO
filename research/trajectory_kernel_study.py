#!/usr/bin/env python3
"""Evaluate SDP/GW kernels assembled from PDBO trajectory batches."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdbo.refinement import refine_binary_incumbent
from src.problem_parser import generate_Max_cut as generate_max_cut, parse_gset
from src.solver import PDBO_CPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gset-ids", nargs="+", type=int, default=[1, 3, 5, 70])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/improvement_study/trajectory_kernel.csv"),
    )
    return parser.parse_args()


def kernel_statistics(
    edge_dot: np.ndarray,
    norm_squared: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_weights: np.ndarray,
) -> tuple[float, float]:
    denominator = np.sqrt(norm_squared[edge_rows] * norm_squared[edge_cols])
    correlation = np.divide(
        edge_dot,
        denominator,
        out=np.zeros_like(edge_dot),
        where=denominator > 0.0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    sdp_value = float(np.sum(edge_weights * (1.0 - correlation) / 2.0))
    arcsine_expected_cut = float(
        np.sum(edge_weights * np.arccos(correlation) / math.pi)
    )
    return sdp_value, arcsine_expected_cut


def select_kernel(
    snapshots: list[np.ndarray],
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_weights: np.ndarray,
) -> tuple[np.ndarray, str, int, float, float]:
    cumulative_norm = np.zeros(snapshots[0].shape[1], dtype=np.float64)
    cumulative_edge_dot = np.zeros(edge_weights.size, dtype=np.float64)
    best = None

    for index, snapshot in enumerate(snapshots):
        norm_squared = np.sum(snapshot * snapshot, axis=0)
        edge_dot = np.sum(
            snapshot[:, edge_rows] * snapshot[:, edge_cols],
            axis=0,
        )
        current_values = kernel_statistics(
            edge_dot, norm_squared, edge_rows, edge_cols, edge_weights
        )
        current = (current_values[0], snapshot, "checkpoint", index, current_values[1])
        if best is None or current[0] > best[0]:
            best = current

        cumulative_norm += norm_squared
        cumulative_edge_dot += edge_dot
        cumulative_values = kernel_statistics(
            cumulative_edge_dot,
            cumulative_norm,
            edge_rows,
            edge_cols,
            edge_weights,
        )
        cumulative = (
            cumulative_values[0],
            np.concatenate(snapshots[: index + 1], axis=0),
            "cumulative",
            index,
            cumulative_values[1],
        )
        if cumulative[0] > best[0]:
            best = cumulative

    sdp_value, matrix, kind, index, expected_cut = best
    return matrix, kind, index, sdp_value, expected_cut


def gaussian_round(
    solver: PDBO_CPU,
    kernel_rows: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_candidate = None
    best_value = float("inf")
    chunk_size = 64
    for start in range(0, rounds, chunk_size):
        count = min(chunk_size, rounds - start)
        directions = rng.standard_normal((count, kernel_rows.shape[0]))
        candidates = (directions @ kernel_rows >= 0.0).astype(np.float32)
        values = solver._score_candidates(candidates)
        index = int(np.argmin(values))
        if float(values[index]) < best_value:
            best_value = float(values[index])
            best_candidate = candidates[index].copy()
    return best_candidate, best_value


def run_once(data: dict, gset_id: int, seed: int, args: argparse.Namespace) -> dict:
    snapshots: list[np.ndarray] = []
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=args.batch,
        max_iters=args.steps,
        primal_lr=args.alpha,
        dual_lr=args.beta,
        dual_init=1.0,
        dual_init_mode="spectral",
        dual_burn_in=args.burn_in,
        primal_init="center_uniform",
        rho=args.rho,
        seed=seed,
        verbose=False,
        conditional_rounding=True,
    )
    snapshots.append(np.asarray(solver.primal, dtype=np.float64) - 0.5)

    def capture(step, _step_time, _objective, _incumbent):
        if (step + 1) % args.snapshot_every == 0 or step + 1 == args.steps:
            snapshots.append(np.asarray(solver.primal, dtype=np.float64) - 0.5)

    solver.step_callback = capture
    solver.optimize()
    conditional_cut = -float(solver.objective)
    conditional_refinement = refine_binary_incumbent("mc", solver.incumbent, data)

    upper = sparse.triu(solver.W, k=1).tocoo()
    edge_rows = np.asarray(upper.row, dtype=np.int64)
    edge_cols = np.asarray(upper.col, dtype=np.int64)
    edge_weights = np.asarray(upper.data, dtype=np.float64)
    kernel_started = time.perf_counter()
    kernel_rows, kind, checkpoint, phi, expected_cut = select_kernel(
        snapshots, edge_rows, edge_cols, edge_weights
    )
    kernel_candidate, kernel_objective = gaussian_round(
        solver, kernel_rows, args.rounds, seed=10_000 + seed
    )
    kernel_seconds = time.perf_counter() - kernel_started
    kernel_refinement = refine_binary_incumbent("mc", kernel_candidate, data)
    total_weight = float(np.sum(edge_weights))
    spectral_upper = total_weight / 2.0 - solver.n * solver.dual_init_lambda_min / 4.0

    return {
        "gset_id": gset_id,
        "n": solver.n,
        "seed": seed,
        "snapshots": len(snapshots),
        "kernel_rank_rows": kernel_rows.shape[0],
        "kernel_kind": kind,
        "kernel_checkpoint": checkpoint * args.snapshot_every,
        "effective_dual_init": solver.dual_init,
        "conditional_cut": conditional_cut,
        "conditional_refined_cut": -float(conditional_refinement.objective),
        "kernel_cut": -kernel_objective,
        "kernel_refined_cut": -float(kernel_refinement.objective),
        "kernel_sdp_value": phi,
        "kernel_arcsine_expected_cut": expected_cut,
        "spectral_upper": spectral_upper,
        "gw_certified_ratio": 0.8785672 * phi / spectral_upper,
        "kernel_seconds": kernel_seconds,
        "solve_seconds": solver.runtime,
    }


def main() -> None:
    args = parse_args()
    if args.snapshot_every < 1 or args.rounds < 1:
        raise ValueError("snapshot-every and rounds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gset_id in dict.fromkeys(args.gset_ids):
        graph = parse_gset(str(gset_id))
        data = generate_max_cut(graph)
        for seed in dict.fromkeys(args.seeds):
            row = run_once(data, gset_id, seed, args)
            rows.append(row)
            print(
                f"G{gset_id} seed={seed} CE={row['conditional_refined_cut']:.0f} "
                f"kernel={row['kernel_refined_cut']:.0f} phi={row['kernel_sdp_value']:.1f}"
            )

    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
