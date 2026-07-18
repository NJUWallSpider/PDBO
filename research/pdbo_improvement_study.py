#!/usr/bin/env python3
"""Paired multi-instance study for spectrum-scaled PDBO improvements."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

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
    parser.add_argument("--configured-duals", nargs="*", type=float, default=[15.0])
    parser.add_argument("--burn-ins", nargs="*", type=int, default=[0, 250, 500])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument(
        "--alpha-mode", choices=("configured", "spectral"), default="configured"
    )
    parser.add_argument("--spectral-step-fraction", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=1e-8)
    parser.add_argument("--refine-max-passes", type=int, default=None)
    parser.add_argument(
        "--conditional-rounding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/improvement_study/runs.csv"),
    )
    return parser.parse_args()


def configurations(args: argparse.Namespace) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for value in args.configured_duals:
        result.append(
            {
                "name": f"configured-{value:g}",
                "dual_init_mode": "configured",
                "dual_init": value,
                "dual_burn_in": 0,
            }
        )
    for burn_in in args.burn_ins:
        result.append(
            {
                "name": f"spectral-{burn_in}",
                "dual_init_mode": "spectral",
                "dual_init": 1.0,
                "dual_burn_in": burn_in,
            }
        )
    return result


def run_once(
    data: dict,
    gset_id: int,
    seed: int,
    config: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    init_started = time.perf_counter()
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=args.batch,
        max_iters=args.steps,
        primal_lr=args.alpha,
        primal_lr_mode=args.alpha_mode,
        spectral_step_fraction=args.spectral_step_fraction,
        dual_lr=args.beta,
        dual_init=float(config["dual_init"]),
        dual_init_mode=str(config["dual_init_mode"]),
        dual_burn_in=int(config["dual_burn_in"]),
        tolerance=args.delta,
        primal_init="center_uniform",
        rho=args.rho,
        seed=seed,
        verbose=False,
        conditional_rounding=args.conditional_rounding,
    )
    initialization_seconds = time.perf_counter() - init_started

    initial_cut = -float(solver.objective)
    solve_started = time.perf_counter()
    solver.optimize()
    solve_seconds = time.perf_counter() - solve_started
    raw_cut = -float(solver.objective)

    refinement = refine_binary_incumbent(
        "mc",
        np.asarray(solver.incumbent, dtype=np.int32),
        data,
        max_passes=args.refine_max_passes,
    )
    refined_cut = -float(refinement.objective)

    return {
        "gset_id": gset_id,
        "n": solver.n,
        "seed": seed,
        "configuration": config["name"],
        "conditional_rounding": int(args.conditional_rounding),
        "dual_init_mode": solver.dual_init_mode,
        "dual_burn_in": solver.dual_burn_in,
        "effective_dual_init": solver.dual_init,
        "effective_primal_lr": solver.primal_lr,
        "lambda_min": (
            solver.dual_init_lambda_min
            if solver.dual_init_lambda_min is not None
            else float("nan")
        ),
        "initial_cut": initial_cut,
        "raw_cut": raw_cut,
        "best_relaxed_expected_cut": (
            -float(solver.conditional_rounding_expected_objective)
            if solver.conditional_rounding_expected_objective is not None
            else float("nan")
        ),
        "refined_cut": refined_cut,
        "refinement_gain": refined_cut - raw_cut,
        "initialization_seconds": initialization_seconds,
        "solve_seconds": solve_seconds,
        "refinement_seconds": refinement.seconds,
        "perturbations": solver.perturbation_count,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.batch < 1:
        raise ValueError("steps must be non-negative and batch must be positive")
    if any(value < 0 for value in args.burn_ins):
        raise ValueError("burn-ins must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    configs = configurations(args)
    for gset_id in dict.fromkeys(args.gset_ids):
        graph = parse_gset(str(gset_id))
        data = generate_max_cut(graph)
        for seed in dict.fromkeys(args.seeds):
            for config in configs:
                row = run_once(data, gset_id, seed, config, args)
                rows.append(row)
                print(
                    f"G{gset_id} seed={seed} {config['name']} "
                    f"raw={row['raw_cut']:.0f} refined={row['refined_cut']:.0f}"
                )

    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
