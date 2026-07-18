"""Small examples for the public PDBO package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem_parser import (
    evaluate_LABS_bits as evaluate_labs_bits,
    generate_LABS as generate_labs,
    generate_Max_cut as generate_max_cut,
    generate_MIS as generate_mis,
    parse_gset,
    random_graph,
)
from src.solver import PDBO_CPU


def run_quadratic_mis():
    graph = random_graph(n=20, d=3, seed=0)
    data = generate_mis(graph, penalty=4)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=4,
        max_iters=20,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    print(f"MIS objective={result.objective:.3f}")


def run_gset_maxcut():
    graph = parse_gset(1)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=8,
        max_iters=20,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    print(f"Gset G1 Max-Cut QUBO objective={result.objective:.3f}")


def run_labs_qubo():
    n = 12
    data = generate_labs(n, penalty=10000)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=32,
        max_iters=100,
        primal_lr=0.03,
        dual_lr=0.03,
        dual_init=100,
        verbose=False,
        seed=0,
    )
    solver.optimize()
    energy = evaluate_labs_bits(solver.incumbent[:n])
    print(f"LABS QUBO energy={energy}")


if __name__ == "__main__":
    run_quadratic_mis()
    run_gset_maxcut()
    run_labs_qubo()
