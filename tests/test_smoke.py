import subprocess
import sys

import numpy as np

from pdbo import PDBOSolver, generate_mis, random_graph


def empty_qubo():
    return np.empty((2, 0), dtype=np.int32), np.empty(0, dtype=np.float32)


def test_paper_update_uses_old_primal_for_dual():
    indices, values = empty_qubo()
    solver = PDBOSolver(
        n_vars=1,
        Q_indices=indices,
        Q_values=values,
        c=np.zeros(1, dtype=np.float32),
        batch_size=1,
        primal_lr=0.1,
        dual_lr=0.2,
        max_iters=1,
        verbose=False,
    )
    solver.primal[...] = 0.25
    solver.dual[...] = 1.0
    solver.optimize()

    assert np.allclose(solver.primal, [[0.30]])
    assert np.allclose(solver.dual, [[0.9625]])


def test_paper_centre_perturbation_rule():
    indices, values = empty_qubo()
    solver = PDBOSolver(
        n_vars=1,
        Q_indices=indices,
        Q_values=values,
        c=np.zeros(1, dtype=np.float32),
        batch_size=1,
        primal_lr=0.1,
        dual_lr=0.1,
        tolerance=0.1,
        max_iters=1,
        primal_init="half",
        verbose=False,
    )
    solver.dual[...] = 0.0
    solver.optimize()

    assert np.allclose(solver.primal, [[0.4]])
    assert np.allclose(solver.dual, [[-0.025]])
    assert solver.perturbation_count == 1


def test_absolute_g_only_perturbs_at_exact_half():
    indices, values = empty_qubo()
    solver = PDBOSolver(
        n_vars=2,
        Q_indices=indices,
        Q_values=values,
        c=np.zeros(2, dtype=np.float32),
        batch_size=1,
        primal_lr=0.1,
        dual_lr=0.1,
        tolerance=0.1,
        max_iters=1,
        g_type="absolute",
        verbose=False,
    )
    solver.primal[...] = [[0.5, 0.55]]
    solver.dual[...] = 0.0
    solver.optimize()

    assert np.allclose(solver.primal, [[0.4, 0.55]])
    assert np.allclose(solver.dual, [[-0.05, -0.045]])
    assert solver.perturbation_count == 1


def test_absolute_g_subgradient_away_from_half():
    indices, values = empty_qubo()
    solver = PDBOSolver(
        n_vars=2,
        Q_indices=indices,
        Q_values=values,
        c=np.zeros(2, dtype=np.float32),
        batch_size=1,
        primal_lr=0.1,
        dual_lr=0.2,
        max_iters=1,
        g_type="absolute",
        verbose=False,
    )
    solver.primal[...] = [[0.25, 0.75]]
    solver.dual[...] = 1.0
    solver.optimize()

    assert np.allclose(solver.primal, [[0.35, 0.65]])
    assert np.allclose(solver.dual, [[0.95, 0.95]])


def test_restart_reuses_incumbent_and_resets_dual():
    indices, values = empty_qubo()
    solver = PDBOSolver(
        n_vars=1,
        Q_indices=indices,
        Q_values=values,
        c=np.array([-1.0], dtype=np.float32),
        batch_size=2,
        primal_lr=0.1,
        dual_lr=0.1,
        dual_init=3.0,
        max_iters=1,
        restart=True,
        verbose=False,
    )
    solver.primal[...] = 0.0
    solver.dual[...] = 3.0
    restart_state = {}
    original_restart = solver._restart_from_incumbent
    original_step = solver._paper_step

    def record_step():
        restart_state["steps"] = restart_state.get("steps", 0) + 1
        original_step()

    def record_restart():
        restart_state["incumbent"] = solver.incumbent.copy()
        original_restart()
        restart_state["primal"] = solver.primal.copy()
        restart_state["dual"] = solver.dual.copy()

    solver._paper_step = record_step
    solver._restart_from_incumbent = record_restart
    solver.optimize()

    assert solver.stop_reason == "max_iters"
    assert restart_state["steps"] == 2
    assert np.allclose(
        restart_state["primal"],
        np.broadcast_to(restart_state["incumbent"], solver.primal.shape),
    )
    assert np.allclose(restart_state["dual"], 3.0)


def test_quadratic_solver_smoke():
    graph = random_graph(n=12, d=3, seed=0)
    data = generate_mis(graph, penalty=4)
    solver = PDBOSolver(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=2,
        max_iters=5,
        check_every=5,
        rounding_samples=2,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    assert result.incumbent.shape == (data["num_vars"],)
    assert result.runtime >= 0.0


def test_cli_smoke():
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--task",
            "mis",
            "--graph",
            "reg",
            "--n",
            "12",
            "--d",
            "3",
            "--batch",
            "2",
            "--max_iters",
            "5",
            "--check_every",
            "5",
            "--no-verbose",
            "--refine",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "best=" in proc.stdout
    assert "refined_best=" in proc.stdout
