import subprocess
import sys

import numpy as np

from src.problem_parser import (
    generate_Max_cut as generate_max_cut,
    generate_MIS as generate_mis,
    random_graph,
)
from src.solver import PDBO_CPU
from src.main import build_parser
from src.spectral import (
    compute_animation_modes,
    compute_h_s_spectrum_certificate,
    compute_spectral_window,
    compute_spectrum_distribution,
    histogram_edges,
)
from src.spectral_animation import SpectralAnimation


def empty_qubo():
    return np.empty((2, 0), dtype=np.int32), np.empty(0, dtype=np.float32)


def test_center_uniform_initialization_stays_within_rho():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
        n_vars=100,
        Q_indices=indices,
        Q_values=values,
        batch_size=4,
        max_iters=0,
        rho=0.05,
        seed=0,
        verbose=False,
    )

    assert solver.primal.shape == (4, 100)
    assert np.all(solver.primal >= 0.45)
    assert np.all(solver.primal < 0.55)
    assert np.std(solver.primal) > 0.0


def test_initial_threshold_candidates_are_archived_before_first_step():
    graph = random_graph(n=12, d=3, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=8,
        max_iters=0,
        seed=3,
        verbose=False,
    )
    rounded = np.rint(solver.primal).astype(np.float32)
    candidates = np.concatenate(
        (rounded, np.zeros((1, solver.n), dtype=np.float32)),
        axis=0,
    )

    assert np.isclose(solver.objective, np.min(solver._score_candidates(candidates)))
    assert any(np.array_equal(solver.incumbent, candidate) for candidate in candidates)


def test_nearest_and_bernoulli_rounding_modes():
    indices, values = empty_qubo()
    primal = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)

    nearest = PDBO_CPU(
        n_vars=5,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        rounding_mode="nearest",
        verbose=False,
    )
    assert np.array_equal(nearest._round_primal(primal), [[0, 0, 0, 1, 1]])

    bernoulli = PDBO_CPU(
        n_vars=5,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        rounding_mode="bernoulli",
        seed=7,
        verbose=False,
    )
    bernoulli.rng = np.random.default_rng(123)
    expected_rng = np.random.default_rng(123)
    expected = (expected_rng.random(primal.shape) < primal).astype(np.float32)
    assert np.array_equal(bernoulli._round_primal(primal), expected)


def test_cli_accepts_bernoulli_rounding():
    args = build_parser().parse_args(["--rounding", "bernoulli"])
    assert args.rounding == "bernoulli"


def test_spectral_dual_initialization_targets_requested_burn_in():
    graph = random_graph(n=2, d=1, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=1,
        dual_lr=0.2,
        dual_init_mode="spectral",
        dual_burn_in=10,
        dual_psd_margin=0.1,
        max_iters=0,
        verbose=False,
    )

    assert np.isclose(solver.dual_init_lambda_min, -1.0)
    assert np.isclose(solver.dual_init, 1.6, atol=1e-6)
    assert np.allclose(solver.dual, 1.6, atol=1e-6)


def test_spectral_primal_step_has_requested_contractive_worst_mode_multiplier():
    graph = random_graph(n=2, d=1, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=1,
        primal_lr_mode="spectral",
        spectral_step_fraction=0.75,
        dual_lr=0.2,
        dual_init_mode="spectral",
        dual_burn_in=10,
        dual_psd_margin=0.1,
        max_iters=0,
        verbose=False,
    )

    worst_multiplier = 1.0 - 2.0 * solver.primal_lr * (
        solver.primal_lr_lambda_max + solver.dual_init
    )
    assert np.isclose(worst_multiplier, -0.5)
    assert abs(worst_multiplier) < 1.0
    assert solver.primal_lr > 0.0


def test_float32_center_perturbation_uses_representable_delta():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
        n_vars=1,
        Q_indices=indices,
        Q_values=values,
        c=np.zeros(1, dtype=np.float32),
        batch_size=1,
        primal_lr=0.1,
        dual_lr=0.1,
        tolerance=1e-8,
        max_iters=0,
        primal_init="half",
        verbose=False,
    )
    solver.dual[...] = 0.0
    solver._paper_step()

    assert solver.delta >= float(np.spacing(np.float32(0.5)))
    assert solver.primal[0, 0] != np.float32(0.5)


def test_conditional_rounding_beats_best_relaxed_expectation():
    graph = random_graph(n=20, d=3, seed=4)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=4,
        max_iters=20,
        conditional_rounding=True,
        seed=2,
        verbose=False,
    )

    result = solver.optimize()

    assert result.conditional_rounding_expected_objective is not None
    assert result.objective <= result.conditional_rounding_expected_objective + 1e-6
    assert np.all((result.incumbent == 0) | (result.incumbent == 1))


def test_conditional_rounding_handles_diagonal_and_nonsymmetric_qubo():
    indices = np.array(
        [[0, 0, 1, 1], [0, 1, 0, 1]],
        dtype=np.int32,
    )
    values = np.array([2.0, 3.0, -1.0, -4.0], dtype=np.float32)
    solver = PDBO_CPU(
        n_vars=2,
        Q_indices=indices,
        Q_values=values,
        c=np.array([1.0, 2.0], dtype=np.float32),
        max_iters=0,
        verbose=False,
    )
    probabilities = np.array([[0.2, 0.7]], dtype=np.float64)
    expected = float(solver._multilinear_values(probabilities)[0])
    candidate = solver._conditional_expectation_candidate(probabilities[0])
    objective = float(solver._score_candidates(candidate[np.newaxis, :])[0])

    assert objective <= expected + 1e-10


def test_delta_must_keep_center_perturbation_inside_box():
    indices, values = empty_qubo()
    with np.testing.assert_raises(ValueError):
        PDBO_CPU(
            n_vars=1,
            Q_indices=indices,
            Q_values=values,
            tolerance=0.5,
            verbose=False,
        )


def test_spectral_window_detects_minimum_eigenspace_multiplicity():
    indices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    values = np.array([-2.0, -2.0, 1.0], dtype=np.float32)
    solver = PDBO_CPU(
        n_vars=3,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        verbose=False,
    )

    distribution = compute_spectrum_distribution(solver.W, solver.n, solver.seed)
    lambda_1, lambda_next, multiplicity = compute_spectral_window(
        solver.W,
        solver.n,
        solver.seed,
        distribution.eigenvalues,
    )
    assert distribution.mode == "exact"
    assert np.isclose(np.sum(distribution.counts), 3.0)
    assert np.isclose(lambda_1, -2.0)
    assert np.isclose(lambda_next, 1.0)
    assert multiplicity == 2


def test_spectral_animation_bins_mean_mode_lengths():
    indices = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    values = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    solver = PDBO_CPU(
        n_vars=4,
        Q_indices=indices,
        Q_values=values,
        batch_size=1,
        max_iters=0,
        spectral_animation=True,
        spectral_animation_bins=2,
        verbose=False,
    )
    solver.primal[...] = [[0.6, 0.3, 0.8, 0.1]]
    distribution = compute_spectrum_distribution(
        solver.W,
        solver.n,
        solver.seed,
        include_eigenvectors=True,
    )
    animation = SpectralAnimation(
        distribution.eigenvalues,
        distribution.eigenvectors,
        bins=2,
    )
    edges = histogram_edges(-2.0, 2.0, 2)
    animation.bin_indices = np.clip(
        np.searchsorted(edges, distribution.eigenvalues, side="right") - 1,
        0,
        1,
    )

    assert np.allclose(animation.mode_bin_means(solver.primal), [0.15, 0.35])


def test_h_s_spectrum_certificate_is_exact_on_single_edge():
    graph = random_graph(n=2, d=1, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=2,
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        max_iters=0,
        verbose=False,
    )

    optimal = compute_h_s_spectrum_certificate(
        solver.W,
        np.array([-1.0, 1.0]),
        seed=0,
        bins=2,
    )
    uncut = compute_h_s_spectrum_certificate(
        solver.W,
        np.array([1.0, 1.0]),
        seed=0,
        bins=2,
    )

    assert optimal.distribution.mode == "exact"
    assert np.allclose(optimal.distribution.eigenvalues, [0.0, 2.0])
    assert np.isclose(optimal.gap_upper_bound, 0.0)
    assert np.allclose(uncut.distribution.eigenvalues, [-2.0, 0.0])
    assert np.isclose(uncut.gap_upper_bound, 1.0)


def test_a_t_animation_payload_has_no_terminal_certificate_log(capsys):
    graph = random_graph(n=2, d=1, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=2,
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        max_iters=0,
        maxcut_certificate=True,
        spectral_animation_bins=2,
        verbose=False,
    )

    payload = solver._a_t_animation_payload()
    first_certificate = solver._current_incumbent_certificate()
    second_certificate = solver._current_incumbent_certificate()
    output = capsys.readouterr().out

    assert payload["a_t_counts"].shape == (2,)
    assert np.isclose(payload["a_t_kappa"], max(-payload["a_t_lambda_min"], 0.0))
    assert np.isclose(payload["a_t_gap_bound"], solver.n * payload["a_t_kappa"] / 4.0)
    assert np.isclose(
        payload["incumbent_cert_kappa"],
        max(-first_certificate.lambda_min, 0.0),
    )
    assert first_certificate is second_certificate
    assert np.isclose(
        payload["incumbent_cert_gap_bound"],
        solver.n * max(-first_certificate.lambda_min, 0.0) / 4.0,
    )
    assert "h_s_certificate" not in output


def test_large_spectral_animation_uses_approximate_modes_without_size_limit():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
        n_vars=2001,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        spectral_animation=True,
        spectral_animation_modes=8,
        verbose=False,
    )

    assert solver.spectral_animation_modes == 8


def test_lanczos_animation_modes_are_orthonormal_ritz_vectors():
    indices = np.vstack((np.arange(20), np.arange(20))).astype(np.int32)
    values = np.linspace(-3.0, 4.0, 20, dtype=np.float32)
    solver = PDBO_CPU(
        n_vars=20,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        verbose=False,
    )

    eigenvalues, eigenvectors = compute_animation_modes(
        solver.W, solver.n, seed=3, modes=8
    )
    projected = eigenvectors.T @ (solver.W @ eigenvectors)

    assert eigenvalues.shape == (8,)
    assert eigenvectors.shape == (20, 8)
    assert np.all(np.diff(eigenvalues) >= 0.0)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(8), atol=1e-5)
    assert np.allclose(np.diag(projected), eigenvalues, atol=1e-5)


def test_spectral_animation_control_state():
    animation = SpectralAnimation(np.array([-1.0, 1.0]), np.eye(2), bins=2)

    assert not animation.run_continuous
    animation.request_step()
    assert animation.step_requested
    animation.toggle_run()
    assert animation.run_continuous


def test_animation_objective_means_use_current_primal_dual_state():
    indices = np.array([[0, 1], [0, 1]], dtype=np.int32)
    values = np.array([1.0, 2.0], dtype=np.float32)
    solver = PDBO_CPU(
        n_vars=2,
        Q_indices=indices,
        Q_values=values,
        c=np.array([3.0, 4.0], dtype=np.float32),
        batch_size=1,
        max_iters=0,
        verbose=False,
    )
    solver.primal[...] = [[0.25, 0.75]]
    solver.dual[...] = [[5.0, 6.0]]

    lagrangian_value, objective_value = solver._animation_objective_means()

    assert np.isclose(objective_value, 4.9375)
    assert np.isclose(lagrangian_value, 2.875)


def test_animation_objective_history_uses_global_iteration_and_deduplicates_refreshes():
    animation = SpectralAnimation(np.array([-1.0, 1.0]), np.eye(2), bins=2)

    animation._update_objective_history(1, 5, 5, 3.0, 4.0)
    animation._update_objective_history(1, 5, 5, 2.0, 3.0)
    animation._update_objective_history(2, 0, 5, 1.0, 2.0)

    assert animation.objective_iterations == [5, 5]
    assert animation.lagrangian_history == [2.0, 1.0]
    assert animation.objective_history == [3.0, 2.0]


def test_spectral_animation_every_controls_continuous_refresh():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
        n_vars=2,
        Q_indices=indices,
        Q_values=values,
        max_iters=0,
        spectral_animation_every=3,
        verbose=False,
    )

    class RecordingAnimation:
        run_continuous = True

        def __init__(self):
            self.iterations = []

        def update(self, _primal, _phase, _phases, iteration):
            self.iterations.append(iteration)

    animation = RecordingAnimation()
    solver.animation = animation
    for iteration in range(1, 4):
        solver._update_animation(1, 1, iteration)
    assert animation.iterations == [3]

    animation.run_continuous = False
    solver._update_animation(1, 1, 4)
    assert animation.iterations == [3, 4]


def test_removed_quadratic_backend_is_not_exposed_by_cli():
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--quadratic_backend" not in option_strings
    assert "--spectral_animation_modes" in option_strings


def test_paper_update_uses_old_primal_for_dual():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
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
    solver = PDBO_CPU(
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
    solver = PDBO_CPU(
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
    solver = PDBO_CPU(
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
    solver = PDBO_CPU(
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
    solver = PDBO_CPU(
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


def test_progress_logging_excludes_maxcut_bounds(capsys):
    graph = random_graph(n=4, d=1, seed=0)
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=1,
        max_iters=1,
        check_every=1,
        verbose=True,
        seed=0,
    )

    solver.optimize()
    output = capsys.readouterr().out

    assert "iter=0/1 best=" in output
    assert "iter=1/1 best=" in output
    assert output.count("dual_mean=") == 2
    assert "U_spec=" not in output
    assert "CutLB=" not in output
    assert "UB=" not in output
    assert "BestUB=" not in output
    assert "Gap=" not in output
    assert "ell=" not in output


def test_dual_patience_stops_on_small_mean_dual_decrease():
    indices, values = empty_qubo()
    solver = PDBO_CPU(
        n_vars=1,
        Q_indices=indices,
        Q_values=values,
        max_iters=5,
        primal_init="half",
        dual_init=1.0,
        dual_lr=0.1,
        dual_patience_threshold=0.1,
        dual_patience_every=1,
        verbose=False,
    )

    result = solver.optimize()

    assert result.stop_reason == "dual_patience"
    assert solver.iterations_completed == 1


def test_cli_smoke():
    proc = subprocess.run(
        [
            sys.executable,
            "src/main.py",
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "best=" in proc.stdout
