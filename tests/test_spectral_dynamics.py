import numpy as np

from research.g3_spectral_dynamics import compute_duhamel_history, select_major_modes
from research.gset_duhamel_sweep import summarize_result


def test_duhamel_history_reconstructs_forced_scalar_modes():
    eigenvalues = np.array([-2.0, 1.0])
    eigenvectors = np.eye(2)
    initial = np.array([0.5, -0.25])
    mean_dual = np.array([3.0, 1.5, 0.0])
    forcing = np.array([[0.1, -0.2], [0.05, 0.3], [-0.4, 0.1]])
    alpha = 0.1

    state = initial.copy()
    states = [state.copy()]
    for dual, step_forcing in zip(mean_dual, forcing):
        multiplier = 1.0 - 2.0 * alpha * (eigenvalues + dual)
        state = multiplier * state + step_forcing
        states.append(state.copy())
    actual = np.asarray(states)[[0, 2, 3]]

    history = compute_duhamel_history(
        alpha=alpha,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        initial_coefficients=initial,
        mean_dual_history=mean_dual,
        forcing_history=forcing,
        sample_times=np.array([0, 2, 3]),
        actual_coefficients=actual,
        step_coefficients=np.asarray(states),
    )

    assert np.allclose(history["reconstruction"], actual)
    assert np.allclose(history["identity_residual"], 0.0)
    assert np.all(history["absolute_error"] <= history["forcing_radius"] + 1e-14)


def test_major_modes_include_low_spectrum_and_peak_energy_modes():
    eigenvalues = np.arange(5, dtype=float)
    coefficients = np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 2.0]])

    selected, peak_energy = select_major_modes(eigenvalues, coefficients, 2, 2)

    assert np.array_equal(selected, np.array([0, 1, 3, 4]))
    assert np.array_equal(peak_energy, np.array([0.0, 0.0, 1.0, 9.0, 4.0]))


def test_sweep_summary_uses_samples_before_and_at_first_clip():
    summary = {
        "parameters": {"gset_id": 7, "initialization": "center-uniform"},
        "n": 4,
        "lambda_min": -2.0,
        "lambda_max": 3.0,
        "effective_initial_dual": 3.0,
        "initial_convex_condition": True,
        "positive_multiplier_condition": True,
        "first_clip": 20,
        "first_prediction_error_above_10pct": 30,
        "first_forcing_relative_above_1pct": 40,
        "best_threshold_cut": 5.0,
        "spectral_upper": 6.0,
        "spectral_certified_ratio": 5.0 / 6.0,
        "duhamel": {
            "selected_mode_count": 2,
            "max_relative_identity_residual": 1e-14,
            "max_forcing_projection_discrepancy": 2e-14,
            "max_bound_violation": 0.0,
            "max_relative_bound_violation": 0.0,
        },
    }
    rows = []
    for time, epsilons, energies in [
        (0, [0.0, 0.0], [0.1, 0.1]),
        (10, [0.01, 0.2], [0.7, 0.1]),
        (20, [0.2, 2.0], [0.8, 0.1]),
        (30, [2.0, 3.0], [0.2, 0.3]),
    ]:
        rows.extend(
            {
                "time": str(time),
                "epsilon": str(epsilon),
                "mode_energy_fraction": str(energy),
                "selected_modes_energy_fraction": str(sum(energies)),
            }
            for epsilon, energy in zip(epsilons, energies)
        )

    scalar_rows = [
        {"time": str(time), "archive_best_cut": str(cut)}
        for time, cut in [(0, 2.0), (10, 4.0), (20, 4.9), (30, 5.0)]
    ]
    result = summarize_result(summary, rows, scalar_rows)

    assert result["preclip_time"] == 10
    assert result["preclip_epsilon_lt_0_1"] == 1
    assert np.isclose(result["preclip_energy_fraction_epsilon_lt_0_1"], 0.7)
    assert np.isclose(result["preclip_selected_energy_share_epsilon_lt_1"], 1.0)
    assert result["clip_sample_time"] == 20
    assert result["clip_epsilon_lt_1"] == 1
    assert result["final_epsilon_lt_1"] == 0
    assert result["time_to_99pct_final_cut"] == 30
