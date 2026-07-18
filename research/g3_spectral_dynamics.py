#!/usr/bin/env python3
"""Measure full-spectrum PDBO dynamics on a Gset Max-Cut instance.

The script keeps the solver implementation unchanged.  It records the scalar
mean-dual spectral propagator, nonlinear forcing, spectral activation troughs,
and the two terms in the Bernoulli-rounding quality bound.

This diagnostic uses a dense eigendecomposition and is intended for Gset-scale
instances such as G3 (n=800), not the largest graphs in the collection.  JSON
fields beginning with ``first_`` report the first sampled occurrence, with time
resolution ``sample_every``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as dense_linalg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from src.solver import PDBO_CPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gset-id", type=int, default=3)
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/pdbo-g3-spectrum"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def binned_sum(values: np.ndarray, indices: np.ndarray, bins: int) -> np.ndarray:
    return np.bincount(indices, weights=values, minlength=bins).astype(np.float64)


def select_major_modes(
    eigenvalues: np.ndarray,
    coefficient_history: np.ndarray,
    low_count: int,
    energy_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select low-spectrum modes and modes with the largest observed energy."""
    n = eigenvalues.size
    low_indices = np.arange(min(low_count, n), dtype=np.int64)
    peak_energy = np.max(coefficient_history ** 2, axis=0)
    energy_indices = np.argsort(-peak_energy, kind="stable")[: min(energy_count, n)]
    selected = np.unique(np.concatenate((low_indices, energy_indices)))
    return selected, peak_energy


def compute_duhamel_history(
    *,
    alpha: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    initial_coefficients: np.ndarray,
    mean_dual_history: np.ndarray,
    forcing_history: np.ndarray,
    sample_times: np.ndarray,
    actual_coefficients: np.ndarray,
    step_coefficients: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the per-mode discrete Duhamel formula at sampled times."""
    steps = mean_dual_history.size
    if forcing_history.shape[0] != steps:
        raise ValueError("forcing and mean-dual histories must have the same length")
    if sample_times[0] != 0 or sample_times[-1] > steps:
        raise ValueError("sample times must start at zero and lie within the history")

    coordinate_forcing_projection = forcing_history @ eigenvectors
    multipliers = 1.0 - 2.0 * alpha * (
        eigenvalues[np.newaxis, :] + mean_dual_history[:, np.newaxis]
    )
    if step_coefficients is None:
        modal_forcing = coordinate_forcing_projection
    else:
        if step_coefficients.shape != (steps + 1, eigenvalues.size):
            raise ValueError("step coefficients must contain every mode at every step")
        modal_forcing = (
            step_coefficients[1:] - multipliers * step_coefficients[:-1]
        )
    samples = sample_times.size
    modes = eigenvalues.size
    propagator_history = np.empty((samples, modes), dtype=np.float64)
    forcing_sum_history = np.empty_like(propagator_history)
    radius_history = np.empty_like(propagator_history)
    previous_forcing_history = np.zeros_like(propagator_history)

    propagator = np.ones(modes, dtype=np.float64)
    forcing_sum = np.zeros(modes, dtype=np.float64)
    radius = np.zeros(modes, dtype=np.float64)
    propagator_history[0] = propagator
    forcing_sum_history[0] = forcing_sum
    radius_history[0] = radius

    sample_index = 1
    for step in range(steps):
        multiplier = multipliers[step]
        projected_forcing = modal_forcing[step]
        propagator *= multiplier
        forcing_sum = multiplier * forcing_sum + projected_forcing
        radius = np.abs(multiplier) * radius + np.abs(projected_forcing)
        time_index = step + 1
        if sample_index < samples and time_index == sample_times[sample_index]:
            propagator_history[sample_index] = propagator
            forcing_sum_history[sample_index] = forcing_sum
            radius_history[sample_index] = radius
            previous_forcing_history[sample_index] = projected_forcing
            sample_index += 1

    if sample_index != samples:
        raise ValueError("sample times must be strictly increasing integer steps")

    homogeneous = propagator_history * initial_coefficients[np.newaxis, :]
    stable_forcing_sum = actual_coefficients - homogeneous
    reconstruction = homogeneous + forcing_sum_history
    absolute_error = np.abs(actual_coefficients - homogeneous)
    denominator = np.abs(homogeneous)
    epsilon = np.divide(
        radius_history,
        denominator,
        out=np.full_like(radius_history, np.inf),
        where=denominator > 0.0,
    )
    epsilon[(radius_history == 0.0) & (denominator == 0.0)] = 0.0
    term_scale = (
        np.abs(actual_coefficients)
        + np.abs(homogeneous)
        + np.abs(forcing_sum_history)
    )
    identity_residual = actual_coefficients - reconstruction
    return {
        "propagator": propagator_history,
        "homogeneous": homogeneous,
        "forcing_sum": stable_forcing_sum,
        "recursive_forcing_sum": forcing_sum_history,
        "forcing_radius": radius_history,
        "previous_forcing": previous_forcing_history,
        "reconstruction": reconstruction,
        "identity_residual": identity_residual,
        "relative_identity_residual": np.divide(
            np.abs(identity_residual),
            term_scale,
            out=np.zeros_like(identity_residual),
            where=term_scale > 0.0,
        ),
        "cancellation_condition": np.divide(
            np.abs(homogeneous) + radius_history,
            np.abs(actual_coefficients),
            out=np.full_like(radius_history, np.inf),
            where=np.abs(actual_coefficients) > 0.0,
        ),
        "forcing_projection_discrepancy": (
            modal_forcing - coordinate_forcing_projection
        ),
        "absolute_error": absolute_error,
        "epsilon": epsilon,
    }


def manual_step(solver: PDBO_CPU) -> tuple[dict[str, float], np.ndarray]:
    old_primal = np.asarray(solver.primal, dtype=np.float64)
    old_dual = np.asarray(solver.dual, dtype=np.float64)
    _, objective_gradient = solver._objective_values_and_gradients(old_primal)
    constraint_value, constraint_gradient = solver._constraint_values_and_gradients(old_primal)
    gradient = np.asarray(objective_gradient, dtype=np.float64) + old_dual * constraint_gradient

    raw_primal = old_primal - solver.primal_lr * gradient
    projected = np.clip(raw_primal, 0.0, 1.0)
    if solver.g_type == "absolute":
        at_perturbation_point = old_primal == 0.5
    else:
        at_perturbation_point = np.abs(old_primal - 0.5) <= solver.delta
    trigger = (
        at_perturbation_point
        & (np.abs(gradient) <= 2.0 * solver.delta)
        & (old_dual <= 0.0)
    )
    perturbed = np.where(old_primal <= 0.5, 0.5 - solver.delta, 0.5 + solver.delta)
    new_primal = np.where(trigger, perturbed, projected)
    new_dual = old_dual + solver.dual_lr * constraint_value

    old_z = old_primal - 0.5
    stored_primal = new_primal.astype(np.float32, copy=False)
    stored_dual = new_dual.astype(np.float32, copy=False)
    new_z = np.asarray(stored_primal, dtype=np.float64) - 0.5
    mean_dual = np.mean(old_dual, axis=1)
    wz = (solver.W @ old_z.T).T
    scalar_baseline = old_z - 2.0 * solver.primal_lr * (
        wz + mean_dual[:, np.newaxis] * old_z
    )
    forcing = new_z - scalar_baseline
    anisotropy_forcing = -2.0 * solver.primal_lr * (
        old_dual - mean_dual[:, np.newaxis]
    ) * old_z
    projection_forcing = forcing - anisotropy_forcing

    solver.primal = stored_primal
    solver.dual = stored_dual
    solver.perturbation_count += int(np.count_nonzero(trigger))
    solver._update_incumbent()

    def mean_l2(array: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(array, axis=1)))

    metrics = {
        "forcing_l2": mean_l2(forcing),
        "anisotropy_forcing_l2": mean_l2(anisotropy_forcing),
        "projection_forcing_l2": mean_l2(projection_forcing),
        "clip_fraction": float(np.mean((raw_primal < 0.0) | (raw_primal > 1.0))),
        "trigger_fraction": float(np.mean(trigger)),
    }
    return metrics, forcing


def main() -> None:
    from src.problem_parser import generate_Max_cut as generate_max_cut, parse_gset
    from src.solver import PDBO_CPU

    args = parse_args()
    if args.steps < 1 or args.sample_every < 1:
        raise ValueError("steps and sample-every must be positive")
    if args.major_low_modes < 0 or args.major_energy_modes < 0:
        raise ValueError("major mode counts must be non-negative")
    if args.major_low_modes + args.major_energy_modes == 0:
        raise ValueError("at least one major mode count must be positive")
    if not 0.0 < args.v1_max_abs < 0.5:
        raise ValueError("v1-max-abs must lie strictly between zero and 0.5")
    if args.v1_clip_steps < 1:
        raise ValueError("v1-clip-steps must be positive")
    if args.dual_psd_margin < 0.0:
        raise ValueError("dual-psd-margin must be non-negative")
    graph = parse_gset(str(args.gset_id))
    data = generate_max_cut(graph)
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=args.batch,
        primal_lr=args.alpha,
        dual_lr=args.beta,
        dual_init=args.dual_init,
        max_iters=0,
        primal_init="center_uniform",
        rho=args.rho,
        seed=args.seed,
        verbose=False,
    )
    if not 0 <= args.trajectory_index < solver.batch_size:
        raise ValueError("trajectory-index must select one batch row")

    dense_w = solver.W.toarray().astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(dense_w)
    n = solver.n
    edge_weight = float(sum(float(item[2]["weight"]) for item in graph.edges(data=True)))
    spectral_upper = edge_weight / 2.0 - n * float(eigenvalues[0]) / 4.0
    spread = float(eigenvalues[-1] - eigenvalues[0])

    effective_initial_dual = args.dual_init
    if args.dual_init_mode == "psd-margin":
        effective_initial_dual = -float(eigenvalues[0]) + args.dual_psd_margin
        solver.dual.fill(effective_initial_dual)
    if args.initialization == "v1":
        v1 = eigenvectors[:, 0]
        v1_scale = args.v1_max_abs / float(np.max(np.abs(v1)))
        initial_v1 = v1_scale * v1
        batch_signs = np.where(np.arange(solver.batch_size) % 2 == 0, 1.0, -1.0)
        solver.primal = (
            0.5 + batch_signs[:, np.newaxis] * initial_v1[np.newaxis, :]
        ).astype(np.float32)
        if args.v1_dual_mode == "psd-boundary":
            effective_initial_dual = -float(eigenvalues[0])
        elif args.v1_dual_mode == "clip-threshold":
            target_multiplier = (0.5 / args.v1_max_abs) ** (
                1.0 / args.v1_clip_steps
            )
            effective_initial_dual = (
                (1.0 - target_multiplier) / (2.0 * args.alpha)
                - float(eigenvalues[0])
            )
        solver.dual.fill(effective_initial_dual)

    bin_edges = np.linspace(eigenvalues[0], eigenvalues[-1], args.bins + 1)
    bin_indices = np.clip(
        np.searchsorted(bin_edges, eigenvalues, side="right") - 1,
        0,
        args.bins - 1,
    )
    bin_counts = np.bincount(bin_indices, minlength=args.bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    initial_z = np.asarray(solver.primal, dtype=np.float64) - 0.5
    initial_coefficients = initial_z @ eigenvectors
    log_abs_propagator = np.zeros_like(initial_coefficients)
    propagator_sign = np.ones_like(initial_coefficients)

    sample_times: list[int] = []
    scalar_rows: list[list[float]] = []
    band_rms_rows: list[np.ndarray] = []
    band_energy_rows: list[np.ndarray] = []
    band_predicted_rms_rows: list[np.ndarray] = []
    mode_rms_rows: list[np.ndarray] = []
    mode_predicted_rms_rows: list[np.ndarray] = []
    mode_residual_rms_rows: list[np.ndarray] = []
    trajectory_coefficient_rows: list[np.ndarray] = []
    trajectory_forcing_rows: list[np.ndarray] = []
    trajectory_mean_dual_rows: list[float] = []
    trajectory_state_rows: list[np.ndarray] = [
        initial_z[args.trajectory_index].copy()
    ]
    best_cut = 0.0
    zero_forcing = {
        "forcing_l2": 0.0,
        "anisotropy_forcing_l2": 0.0,
        "projection_forcing_l2": 0.0,
        "clip_fraction": 0.0,
        "trigger_fraction": 0.0,
    }

    scalar_names = [
        "time",
        "theta_mean",
        "theta_std",
        "theta_min",
        "theta_max",
        "fractionality",
        "dual_anisotropy_rms",
        "dual_anisotropy_inf",
        "forcing_l2",
        "anisotropy_forcing_l2",
        "projection_forcing_l2",
        "forcing_relative",
        "clip_fraction",
        "trigger_fraction",
        "prediction_relative_error",
        "rayleigh",
        "weighted_spectral_excess",
        "flatness",
        "weighted_tail",
        "saturation_deficit",
        "bernoulli_gap_bound",
        "bernoulli_expected_cut",
        "current_best_threshold_cut",
        "archive_best_cut",
        "threshold_spectral_gap_bound",
        "spectral_entropy",
        "effective_rank",
    ]

    def sample(time_index: int, forcing_metrics: dict[str, float]) -> None:
        nonlocal best_cut
        z = np.asarray(solver.primal, dtype=np.float64) - 0.5
        dual = np.asarray(solver.dual, dtype=np.float64)
        coefficients = z @ eigenvectors
        mode_energy = np.mean(coefficients ** 2, axis=0)
        total_energy = float(np.sum(mode_energy))
        if total_energy > 0.0:
            spectral_probability = mode_energy / total_energy
        else:
            spectral_probability = np.zeros_like(mode_energy)
        mode_rms = np.sqrt(mode_energy)

        predicted_coefficients = initial_coefficients * propagator_sign * np.exp(
            np.clip(log_abs_propagator, -700.0, 700.0)
        )
        predicted_mode_rms = np.sqrt(np.mean(predicted_coefficients ** 2, axis=0))
        mode_residual_rms = np.sqrt(
            np.mean((coefficients - predicted_coefficients) ** 2, axis=0)
        )
        prediction_error = float(
            np.linalg.norm(coefficients - predicted_coefficients)
            / max(np.linalg.norm(coefficients), 1e-15)
        )

        band_energy = binned_sum(mode_energy, bin_indices, args.bins)
        band_predicted_energy = binned_sum(
            predicted_mode_rms ** 2, bin_indices, args.bins
        )
        band_rms = np.sqrt(
            np.divide(
                band_energy,
                bin_counts,
                out=np.zeros(args.bins, dtype=np.float64),
                where=bin_counts > 0,
            )
        )
        band_predicted_rms = np.sqrt(
            np.divide(
                band_predicted_energy,
                bin_counts,
                out=np.zeros(args.bins, dtype=np.float64),
                where=bin_counts > 0,
            )
        )
        if total_energy > 0.0:
            band_energy_fraction = band_energy / total_energy
        else:
            band_energy_fraction = np.zeros_like(band_energy)

        batch_mean_dual = np.mean(dual, axis=1)
        theta = -batch_mean_dual
        anisotropy = dual - batch_mean_dual[:, np.newaxis]
        fractionality = 1.0 - 4.0 * np.mean(z ** 2, axis=1)
        norms = np.linalg.norm(z, axis=1)
        normalized = np.divide(
            np.sqrt(n) * z,
            norms[:, np.newaxis],
            out=np.zeros_like(z),
            where=norms[:, np.newaxis] > 0.0,
        )
        flatness_per_batch = np.linalg.norm(np.abs(normalized) - 1.0, axis=1)

        rayleigh = float(np.sum(spectral_probability * eigenvalues))
        weighted_excess = float(
            np.sum(spectral_probability * (eigenvalues - eigenvalues[0]))
        )
        entropy = float(
            -np.sum(
                spectral_probability[spectral_probability > 0.0]
                * np.log(spectral_probability[spectral_probability > 0.0])
            )
        )

        batch_z_energy = np.sum(coefficients ** 2, axis=1)
        batch_weighted_tail = np.sum(
            coefficients ** 2 * (eigenvalues - eigenvalues[0]), axis=1
        )
        batch_saturation_deficit = (-eigenvalues[0]) * (
            n / 4.0 - batch_z_energy
        )
        batch_bernoulli_gap = batch_weighted_tail + batch_saturation_deficit
        batch_bernoulli_expected_cut = spectral_upper - batch_bernoulli_gap

        bits = (z >= 0.0).astype(np.float32)
        cut_values = -solver._score_candidates(bits)
        current_best_cut = float(np.max(cut_values))
        best_cut = max(best_cut, current_best_cut)
        sign_vectors = 2.0 * bits - 1.0
        sign_rayleigh = np.einsum("bi,ij,bj->b", sign_vectors, dense_w, sign_vectors)
        threshold_gap_bound = 0.25 * (sign_rayleigh - n * eigenvalues[0])

        forcing_relative = forcing_metrics["forcing_l2"] / max(
            float(np.mean(norms)), 1e-15
        )
        scalar_rows.append(
            [
                float(time_index),
                float(np.mean(theta)),
                float(np.std(theta)),
                float(np.min(theta)),
                float(np.max(theta)),
                float(np.mean(fractionality)),
                float(np.sqrt(np.mean(anisotropy ** 2))),
                float(np.max(np.abs(anisotropy))),
                forcing_metrics["forcing_l2"],
                forcing_metrics["anisotropy_forcing_l2"],
                forcing_metrics["projection_forcing_l2"],
                forcing_relative,
                forcing_metrics["clip_fraction"],
                forcing_metrics["trigger_fraction"],
                prediction_error,
                rayleigh,
                weighted_excess,
                float(np.mean(flatness_per_batch)),
                float(np.mean(batch_weighted_tail)),
                float(np.mean(batch_saturation_deficit)),
                float(np.mean(batch_bernoulli_gap)),
                float(np.mean(batch_bernoulli_expected_cut)),
                current_best_cut,
                best_cut,
                float(np.min(threshold_gap_bound)),
                entropy,
                float(np.exp(entropy)),
            ]
        )
        sample_times.append(time_index)
        band_rms_rows.append(band_rms)
        band_energy_rows.append(band_energy_fraction)
        band_predicted_rms_rows.append(band_predicted_rms)
        mode_rms_rows.append(mode_rms)
        mode_predicted_rms_rows.append(predicted_mode_rms)
        mode_residual_rms_rows.append(mode_residual_rms)
        trajectory_coefficient_rows.append(coefficients[args.trajectory_index].copy())

    solver._update_incumbent()
    sample(0, zero_forcing)
    last_forcing = zero_forcing
    first_clip_exact = -1
    for step in range(args.steps):
        old_dual = np.asarray(solver.dual, dtype=np.float64)
        old_mean_dual = np.mean(old_dual, axis=1)
        multipliers = 1.0 - 2.0 * args.alpha * (
            eigenvalues[np.newaxis, :] + old_mean_dual[:, np.newaxis]
        )
        propagator_sign *= np.sign(multipliers)
        log_abs_propagator += np.log(np.maximum(np.abs(multipliers), 1e-300))
        last_forcing, forcing = manual_step(solver)
        if first_clip_exact < 0 and last_forcing["clip_fraction"] > 0.0:
            first_clip_exact = step + 1
        trajectory_forcing_rows.append(forcing[args.trajectory_index].copy())
        trajectory_mean_dual_rows.append(float(old_mean_dual[args.trajectory_index]))
        trajectory_state_rows.append(
            np.asarray(solver.primal[args.trajectory_index], dtype=np.float64) - 0.5
        )
        best_cut = max(best_cut, -float(solver.objective))
        time_index = step + 1
        if time_index % args.sample_every == 0 or time_index == args.steps:
            sample(time_index, last_forcing)

    scalar = np.asarray(scalar_rows, dtype=np.float64)
    band_rms = np.asarray(band_rms_rows, dtype=np.float64)
    band_energy = np.asarray(band_energy_rows, dtype=np.float64)
    band_predicted_rms = np.asarray(band_predicted_rms_rows, dtype=np.float64)
    mode_rms = np.asarray(mode_rms_rows, dtype=np.float64)
    mode_predicted_rms = np.asarray(mode_predicted_rms_rows, dtype=np.float64)
    mode_residual_rms = np.asarray(mode_residual_rms_rows, dtype=np.float64)
    trajectory_coefficients = np.asarray(
        trajectory_coefficient_rows, dtype=np.float64
    )
    trajectory_forcing = np.asarray(trajectory_forcing_rows, dtype=np.float64)
    trajectory_mean_dual = np.asarray(trajectory_mean_dual_rows, dtype=np.float64)
    trajectory_states = np.asarray(trajectory_state_rows, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.int64)

    selected_modes, peak_mode_energy = select_major_modes(
        eigenvalues,
        trajectory_coefficients,
        args.major_low_modes,
        args.major_energy_modes,
    )
    selected_eigenvalues = eigenvalues[selected_modes]
    selected_actual = trajectory_coefficients[:, selected_modes]
    step_coefficients = trajectory_states @ eigenvectors[:, selected_modes]
    duhamel = compute_duhamel_history(
        alpha=args.alpha,
        eigenvalues=selected_eigenvalues,
        eigenvectors=eigenvectors[:, selected_modes],
        initial_coefficients=initial_coefficients[
            args.trajectory_index, selected_modes
        ],
        mean_dual_history=trajectory_mean_dual,
        forcing_history=trajectory_forcing,
        sample_times=times,
        actual_coefficients=selected_actual,
        step_coefficients=step_coefficients,
    )

    trough_indices = np.argmin(
        np.where(bin_counts[np.newaxis, :] > 0, band_rms, np.inf), axis=0
    )
    trough_times = times[trough_indices]
    crossing_times = np.full(args.bins, -1, dtype=np.int64)
    theta_series = scalar[:, scalar_names.index("theta_mean")]
    for index, center in enumerate(bin_centers):
        hits = np.flatnonzero(theta_series >= center)
        if hits.size:
            crossing_times[index] = int(times[hits[0]])

    rayleigh_series = scalar[:, scalar_names.index("rayleigh")]
    bernoulli_gap_series = scalar[:, scalar_names.index("bernoulli_gap_bound")]
    forcing_series = scalar[:, scalar_names.index("forcing_relative")]
    clip_series = scalar[:, scalar_names.index("clip_fraction")]
    prediction_series = scalar[:, scalar_names.index("prediction_relative_error")]
    nonempty = np.flatnonzero(bin_counts > 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duhamel_csv_path = args.output.parent / f"{args.output.name}-duhamel.csv"
    duhamel_summary_csv_path = (
        args.output.parent / f"{args.output.name}-duhamel-summary.csv"
    )
    scalar_csv_path = args.output.parent / f"{args.output.name}-scalars.csv"
    with scalar_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(scalar_names)
        writer.writerows(scalar)
    low_mode_set = set(range(min(args.major_low_modes, n)))
    energy_mode_order = np.argsort(-peak_mode_energy, kind="stable")
    energy_mode_set = set(energy_mode_order[: min(args.major_energy_modes, n)].tolist())
    total_sample_energy = np.sum(trajectory_coefficients ** 2, axis=1)
    selected_energy_fraction = np.divide(
        np.sum(selected_actual ** 2, axis=1),
        total_sample_energy,
        out=np.zeros(times.size, dtype=np.float64),
        where=total_sample_energy > 0.0,
    )

    def selection_reason(mode_index: int) -> str:
        is_low = mode_index in low_mode_set
        is_energy = mode_index in energy_mode_set
        if is_low and is_energy:
            return "low+peak_energy"
        return "low" if is_low else "peak_energy"

    duhamel_fields = [
        "trajectory_index",
        "mode_index",
        "selection_reason",
        "eigenvalue",
        "time",
        "initial_coefficient",
        "actual_coefficient",
        "propagator",
        "homogeneous_coefficient",
        "net_forcing_injection",
        "previous_step_forcing_projection",
        "duhamel_reconstruction",
        "identity_residual",
        "relative_identity_residual",
        "cancellation_condition",
        "absolute_prediction_error",
        "forcing_radius_R",
        "epsilon",
        "shadowing",
        "mode_energy_fraction",
        "selected_modes_energy_fraction",
    ]
    with duhamel_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=duhamel_fields)
        writer.writeheader()
        for selected_index, mode_index in enumerate(selected_modes):
            initial_coefficient = initial_coefficients[
                args.trajectory_index, mode_index
            ]
            mode_energy_fraction = np.divide(
                selected_actual[:, selected_index] ** 2,
                total_sample_energy,
                out=np.zeros(times.size, dtype=np.float64),
                where=total_sample_energy > 0.0,
            )
            for sample_index, time_index in enumerate(times):
                epsilon = duhamel["epsilon"][sample_index, selected_index]
                writer.writerow(
                    {
                        "trajectory_index": args.trajectory_index,
                        "mode_index": int(mode_index),
                        "selection_reason": selection_reason(int(mode_index)),
                        "eigenvalue": selected_eigenvalues[selected_index],
                        "time": int(time_index),
                        "initial_coefficient": initial_coefficient,
                        "actual_coefficient": selected_actual[
                            sample_index, selected_index
                        ],
                        "propagator": duhamel["propagator"][
                            sample_index, selected_index
                        ],
                        "homogeneous_coefficient": duhamel["homogeneous"][
                            sample_index, selected_index
                        ],
                        "net_forcing_injection": duhamel["forcing_sum"][
                            sample_index, selected_index
                        ],
                        "previous_step_forcing_projection": duhamel[
                            "previous_forcing"
                        ][sample_index, selected_index],
                        "duhamel_reconstruction": duhamel["reconstruction"][
                            sample_index, selected_index
                        ],
                        "identity_residual": duhamel["identity_residual"][
                            sample_index, selected_index
                        ],
                        "relative_identity_residual": duhamel[
                            "relative_identity_residual"
                        ][sample_index, selected_index],
                        "cancellation_condition": duhamel[
                            "cancellation_condition"
                        ][sample_index, selected_index],
                        "absolute_prediction_error": duhamel["absolute_error"][
                            sample_index, selected_index
                        ],
                        "forcing_radius_R": duhamel["forcing_radius"][
                            sample_index, selected_index
                        ],
                        "epsilon": epsilon,
                        "shadowing": int(epsilon < 1.0),
                        "mode_energy_fraction": mode_energy_fraction[sample_index],
                        "selected_modes_energy_fraction": selected_energy_fraction[
                            sample_index
                        ],
                    }
                )

    summary_fields = [
        "trajectory_index",
        "mode_index",
        "selection_reason",
        "eigenvalue",
        "peak_energy",
        "final_energy_fraction",
        "final_actual_coefficient",
        "final_homogeneous_coefficient",
        "final_net_forcing_injection",
        "final_forcing_radius_R",
        "final_epsilon",
        "final_shadowing",
        "shadowing_sample_fraction",
        "first_epsilon_ge_0_1",
        "first_epsilon_ge_1",
        "max_identity_residual",
        "max_relative_identity_residual",
        "max_cancellation_condition",
        "max_forcing_projection_discrepancy",
        "max_bound_violation",
    ]
    with duhamel_summary_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        for selected_index, mode_index in enumerate(selected_modes):
            epsilon_series = duhamel["epsilon"][:, selected_index]
            first_point_one = np.flatnonzero(epsilon_series >= 0.1)
            first_one = np.flatnonzero(epsilon_series >= 1.0)
            identity_residual = duhamel["identity_residual"][:, selected_index]
            bound_violation = (
                duhamel["absolute_error"][:, selected_index]
                - duhamel["forcing_radius"][:, selected_index]
            )
            writer.writerow(
                {
                    "trajectory_index": args.trajectory_index,
                    "mode_index": int(mode_index),
                    "selection_reason": selection_reason(int(mode_index)),
                    "eigenvalue": selected_eigenvalues[selected_index],
                    "peak_energy": peak_mode_energy[mode_index],
                    "final_energy_fraction": (
                        selected_actual[-1, selected_index] ** 2
                        / max(total_sample_energy[-1], 1e-300)
                    ),
                    "final_actual_coefficient": selected_actual[-1, selected_index],
                    "final_homogeneous_coefficient": duhamel["homogeneous"][
                        -1, selected_index
                    ],
                    "final_net_forcing_injection": duhamel["forcing_sum"][
                        -1, selected_index
                    ],
                    "final_forcing_radius_R": duhamel["forcing_radius"][
                        -1, selected_index
                    ],
                    "final_epsilon": epsilon_series[-1],
                    "final_shadowing": int(epsilon_series[-1] < 1.0),
                    "shadowing_sample_fraction": float(np.mean(epsilon_series < 1.0)),
                    "first_epsilon_ge_0_1": (
                        int(times[first_point_one[0]]) if first_point_one.size else -1
                    ),
                    "first_epsilon_ge_1": int(times[first_one[0]]) if first_one.size else -1,
                    "max_identity_residual": float(np.max(np.abs(identity_residual))),
                    "max_relative_identity_residual": float(
                        np.max(duhamel["relative_identity_residual"][:, selected_index])
                    ),
                    "max_cancellation_condition": float(
                        np.max(duhamel["cancellation_condition"][:, selected_index])
                    ),
                    "max_forcing_projection_discrepancy": float(
                        np.max(
                            np.abs(
                                duhamel["forcing_projection_discrepancy"][
                                    :, selected_index
                                ]
                            )
                        )
                    ),
                    "max_bound_violation": float(np.max(bound_violation)),
                }
            )

    npz_path = args.output.with_suffix(".npz")
    if args.save_npz:
        np.savez_compressed(
            npz_path,
            scalar=scalar,
            scalar_names=np.asarray(scalar_names),
            times=times,
            eigenvalues=eigenvalues,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            bin_counts=bin_counts,
            band_rms=band_rms,
            band_energy=band_energy,
            band_predicted_rms=band_predicted_rms,
            mode_rms=mode_rms,
            mode_predicted_rms=mode_predicted_rms,
            mode_residual_rms=mode_residual_rms,
            trajectory_index=np.asarray(args.trajectory_index),
            selected_modes=selected_modes,
            trajectory_coefficients=trajectory_coefficients,
            trajectory_mean_dual=trajectory_mean_dual,
            trajectory_forcing=trajectory_forcing,
            trajectory_states=trajectory_states,
            duhamel_propagator=duhamel["propagator"],
            duhamel_homogeneous=duhamel["homogeneous"],
            duhamel_forcing_sum=duhamel["forcing_sum"],
            duhamel_forcing_radius=duhamel["forcing_radius"],
            duhamel_epsilon=duhamel["epsilon"],
            duhamel_identity_residual=duhamel["identity_residual"],
            duhamel_relative_identity_residual=duhamel[
                "relative_identity_residual"
            ],
            duhamel_cancellation_condition=duhamel["cancellation_condition"],
            duhamel_forcing_projection_discrepancy=duhamel[
                "forcing_projection_discrepancy"
            ],
            final_primal=np.asarray(solver.primal, dtype=np.float64),
            final_dual=np.asarray(solver.dual, dtype=np.float64),
            incumbent=np.asarray(solver.incumbent, dtype=np.float64),
            trough_times=trough_times,
            crossing_times=crossing_times,
        )

    incumbent_bits = np.asarray(solver.incumbent, dtype=np.float64)
    incumbent_sign = 2.0 * incumbent_bits - 1.0
    incumbent_cut = float(-solver._score_candidates(incumbent_bits[np.newaxis, :])[0])
    incumbent_shift = -incumbent_sign * (dense_w @ incumbent_sign)
    incumbent_shift_matrix = dense_w + np.diag(incumbent_shift)
    incumbent_shift_lambda_min = float(
        dense_linalg.eigh(
            incumbent_shift_matrix,
            subset_by_index=(0, 0),
            eigvals_only=True,
            check_finite=False,
        )[0]
    )
    incumbent_shift_upper = incumbent_cut - n * incumbent_shift_lambda_min / 4.0

    final_dual = np.asarray(solver.dual, dtype=np.float64)
    final_dual_upper_bounds = []
    final_dual_lambda_mins = []
    for dual_row in final_dual:
        shifted_matrix = dense_w + np.diag(dual_row)
        lambda_min = float(
            dense_linalg.eigh(
                shifted_matrix,
                subset_by_index=(0, 0),
                eigvals_only=True,
                check_finite=False,
            )[0]
        )
        final_dual_lambda_mins.append(lambda_min)
        final_dual_upper_bounds.append(
            edge_weight / 2.0 + (float(np.sum(dual_row)) - n * lambda_min) / 4.0
        )
    best_diagonal_upper = min(final_dual_upper_bounds)
    best_available_upper = min(spectral_upper, best_diagonal_upper, incumbent_shift_upper)

    summary = {
        "parameters": vars(args) | {"output": str(args.output)},
        "n": n,
        "edge_weight": edge_weight,
        "lambda_min": float(eigenvalues[0]),
        "lambda_2": float(eigenvalues[1]),
        "lambda_max": float(eigenvalues[-1]),
        "effective_initial_dual": float(effective_initial_dual),
        "positive_multiplier_condition": bool(
            2.0 * args.alpha * (eigenvalues[-1] + effective_initial_dual) < 1.0
        ),
        "initial_convex_condition": bool(
            effective_initial_dual >= -eigenvalues[0]
        ),
        "spectral_upper": spectral_upper,
        "best_threshold_cut": best_cut,
        "incumbent_cut": incumbent_cut,
        "spectral_certified_ratio": incumbent_cut / spectral_upper,
        "incumbent_shift_lambda_min": incumbent_shift_lambda_min,
        "incumbent_shift_upper": incumbent_shift_upper,
        "best_final_dual_lambda_min": float(
            final_dual_lambda_mins[int(np.argmin(final_dual_upper_bounds))]
        ),
        "best_final_dual_upper": best_diagonal_upper,
        "best_available_upper": best_available_upper,
        "best_available_certified_ratio": incumbent_cut / best_available_upper,
        "duhamel": {
            "trajectory_index": args.trajectory_index,
            "selected_mode_count": int(selected_modes.size),
            "selected_modes": selected_modes.tolist(),
            "timeseries_csv": str(duhamel_csv_path),
            "summary_csv": str(duhamel_summary_csv_path),
            "max_identity_residual": float(
                np.max(np.abs(duhamel["identity_residual"]))
            ),
            "max_relative_identity_residual": float(
                np.max(duhamel["relative_identity_residual"])
            ),
            "max_cancellation_condition": float(
                np.max(duhamel["cancellation_condition"])
            ),
            "max_forcing_projection_discrepancy": float(
                np.max(np.abs(duhamel["forcing_projection_discrepancy"]))
            ),
            "max_bound_violation": float(
                np.max(duhamel["absolute_error"] - duhamel["forcing_radius"])
            ),
            "max_relative_bound_violation": float(
                np.max(
                    np.divide(
                        duhamel["absolute_error"] - duhamel["forcing_radius"],
                        np.maximum.reduce(
                            (
                                duhamel["absolute_error"],
                                duhamel["forcing_radius"],
                                np.ones_like(duhamel["absolute_error"]),
                            )
                        ),
                    )
                )
            ),
            "final_shadowing_mode_count": int(
                np.count_nonzero(duhamel["epsilon"][-1] < 1.0)
            ),
            "final_selected_energy_fraction": float(selected_energy_fraction[-1]),
        },
        "minimum_bernoulli_gap_bound": float(np.min(bernoulli_gap_series)),
        "minimum_bernoulli_gap_time": int(times[np.argmin(bernoulli_gap_series)]),
        "minimum_rayleigh": float(np.min(rayleigh_series)),
        "minimum_rayleigh_time": int(times[np.argmin(rayleigh_series)]),
        "rayleigh_increase_count": int(np.count_nonzero(np.diff(rayleigh_series) > 1e-10)),
        "first_forcing_relative_above_1pct": int(
            times[np.flatnonzero(forcing_series > 0.01)[0]]
            if np.any(forcing_series > 0.01)
            else -1
        ),
        "first_clip": int(
            times[np.flatnonzero(clip_series > 0.0)[0]]
            if np.any(clip_series > 0.0)
            else -1
        ),
        "first_clip_exact": int(first_clip_exact),
        "first_prediction_error_above_10pct": int(
            times[np.flatnonzero(prediction_series > 0.1)[0]]
            if np.any(prediction_series > 0.1)
            else -1
        ),
        "nonempty_bins": [
            {
                "index": int(index),
                "center": float(bin_centers[index]),
                "count": int(bin_counts[index]),
                "trough_time": int(trough_times[index]),
                "crossing_time": int(crossing_times[index]),
            }
            for index in nonempty
        ],
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.plot:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        log_band = np.log10(np.maximum(band_rms[:, nonempty], 1e-12)).T
        image = axes[0, 0].imshow(
            log_band,
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], 0, len(nonempty)),
            cmap="viridis",
        )
        axes[0, 0].set_yticks(np.arange(len(nonempty)) + 0.5)
        axes[0, 0].set_yticklabels([f"{bin_centers[i]:.1f}" for i in nonempty])
        axes[0, 0].set_title("log10 band RMS; labels are eigenvalue centers")
        axes[0, 0].set_xlabel("iteration")
        figure.colorbar(image, ax=axes[0, 0])

        axes[0, 1].plot(times, bernoulli_gap_series, label="Bernoulli gap bound")
        axes[0, 1].plot(
            times,
            scalar[:, scalar_names.index("weighted_tail")],
            label="weighted spectral tail",
        )
        axes[0, 1].plot(
            times,
            scalar[:, scalar_names.index("saturation_deficit")],
            label="saturation deficit",
        )
        axes[0, 1].set_title("General-graph Bernoulli quality potential")
        axes[0, 1].set_xlabel("iteration")
        axes[0, 1].legend()

        axes[1, 0].plot(times, theta_series, label="front theta=-mean(y)")
        axes[1, 0].plot(
            times,
            scalar[:, scalar_names.index("fractionality")],
            label="fractionality",
        )
        axes[1, 0].plot(
            times,
            scalar[:, scalar_names.index("dual_anisotropy_inf")],
            label="dual anisotropy inf",
        )
        axes[1, 0].set_title("Activation front and anisotropy")
        axes[1, 0].set_xlabel("iteration")
        axes[1, 0].legend()

        axes[1, 1].semilogy(times, np.maximum(forcing_series, 1e-12), label="forcing / ||z||")
        axes[1, 1].semilogy(
            times,
            np.maximum(prediction_series, 1e-12),
            label="scalar prediction error",
        )
        axes[1, 1].plot(times, clip_series, label="clip fraction")
        axes[1, 1].set_title("Where scalar spectral theory stops being accurate")
        axes[1, 1].set_xlabel("iteration")
        axes[1, 1].legend()

        figure.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.save_npz:
        print(f"saved {npz_path}")
    print(f"saved {json_path}")
    print(f"saved {duhamel_csv_path}")
    print(f"saved {duhamel_summary_csv_path}")
    print(f"saved {scalar_csv_path}")
    if args.plot:
        print(f"saved {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
