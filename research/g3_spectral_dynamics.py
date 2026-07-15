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
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg as dense_linalg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdbo import PDBOSolver, generate_max_cut, parse_gset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gset-id", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dual-init", type=float, default=15.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("/tmp/pdbo-g3-spectrum"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def binned_sum(values: np.ndarray, indices: np.ndarray, bins: int) -> np.ndarray:
    return np.bincount(indices, weights=values, minlength=bins).astype(np.float64)


def manual_step(solver: PDBOSolver) -> dict[str, float]:
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
    new_z = new_primal - 0.5
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

    solver.primal = new_primal.astype(np.float32, copy=False)
    solver.dual = new_dual.astype(np.float32, copy=False)
    solver.perturbation_count += int(np.count_nonzero(trigger))
    solver._update_incumbent()

    def mean_l2(array: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(array, axis=1)))

    return {
        "forcing_l2": mean_l2(forcing),
        "anisotropy_forcing_l2": mean_l2(anisotropy_forcing),
        "projection_forcing_l2": mean_l2(projection_forcing),
        "clip_fraction": float(np.mean((raw_primal < 0.0) | (raw_primal > 1.0))),
        "trigger_fraction": float(np.mean(trigger)),
    }


def main() -> None:
    args = parse_args()
    graph = parse_gset(str(args.gset_id))
    data = generate_max_cut(graph)
    solver = PDBOSolver(
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

    dense_w = solver.W.toarray().astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(dense_w)
    n = solver.n
    edge_weight = float(sum(float(item[2]["weight"]) for item in graph.edges(data=True)))
    spectral_upper = edge_weight / 2.0 - n * float(eigenvalues[0]) / 4.0
    spread = float(eigenvalues[-1] - eigenvalues[0])

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

    solver._update_incumbent()
    sample(0, zero_forcing)
    last_forcing = zero_forcing
    for step in range(args.steps):
        old_dual = np.asarray(solver.dual, dtype=np.float64)
        old_mean_dual = np.mean(old_dual, axis=1)
        multipliers = 1.0 - 2.0 * args.alpha * (
            eigenvalues[np.newaxis, :] + old_mean_dual[:, np.newaxis]
        )
        propagator_sign *= np.sign(multipliers)
        log_abs_propagator += np.log(np.maximum(np.abs(multipliers), 1e-300))
        last_forcing = manual_step(solver)
        best_cut = max(best_cut, -float(solver.objVal))
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
    times = np.asarray(sample_times, dtype=np.int64)

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
    npz_path = args.output.with_suffix(".npz")
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
        "positive_multiplier_condition": bool(
            2.0 * args.alpha * (eigenvalues[-1] + args.dual_init) < 1.0
        ),
        "initial_convex_condition": bool(args.dual_init > -eigenvalues[0]),
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
    print(f"saved {npz_path}")
    print(f"saved {json_path}")
    if args.plot:
        print(f"saved {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
