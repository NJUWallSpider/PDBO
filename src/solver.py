"""CPU implementation of the original PDBO Algorithm 1.

The primal and dual variables are updated simultaneously from the old state,
using fixed stepsizes and the targeted centre perturbation from the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from src.spectral import (
    DEFAULT_ANIMATION_MODES,
    EXACT_SPECTRUM_LIMIT,
    HsSpectrumCertificate,
    SpectrumDistribution,
    compute_animation_modes,
    compute_h_s_spectrum_certificate,
    compute_spectral_window,
    compute_spectrum_distribution,
    largest_eigenvalue,
    smallest_eigenpair,
)
from src.spectral_animation import SpectralAnimation


@dataclass(frozen=True)
class PDBOResult:
    objective: float
    incumbent: np.ndarray
    objective_history: list
    timing_history: list
    runtime: float
    stop_reason: Optional[str]
    conditional_rounding_expected_objective: Optional[float] = None


class PDBO_CPU:
    """NumPy/SciPy implementation of PDBO for QUBO objectives.

    ``tolerance`` is the paper's perturbation tolerance ``delta``.
    """

    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: Optional[np.ndarray] = None,
            batch_size: int = 1,
            primal_lr: float = 0.001,
            primal_lr_mode: str = 'configured',
            spectral_step_fraction: float = 0.5,
            tolerance: float = 1e-8,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            dual_init_mode: str = 'configured',
            dual_burn_in: int = 0,
            dual_psd_margin: float = 0.0,
            max_iters: int = 5000,
            timelimit: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            primal_init: str = 'center_uniform',
            step_callback: Optional[Callable] = None,
            patience: Optional[int] = None,
            min_delta: float = 0.0,
            check_every: int = 100,
            dual_patience_threshold: Optional[float] = None,
            dual_patience_every: int = 100,
            rounding_samples: int = 0,
            conditional_rounding: bool = False,
            g_type: str = 'quadratic',
            restart: bool = False,
            rho: float = 0.1,
            spectral_animation: bool = False,
            spectral_animation_bins: int = 50,
            spectral_animation_modes: int = DEFAULT_ANIMATION_MODES,
            spectral_animation_every: int = 100,
            maxcut_certificate: bool = False,
            rounding_mode: str = 'nearest',
    ):
        if primal_init not in {'center_uniform', 'uniform', 'half', 'binary'}:
            raise ValueError(
                "primal_init must be 'center_uniform', 'uniform', 'half', or 'binary'"
            )
        if not 0.0 < rho <= 0.5:
            raise ValueError("rho must satisfy 0 < rho <= 0.5")
        if g_type not in {'quadratic', 'absolute'}:
            raise ValueError("g_type must be 'quadratic' or 'absolute'")
        if dual_init_mode not in {'configured', 'spectral'}:
            raise ValueError("dual_init_mode must be 'configured' or 'spectral'")
        if primal_lr_mode not in {'configured', 'spectral'}:
            raise ValueError("primal_lr_mode must be 'configured' or 'spectral'")
        if not 0.0 < spectral_step_fraction < 1.0:
            raise ValueError("spectral_step_fraction must lie strictly between 0 and 1")
        if not 0.0 < tolerance < 0.5:
            raise ValueError("tolerance (the paper's delta) must satisfy 0 < delta < 0.5")
        if primal_lr <= 0.0 or dual_lr <= 0.0:
            raise ValueError("primal_lr and dual_lr must be positive")
        if dual_init <= 0.0:
            raise ValueError("dual_init must be positive as required by paper Algorithm 1")
        if dual_burn_in < 0:
            raise ValueError("dual_burn_in must be non-negative")
        if dual_psd_margin < 0.0:
            raise ValueError("dual_psd_margin must be non-negative")
        if batch_size < 1 or max_iters < 0:
            raise ValueError("batch_size must be positive and max_iters non-negative")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")
        if check_every < 1:
            raise ValueError("check_every must be positive")
        if dual_patience_threshold is not None and dual_patience_threshold < 0.0:
            raise ValueError("dual_patience_threshold must be non-negative or None")
        if dual_patience_every < 1:
            raise ValueError("dual_patience_every must be positive")
        if rounding_mode not in {'nearest', 'bernoulli'}:
            raise ValueError("rounding_mode must be 'nearest' or 'bernoulli'")
        if rounding_samples < 0:
            raise ValueError("rounding_samples must be non-negative")
        if spectral_animation_bins < 1:
            raise ValueError("spectral_animation_bins must be positive")
        if spectral_animation_modes < 1:
            raise ValueError("spectral_animation_modes must be positive")
        if spectral_animation_every < 1:
            raise ValueError("spectral_animation_every must be positive")

        self.n = n_vars
        self.batch_size = batch_size
        self.configured_primal_lr = float(primal_lr)
        self.primal_lr_mode = primal_lr_mode
        self.spectral_step_fraction = float(spectral_step_fraction)
        self.primal_lr_lambda_max = None
        self.primal_lr_eigen_residual = None
        self.primal_lr = float(primal_lr)
        self.dual_lr = dual_lr
        self.configured_dual_init = float(dual_init)
        self.dual_init_mode = dual_init_mode
        self.dual_burn_in = int(dual_burn_in)
        self.dual_psd_margin = float(dual_psd_margin)
        self.dual_init_lambda_min = None
        self.dual_init_eigen_residual = None
        self.dual_init = float(dual_init)
        self.delta = max(float(tolerance), float(np.spacing(np.float32(0.5))))
        self.max_iters = max_iters
        self.timelimit = timelimit
        self.verbose = verbose
        self.restart = restart
        self.spectral_animation = spectral_animation
        self.spectral_animation_bins = spectral_animation_bins
        self.spectral_animation_modes = spectral_animation_modes
        self.spectral_animation_every = spectral_animation_every
        self.maxcut_certificate = maxcut_certificate
        self.step_callback = step_callback
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.dual_patience_threshold = dual_patience_threshold
        self.dual_patience_every = dual_patience_every
        self.rounding_mode = rounding_mode
        self.rounding_samples = rounding_samples
        self.conditional_rounding = conditional_rounding
        self.g_type = g_type
        self.stop_reason = None
        self.iterations_completed = 0
        self.perturbation_count = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        indices = np.asarray(Q_indices, dtype=np.int64)
        values = np.asarray(Q_values, dtype=np.float32)
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError("Q_indices must have shape (2, m)")
        if indices.shape[1] != values.shape[0]:
            raise ValueError("Q_indices and Q_values sizes do not match")
        self.c = np.zeros(n_vars, dtype=np.float32) if c is None else np.asarray(c, dtype=np.float32)
        self.Q = sparse.coo_matrix(
            (values, (indices[0], indices[1])), shape=(n_vars, n_vars), dtype=np.float32
        ).tocsr()
        self.Q_transpose = self.Q.transpose().tocsr()
        self.W = ((self.Q + self.Q_transpose) * 0.5).tocsr()
        self.Q_diagonal = np.asarray(self.Q.diagonal(), dtype=np.float64)
        self._multilinear_interaction = None
        self._multilinear_linear = None
        self.spectral_window = None
        self.spectral_window_error = None
        self.spectrum_distribution: Optional[SpectrumDistribution] = None
        self.spectrum_distribution_error = None
        self.animation: Optional[SpectralAnimation] = None
        self._incumbent_certificate_bits = None
        self._incumbent_certificate: Optional[HsSpectrumCertificate] = None
        self.dual_history: list[tuple[int, np.ndarray]] = []
        self._incumbent_records: dict[tuple[int, ...], tuple[np.ndarray, float, int]] = {}

        self.dual_init = self._resolve_dual_init()
        self.primal_lr = self._resolve_primal_lr()
        self.primal, self.dual = self._init_variables(self.dual_init, primal_init, rho)
        rounded_initial = self._round_primal(self.primal)
        initial_candidates = np.concatenate(
            (rounded_initial, np.zeros((1, self.n), dtype=np.float32)),
            axis=0,
        )
        initial_values = self._score_candidates(initial_candidates)
        initial_index = int(np.argmin(initial_values))
        self.incumbent = initial_candidates[initial_index].copy()
        self.objective = float(initial_values[initial_index])
        self.objective_history = [self.objective]
        self.timing_history = [0.0]
        self.best_relaxed = None
        self.best_relaxed_expected_objective = None
        self.conditional_rounding_expected_objective = None
        if self.conditional_rounding:
            self._consider_relaxed_candidates(self.primal)
        self.start_time = None
        self.runtime = None

    def _init_variables(
            self,
            dual_init: float,
            primal_init: str,
            rho: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape = (self.batch_size, self.n)
        if primal_init == 'half':
            primal = np.full(shape, 0.5, dtype=np.float32)
        elif primal_init == 'binary':
            primal = self.rng.integers(0, 2, size=shape).astype(np.float32)
        elif primal_init == 'uniform':
            primal = self.rng.random(shape, dtype=np.float32)
        else:
            low = np.float32(0.5 - rho)
            high = np.float32(0.5 + rho)
            primal = low + (high - low) * self.rng.random(shape, dtype=np.float32)
        dual = np.full(shape, dual_init, dtype=np.float32)
        return primal, dual

    def _log_verbose(
            self,
            elapsed: float,
            phase: int,
            phases: int,
            phase_iteration: int,
    ) -> None:
        dual_mean = float(np.mean(self.dual, dtype=np.float64))
        dual_min = float(np.min(self.dual))
        dual_max = float(np.max(self.dual))
        print(
            f"round={phase}/{phases} "
            f"iter={phase_iteration}/{self.max_iters} "
            f"best={self.objective} "
            f"dual_mean={dual_mean:.8g} dual_min={dual_min:.8g} dual_max={dual_max:.8g} "
            f"elapsed={elapsed:.3f}s"
        )

    def _resolve_dual_init(self) -> float:
        """Return the configured value or a spectrum-scaled convex burn-in value."""
        if self.dual_init_mode == 'configured':
            return self.configured_dual_init

        lambda_min, _, residual = smallest_eigenpair(self.W, self.n)
        self.dual_init_lambda_min = lambda_min
        self.dual_init_eigen_residual = residual
        target = (
            -lambda_min
            + self.dual_psd_margin
            + self.dual_lr * self.dual_burn_in / 4.0
        )
        target32 = np.float32(max(target, np.finfo(np.float32).tiny))
        return float(np.nextafter(target32, np.float32(np.inf)))

    def _resolve_primal_lr(self) -> float:
        """Return the configured step or make the initial spectral map non-expansive."""
        if self.primal_lr_mode == 'configured':
            return self.configured_primal_lr

        lambda_max, residual = largest_eigenvalue(self.W, self.n)
        self.primal_lr_lambda_max = lambda_max
        self.primal_lr_eigen_residual = residual
        curvature = lambda_max + self.dual_init
        if curvature <= 0.0:
            raise ValueError(
                "spectral primal step requires lambda_max(W) + dual_init > 0"
            )
        return self.spectral_step_fraction / curvature

    def _log_spectrum_distribution(self) -> None:
        if self.spectrum_distribution is None:
            suffix = (
                f" error={self.spectrum_distribution_error}"
                if self.spectrum_distribution_error is not None
                else ""
            )
            print(f"spectrum_distribution=unavailable{suffix}")
            return

        mode = self.spectrum_distribution.mode
        edges = self.spectrum_distribution.edges
        counts = self.spectrum_distribution.counts
        total = float(np.sum(counts))
        peak = max(float(np.max(counts)), 1.0)
        print(
            f"spectrum_distribution={mode} n={self.n} bins={len(counts)} "
            f"range=[{edges[0]:.8g}, {edges[-1]:.8g}]"
        )
        for index, count in enumerate(counts):
            percentage = 0.0 if total == 0.0 else 100.0 * float(count) / total
            bar = '#' * int(round(30.0 * float(count) / peak))
            count_label = f"{int(count)}" if mode == 'exact' else f"~{count:.1f}"
            closing = ']' if index == len(counts) - 1 else ')'
            print(
                f"  [{edges[index]:.6g}, {edges[index + 1]:.6g}{closing} "
                f"count={count_label:>8} pct={percentage:6.2f}% {bar}"
            )

    def _log_spectral_window(self, phase: int, phases: int) -> None:
        if self.spectral_window is None:
            suffix = (
                f" error={self.spectral_window_error}"
                if self.spectral_window_error is not None
                else ""
            )
            print(f"round={phase}/{phases} spectral_window=unavailable{suffix}")
            return

        lambda_1, lambda_next, multiplicity = self.spectral_window
        print(
            f"round={phase}/{phases} "
            f"spectral_window=({-lambda_next:.8g}, {-lambda_1:.8g}) "
            f"lambda_1={lambda_1:.8g} "
            f"lambda_(r+1)={lambda_next:.8g} r={multiplicity}"
        )

    def _quadratic_values(self, x: np.ndarray) -> np.ndarray:
        qx = (self.Q @ x.T).T
        return np.sum(x * qx, axis=1) + x @ self.c

    def _objective_values_and_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        qx = (self.Q @ x.T).T
        qtx = (self.Q_transpose @ x.T).T
        values = np.sum(x * qx, axis=1) + x @ self.c
        gradients = qx + qtx + self.c[np.newaxis, :]
        return np.asarray(values), np.asarray(gradients, dtype=np.float32)

    def _score_candidates(self, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(self._quadratic_values(candidates), dtype=np.float64)

    def _multilinear_values(
            self,
            x: np.ndarray,
            quadratic_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Expected binary QUBO value under independent Bernoulli probabilities x."""
        quadratic = (
            self._quadratic_values(x)
            if quadratic_values is None
            else quadratic_values
        )
        diagonal_correction = (x - x * x) @ self.Q_diagonal
        return np.asarray(quadratic + diagonal_correction, dtype=np.float64)

    def _consider_relaxed_candidates(
            self,
            candidates: np.ndarray,
            quadratic_values: Optional[np.ndarray] = None,
    ) -> None:
        if not self.conditional_rounding:
            return
        values = self._multilinear_values(candidates, quadratic_values)
        index = int(np.argmin(values))
        value = float(values[index])
        if (
            self.best_relaxed_expected_objective is None
            or value < self.best_relaxed_expected_objective
        ):
            self.best_relaxed_expected_objective = value
            self.best_relaxed = np.asarray(candidates[index], dtype=np.float64).copy()

    def _conditional_expectation_candidate(self, probabilities: np.ndarray) -> np.ndarray:
        """Derandomize independent Bernoulli rounding for the multilinear QUBO."""
        if self._multilinear_interaction is None:
            interaction = (self.Q + self.Q_transpose).astype(np.float64)
            interaction.setdiag(0.0)
            interaction.eliminate_zeros()
            self._multilinear_interaction = interaction.tocsc()
            self._multilinear_linear = (
                np.asarray(self.c, dtype=np.float64) + self.Q_diagonal
            )
        candidate = np.asarray(probabilities, dtype=np.float64).copy()
        coefficients = (
            self._multilinear_linear
            + self._multilinear_interaction @ candidate
        )
        interaction = self._multilinear_interaction
        for index in range(self.n):
            old_value = candidate[index]
            new_value = 1.0 if coefficients[index] < 0.0 else 0.0
            candidate[index] = new_value
            difference = new_value - old_value
            start = interaction.indptr[index]
            end = interaction.indptr[index + 1]
            neighbors = interaction.indices[start:end]
            coefficients[neighbors] += interaction.data[start:end] * difference
        return candidate.astype(np.float32)

    def _constraint_values_and_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.g_type == 'absolute':
            values = np.abs(x - 0.5) - 0.5
            gradients = np.sign(x - 0.5)
            return values, gradients
        return x ** 2 - x, 2.0 * x - 1.0

    def _paper_step(self) -> None:
        """Perform one simultaneous primal-dual update."""
        old_primal = self.primal
        old_dual = self.dual
        objective_values, objective_gradient = self._objective_values_and_gradients(old_primal)
        self._consider_relaxed_candidates(old_primal, objective_values)
        constraint_value, constraint_gradient = self._constraint_values_and_gradients(old_primal)
        lagrangian_gradient = objective_gradient + old_dual * constraint_gradient

        projected = np.clip(old_primal - self.primal_lr * lagrangian_gradient, 0.0, 1.0)
        if self.g_type == 'absolute':
            at_perturbation_point = old_primal == 0.5
        else:
            at_perturbation_point = np.abs(old_primal - 0.5) <= self.delta
        trigger = (
            at_perturbation_point
            & (np.abs(lagrangian_gradient) <= 2.0 * self.delta)
            & (old_dual <= 0.0)
        )
        perturbed = np.where(old_primal <= 0.5, 0.5 - self.delta, 0.5 + self.delta)

        # Both assignments deliberately read only the old state (x^t, y^t).
        self.primal = np.where(trigger, perturbed, projected).astype(np.float32, copy=False)
        self.dual = (
            old_dual + self.dual_lr * constraint_value
        ).astype(np.float32, copy=False)
        self.perturbation_count += int(np.count_nonzero(trigger))

    def _round_primal(self, primal: np.ndarray) -> np.ndarray:
        """Round relaxed states according to the configured archive policy."""
        if self.rounding_mode == 'bernoulli':
            return (self.rng.random(primal.shape) < primal).astype(np.float32)
        return np.rint(primal).astype(np.float32)

    def _update_incumbent(self) -> bool:
        rounded = self._round_primal(self.primal)
        candidates = np.concatenate((rounded, self.incumbent[np.newaxis, :]), axis=0)
        values = self._score_candidates(candidates)
        index = int(np.argmin(values))
        candidate_value = float(values[index])
        improved = candidate_value < self.objective - self.min_delta
        self.objective = candidate_value
        self.incumbent = candidates[index].copy()
        return improved

    def _stability_tracking_enabled(self) -> bool:
        return self.batch_size == 1 and self.maxcut_certificate

    def _record_dual_state(self, iteration: int) -> None:
        if self._stability_tracking_enabled():
            self.dual_history.append(
                (int(iteration), np.asarray(self.dual[0], dtype=np.float32).copy())
            )

    def _record_incumbent(self, iteration: int) -> None:
        if not self._stability_tracking_enabled():
            return
        incumbent = np.rint(self.incumbent).astype(np.int8, copy=False)
        key = tuple(int(value) for value in incumbent)
        if key not in self._incumbent_records:
            self._incumbent_records[key] = (
                incumbent.copy(),
                float(self.objective),
                int(iteration),
            )

    def _report_incumbent_stability(self) -> None:
        if not self._stability_tracking_enabled() or not self.verbose:
            return

        records = []
        for incumbent, objective, first_seen in self._incumbent_records.values():
            sign = 2.0 * incumbent.astype(np.float64) - 1.0
            gains = sign * np.asarray(self.W @ sign, dtype=np.float64).reshape(-1)
            first_stable = None
            max_violation = None
            for iteration, dual in self.dual_history:
                residual = gains + dual.astype(np.float64)
                violation = float(np.max(residual))
                if violation <= 1e-8:
                    first_stable = iteration
                    max_violation = violation
                    break
            records.append(
                {
                    "cut_value": -float(objective),
                    "first_seen": first_seen,
                    "first_stable": first_stable,
                    "max_violation": max_violation,
                }
            )

        records.sort(key=lambda record: (-record["cut_value"], record["first_seen"]))
        print("incumbent first-stability times (sorted by cut value):")
        for rank, record in enumerate(records, start=1):
            stable = (
                str(record["first_stable"])
                if record["first_stable"] is not None
                else "never"
            )
            print(
                f"  {rank}: cut={record['cut_value']:.10g} "
                f"first_seen={record['first_seen']} first_stable_t={stable}"
            )

    def _restart_from_incumbent(self) -> None:
        """Restart both variables from the first phase's best rounded solution."""
        self.primal[...] = self.incumbent[np.newaxis, :]
        self.dual.fill(self.dual_init)

    def _prepare_run(self) -> None:
        exact_animation = self.spectral_animation and self.n <= EXACT_SPECTRUM_LIMIT
        if (
            (self.verbose or exact_animation)
            and self.spectrum_distribution is None
        ):
            try:
                self.spectrum_distribution = compute_spectrum_distribution(
                    self.W,
                    self.n,
                    self.seed,
                    include_eigenvectors=exact_animation,
                )
            except (sparse_linalg.ArpackError, np.linalg.LinAlgError, ValueError) as error:
                self.spectrum_distribution_error = type(error).__name__
                if exact_animation:
                    raise RuntimeError(
                        "failed to compute eigenvectors for spectral animation"
                    ) from error
        if self.verbose and self.spectral_window is None:
            try:
                exact_eigenvalues = (
                    None
                    if self.spectrum_distribution is None
                    else self.spectrum_distribution.eigenvalues
                )
                self.spectral_window = compute_spectral_window(
                    self.W,
                    self.n,
                    self.seed,
                    exact_eigenvalues,
                )
            except (sparse_linalg.ArpackError, np.linalg.LinAlgError, ValueError) as error:
                self.spectral_window_error = type(error).__name__
        if self.verbose:
            print(
                f"effective_parameters primal_lr={self.primal_lr:.8g} "
                f"dual_init={self.dual_init:.8g} "
                f"delta={self.delta:.8g}"
            )
            self._log_spectrum_distribution()

        if self.spectral_animation and self.animation is None:
            if exact_animation:
                distribution = self.spectrum_distribution
                if (
                    distribution is None
                    or distribution.eigenvalues is None
                    or distribution.eigenvectors is None
                ):
                    raise RuntimeError(
                        "spectral animation requires an exact eigendecomposition"
                    )
                eigenvalues = distribution.eigenvalues
                eigenvectors = distribution.eigenvectors
                mode_label = "exact spectrum"
            else:
                try:
                    eigenvalues, eigenvectors = compute_animation_modes(
                        self.W,
                        self.n,
                        self.seed,
                        self.spectral_animation_modes,
                    )
                except (np.linalg.LinAlgError, ValueError) as error:
                    raise RuntimeError(
                        "failed to compute approximate modes for spectral animation"
                    ) from error
                mode_label = f"Lanczos-Ritz approximation ({len(eigenvalues)} modes)"
            self.animation = SpectralAnimation(
                eigenvalues,
                eigenvectors,
                self.spectral_animation_bins,
                mode_label=mode_label,
                show_a_t=self.maxcut_certificate,
            )
            self.animation.setup(self.primal, self.dual)

        self.start_time = time.perf_counter()

    def _log_progress(self, phase: int, phases: int, iteration: int) -> None:
        if self.verbose:
            self._log_verbose(
                time.perf_counter() - self.start_time,
                phase,
                phases,
                iteration,
            )

    def _animation_objective_means(self) -> Tuple[float, float]:
        objective_values = self._quadratic_values(self.primal)
        constraint_values, _ = self._constraint_values_and_gradients(self.primal)
        lagrangian_values = objective_values + np.sum(
            self.dual * constraint_values,
            axis=1,
        )
        return (
            float(np.mean(lagrangian_values, dtype=np.float64)),
            float(np.mean(objective_values, dtype=np.float64)),
        )

    def _current_incumbent_certificate(self) -> HsSpectrumCertificate:
        """Return the cached spectral certificate for the current best cut."""
        bits = np.asarray(self.incumbent, dtype=np.int8)
        if (
            self._incumbent_certificate is not None
            and self._incumbent_certificate_bits is not None
            and np.array_equal(bits, self._incumbent_certificate_bits)
        ):
            return self._incumbent_certificate

        sign = 2.0 * bits.astype(np.float64) - 1.0
        # G(x) = s * (W s), so H_s = W - Diag(G(x)).
        certificate = compute_h_s_spectrum_certificate(
            self.W,
            sign,
            self.seed,
            bins=self.spectral_animation_bins,
        )
        self._incumbent_certificate_bits = bits.copy()
        self._incumbent_certificate = certificate
        return certificate

    def _a_t_animation_payload(self) -> dict:
        """Return the current batch-mean ``A_t`` spectrum for the animation."""
        if not self.maxcut_certificate:
            return {}

        # A batch contains one coordinate-dual vector per trajectory.  The
        # single spectrum panel represents their coordinate-wise mean.
        mean_dual = np.mean(np.asarray(self.dual, dtype=np.float64), axis=0)
        a_t = (
            self.W.astype(np.float64)
            + sparse.diags(mean_dual, format="csr", dtype=np.float64)
        ).tocsr()
        distribution = compute_spectrum_distribution(
            a_t,
            self.n,
            self.seed,
            bins=self.spectral_animation_bins,
        )
        if distribution.eigenvalues is not None:
            lambda_min = float(distribution.eigenvalues[0])
        else:
            lambda_min = float(distribution.edges[0])
        kappa = max(-lambda_min, 0.0)
        incumbent_certificate = self._current_incumbent_certificate()
        incumbent_cert_kappa = max(-float(incumbent_certificate.lambda_min), 0.0)
        return {
            "a_t_edges": distribution.edges,
            "a_t_counts": distribution.counts,
            "a_t_lambda_min": lambda_min,
            "a_t_kappa": kappa,
            "a_t_gap_bound": 0.25 * self.n * kappa,
            "incumbent_cert_kappa": incumbent_cert_kappa,
            "incumbent_cert_gap_bound": incumbent_certificate.gap_upper_bound,
            "a_t_mode": distribution.mode,
        }

    def _update_animation(
        self,
        phase: int,
        phases: int,
        iteration: int,
        total_iteration: Optional[int] = None,
    ) -> None:
        if self.animation is None:
            return
        if (
            not self.animation.run_continuous
            or iteration % self.spectral_animation_every == 0
        ):
            self._render_animation(
                phase,
                phases,
                iteration,
                total_iteration=total_iteration,
            )

    def _render_animation(
        self,
        phase: int,
        phases: int,
        iteration: int,
        total_iteration: Optional[int] = None,
    ) -> None:
        """Refresh the built-in animation, preserving the callback API for test doubles."""
        if isinstance(self.animation, SpectralAnimation):
            lagrangian_value, objective_value = self._animation_objective_means()
            a_t_payload = self._a_t_animation_payload()
            self.animation.update(
                self.primal,
                phase,
                phases,
                iteration,
                dual=self.dual,
                total_iteration=iteration if total_iteration is None else total_iteration,
                lagrangian_value=lagrangian_value,
                objective_value=objective_value,
                **a_t_payload,
            )
        else:
            self.animation.update(self.primal, phase, phases, iteration)

    def _run_phase(
        self,
        phase: int,
        phases: int,
        total_step: int,
    ) -> Tuple[int, int, Optional[str]]:
        if self.verbose:
            self._log_spectral_window(phase, phases)
            self._log_progress(phase, phases, 0)
        if self.animation is not None:
            self._render_animation(phase, phases, 0, total_iteration=total_step)

        last_improvement_step = 0
        stop_reason = None
        completed_iterations = 0
        previous_dual_mean = float(np.mean(self.dual, dtype=np.float64))
        for phase_step in range(self.max_iters):
            if self.animation is not None:
                self.start_time += self.animation.wait_for_step()

            callback_step = total_step
            total_step += 1
            step_start = time.perf_counter()
            self._paper_step()
            improved = self._update_incumbent()
            self._record_dual_state(total_step)
            self._record_incumbent(total_step)
            if improved:
                self.objective_history.append(self.objective)
                self.timing_history.append(time.perf_counter() - self.start_time)
                last_improvement_step = phase_step

            step_time = time.perf_counter() - step_start
            completed_iterations = phase_step + 1
            if completed_iterations % self.check_every == 0:
                self._log_progress(phase, phases, completed_iterations)
            self._update_animation(
                phase,
                phases,
                completed_iterations,
                total_iteration=total_step,
            )
            if self.step_callback is not None:
                self.step_callback(
                    callback_step,
                    step_time,
                    self.objective,
                    self.incumbent,
                )

            if (
                self.dual_patience_threshold is not None
                and completed_iterations % self.dual_patience_every == 0
            ):
                current_dual_mean = float(np.mean(self.dual, dtype=np.float64))
                dual_decrease = previous_dual_mean - current_dual_mean
                if dual_decrease < self.dual_patience_threshold:
                    stop_reason = "dual_patience"
                    break
                previous_dual_mean = current_dual_mean

            if (
                self.timelimit is not None
                and time.perf_counter() - self.start_time >= self.timelimit
            ):
                stop_reason = "timelimit"
                break
            if (
                self.patience is not None
                and phase_step - last_improvement_step >= self.patience
            ):
                stop_reason = "patience"
                break

        if self.verbose and completed_iterations % self.check_every != 0:
            self._log_progress(phase, phases, completed_iterations)
        return total_step, completed_iterations, stop_reason

    def _record_final_candidate(self, candidates: np.ndarray, min_delta: float) -> None:
        values = self._score_candidates(candidates)
        index = int(np.argmin(values))
        candidate_value = float(values[index])
        if candidate_value < self.objective - min_delta:
            self.objective = candidate_value
            self.incumbent = candidates[index].copy()
            self.objective_history.append(self.objective)
            self.timing_history.append(time.perf_counter() - self.start_time)

    def _finalize_candidates(self) -> None:
        if self.conditional_rounding:
            self._consider_relaxed_candidates(self.primal)
            conditional_candidate = self._conditional_expectation_candidate(
                self.best_relaxed
            )
            self.conditional_rounding_expected_objective = (
                self.best_relaxed_expected_objective
            )
            candidates = np.stack((conditional_candidate, self.incumbent))
            self._record_final_candidate(candidates, min_delta=0.0)

        if self.rounding_samples > 0:
            sampled = (
                self.rng.random((self.rounding_samples, self.batch_size, self.n)) < self.primal
            ).astype(np.float32).reshape((-1, self.n))
            candidates = np.concatenate((sampled, self.incumbent[np.newaxis, :]), axis=0)
            self._record_final_candidate(candidates, min_delta=self.min_delta)

    def _build_result(self) -> PDBOResult:
        return PDBOResult(
            objective=float(self.objective),
            incumbent=np.asarray(self.incumbent, dtype=np.int32),
            objective_history=list(self.objective_history),
            timing_history=list(self.timing_history),
            runtime=self.runtime,
            stop_reason=self.stop_reason,
            conditional_rounding_expected_objective=(
                self.conditional_rounding_expected_objective
            ),
        )

    def optimize(self) -> PDBOResult:
        self.stop_reason = None
        phases = 2 if self.restart else 1
        self.dual_history.clear()
        self._incumbent_records.clear()
        self._prepare_run()
        total_step = 0
        self._record_dual_state(total_step)
        self._record_incumbent(total_step)
        last_phase = 1
        last_phase_iteration = 0

        for phase_index in range(phases):
            last_phase = phase_index + 1
            total_step, last_phase_iteration, phase_stop_reason = self._run_phase(
                last_phase,
                phases,
                total_step,
            )
            if phase_stop_reason == "timelimit":
                self.stop_reason = phase_stop_reason
                break
            if phase_index == 0 and self.restart:
                self._restart_from_incumbent()
                continue
            if phase_stop_reason is not None:
                self.stop_reason = phase_stop_reason
                break

        if self.stop_reason is None:
            self.stop_reason = "max_iters"

        self.iterations_completed = total_step

        self._finalize_candidates()
        self._record_incumbent(total_step)
        self._report_incumbent_stability()

        self.runtime = time.perf_counter() - self.start_time
        if self.animation is not None:
            if isinstance(self.animation, SpectralAnimation):
                lagrangian_value, objective_value = self._animation_objective_means()
                a_t_payload = self._a_t_animation_payload()
                self.animation.finish(
                    self.primal,
                    last_phase,
                    phases,
                    last_phase_iteration,
                    dual=self.dual,
                    total_iteration=total_step,
                    lagrangian_value=lagrangian_value,
                    objective_value=objective_value,
                    **a_t_payload,
                )
            else:
                self.animation.finish(
                    self.primal,
                    last_phase,
                    phases,
                    last_phase_iteration,
                )
        return self._build_result()
