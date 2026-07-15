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


@dataclass(frozen=True)
class PDBOResult:
    objective: float
    incumbent: np.ndarray
    objective_history: list
    timing_history: list
    runtime: float
    stop_reason: Optional[str]


class PDBO_CPU:
    """NumPy/SciPy implementation of PDBO for QUBO objectives.

    ``tolerance`` is the paper's perturbation tolerance ``delta``.
    """

    _SPECTRUM_BINS = 20
    _EXACT_SPECTRUM_LIMIT = 2000
    _SLQ_PROBES = 4
    _SLQ_STEPS = 40

    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: Optional[np.ndarray] = None,
            batch_size: int = 1,
            primal_lr: float = 0.001,
            tolerance: float = 1e-8,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            timelimit: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            primal_init: str = 'center_uniform',
            step_callback: Optional[Callable] = None,
            patience: Optional[int] = None,
            min_delta: float = 0.0,
            check_every: int = 100,
            quadratic_backend: str = 'sparse',
            rounding_samples: int = 0,
            g_type: str = 'quadratic',
            restart: bool = False,
            rho: float = 0.1,
            spectral_animation: bool = False,
            spectral_animation_bins: int = 50,
            spectral_animation_every: int = 100,
            spectral_animation_hold: bool = True,
            spectral_animation_manual: bool = False,
    ):
        if primal_init not in {'center_uniform', 'uniform', 'half', 'binary'}:
            raise ValueError(
                "primal_init must be 'center_uniform', 'uniform', 'half', or 'binary'"
            )
        if not 0.0 < rho <= 0.5:
            raise ValueError("rho must satisfy 0 < rho <= 0.5")
        if quadratic_backend not in {'edge', 'sparse'}:
            raise ValueError("quadratic_backend must be 'edge' or 'sparse'")
        if g_type not in {'quadratic', 'absolute'}:
            raise ValueError("g_type must be 'quadratic' or 'absolute'")
        if tolerance <= 0.0:
            raise ValueError("tolerance (the paper's delta) must be positive")
        if primal_lr <= 0.0 or dual_lr <= 0.0:
            raise ValueError("primal_lr and dual_lr must be positive")
        if dual_init <= 0.0:
            raise ValueError("dual_init must be positive as required by paper Algorithm 1")
        if batch_size < 1 or max_iters < 0:
            raise ValueError("batch_size must be positive and max_iters non-negative")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")
        if check_every < 1:
            raise ValueError("check_every must be positive")
        if rounding_samples < 0:
            raise ValueError("rounding_samples must be non-negative")
        if spectral_animation_bins < 1:
            raise ValueError("spectral_animation_bins must be positive")
        if spectral_animation_every < 1:
            raise ValueError("spectral_animation_every must be positive")
        if spectral_animation and n_vars > self._EXACT_SPECTRUM_LIMIT:
            raise ValueError(
                "spectral_animation requires all eigenvectors and is limited to "
                f"n <= {self._EXACT_SPECTRUM_LIMIT}; got n={n_vars}"
            )

        self.n = n_vars
        self.batch_size = batch_size
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.dual_init = dual_init
        self.tolerance = tolerance
        self.delta = tolerance
        self.max_iters = max_iters
        self.timelimit = timelimit
        self.verbose = verbose
        self.restart = restart
        self.rho = rho
        self.spectral_animation = spectral_animation
        self.spectral_animation_bins = spectral_animation_bins
        self.spectral_animation_every = spectral_animation_every
        self.spectral_animation_hold = spectral_animation_hold
        self.spectral_animation_manual = spectral_animation_manual
        self.step_callback = step_callback
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.quadratic_backend = quadratic_backend
        self.rounding_samples = rounding_samples
        self.g_type = g_type
        self.stop_reason = None
        self.perturbation_count = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        indices = np.asarray(Q_indices, dtype=np.int64)
        values = np.asarray(Q_values, dtype=np.float32)
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError("Q_indices must have shape (2, m)")
        if indices.shape[1] != values.shape[0]:
            raise ValueError("Q_indices and Q_values sizes do not match")
        self.Q_indices = indices
        self.Q_values = values
        self.m = indices.shape[1]
        self.c = np.zeros(n_vars, dtype=np.float32) if c is None else np.asarray(c, dtype=np.float32)
        self.Q = sparse.coo_matrix(
            (values, (indices[0], indices[1])), shape=(n_vars, n_vars), dtype=np.float32
        ).tocsr()
        self.Q_transpose = self.Q.transpose().tocsr()
        self.W = ((self.Q + self.Q_transpose) * 0.5).tocsr()
        self.spectral_window = None
        self.spectral_window_error = None
        self.spectrum_distribution = None
        self.spectrum_distribution_error = None
        self._exact_eigenvalues = None
        self._exact_eigenvectors = None
        self._animation_figure = None
        self._animation_axes = None
        self._animation_bars = None
        self._animation_bin_edges = None
        self._animation_bin_indices = None
        self._animation_plt = None
        self._animation_next_button = None
        self._animation_run_button = None
        self._animation_step_requested = False
        self._animation_run_continuous = not spectral_animation_manual

        self.primal, self.dual = self._init_variables(dual_init, primal_init, rho)
        self.incumbent = np.zeros(self.n, dtype=np.float32)
        self.objVal = float(self._score_candidates(self.incumbent[np.newaxis, :])[0])
        self.objVal_record = [self.objVal]
        self.timing_record = [0.0]
        self.start_time = None
        self.solving_time = None
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

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
        print(
            f"round={phase}/{phases} "
            f"iter={phase_iteration}/{self.max_iters} "
            f"best={self.objVal} elapsed={elapsed:.3f}s"
        )

    def _compute_spectral_window(self) -> Optional[Tuple[float, float, int]]:
        """Return (lambda_1, lambda_(r+1), r) for W = sym(Q)."""
        if self.n < 2:
            return None

        if self._exact_eigenvalues is not None:
            eigenvalues = self._exact_eigenvalues
        elif self.n <= 256:
            eigenvalues = np.linalg.eigvalsh(self.W.toarray().astype(np.float64))
        else:
            matrix = self.W.astype(np.float64)
            max_k = min(64, self.n - 1)
            k = min(8, max_k)
            eigenvalues = np.empty(0, dtype=np.float64)
            rng = np.random.default_rng(self.seed)
            v0 = rng.standard_normal(self.n)

            while True:
                eigenvalues = np.sort(
                    sparse_linalg.eigsh(
                        matrix,
                        k=k,
                        which='SA',
                        return_eigenvectors=False,
                        v0=v0,
                    )
                )
                gap_tolerance = 1e-6 * max(1.0, abs(float(eigenvalues[0])))
                if np.any(eigenvalues > eigenvalues[0] + gap_tolerance) or k >= max_k:
                    break
                k = min(2 * k, max_k)

        lambda_1 = float(eigenvalues[0])
        gap_tolerance = 1e-6 * max(1.0, abs(lambda_1))
        distinct = np.flatnonzero(eigenvalues > lambda_1 + gap_tolerance)
        if distinct.size == 0:
            return None

        r = int(distinct[0])
        return lambda_1, float(eigenvalues[r]), r

    @staticmethod
    def _histogram_edges(lambda_min: float, lambda_max: float, bins: int) -> np.ndarray:
        if np.isclose(lambda_min, lambda_max):
            padding = 0.5 * max(1.0, abs(lambda_min))
            lambda_min -= padding
            lambda_max += padding
        return np.linspace(lambda_min, lambda_max, bins + 1, dtype=np.float64)

    def _lanczos_spectral_measure(
            self,
            matrix: sparse.csr_matrix,
            probe: np.ndarray,
            steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate one probe's spectral measure with reorthogonalized Lanczos."""
        basis = np.empty((self.n, steps), dtype=np.float64)
        alphas = np.empty(steps, dtype=np.float64)
        betas = np.empty(max(steps - 1, 0), dtype=np.float64)
        q = probe / np.linalg.norm(probe)
        beta_previous = 0.0
        q_previous = np.zeros_like(q)
        actual_steps = steps

        for step in range(steps):
            basis[:, step] = q
            z = matrix @ q
            if step > 0:
                z -= beta_previous * q_previous
            alpha = float(q @ z)
            alphas[step] = alpha
            z -= alpha * q

            coefficients = basis[:, :step + 1].T @ z
            z -= basis[:, :step + 1] @ coefficients

            if step == steps - 1:
                break
            beta = float(np.linalg.norm(z))
            if beta <= np.finfo(np.float64).eps:
                actual_steps = step + 1
                break
            betas[step] = beta
            q_previous = q
            q = z / beta
            beta_previous = beta

        tridiagonal = np.diag(alphas[:actual_steps])
        if actual_steps > 1:
            off_diagonal = betas[:actual_steps - 1]
            tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        nodes, eigenvectors = np.linalg.eigh(tridiagonal)
        return nodes, eigenvectors[0, :] ** 2

    def _compute_spectrum_distribution(self) -> Tuple[str, np.ndarray, np.ndarray]:
        """Return a full-spectrum histogram, exact when dense decomposition is feasible."""
        matrix = self.W.astype(np.float64)
        bins = self._SPECTRUM_BINS

        if self.n <= self._EXACT_SPECTRUM_LIMIT:
            dense_matrix = matrix.toarray()
            if self.spectral_animation:
                eigenvalues, self._exact_eigenvectors = np.linalg.eigh(dense_matrix)
            else:
                eigenvalues = np.linalg.eigvalsh(dense_matrix)
            self._exact_eigenvalues = eigenvalues
            edges = self._histogram_edges(float(eigenvalues[0]), float(eigenvalues[-1]), bins)
            counts, _ = np.histogram(eigenvalues, bins=edges)
            return 'exact', edges, counts.astype(np.float64)

        lambda_min = float(
            sparse_linalg.eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        )
        lambda_max = float(
            sparse_linalg.eigsh(matrix, k=1, which='LA', return_eigenvectors=False)[0]
        )
        edges = self._histogram_edges(lambda_min, lambda_max, bins)
        normalized_counts = np.zeros(bins, dtype=np.float64)
        rng = np.random.default_rng(self.seed + 1)
        steps = min(self._SLQ_STEPS, self.n)

        for _ in range(self._SLQ_PROBES):
            probe = rng.choice((-1.0, 1.0), size=self.n)
            nodes, weights = self._lanczos_spectral_measure(matrix, probe, steps)
            probe_counts, _ = np.histogram(nodes, bins=edges, weights=weights)
            normalized_counts += probe_counts

        estimated_counts = normalized_counts * (self.n / self._SLQ_PROBES)
        return 'approx-slq', edges, estimated_counts

    def _log_spectrum_distribution(self) -> None:
        if self.spectrum_distribution is None:
            suffix = (
                f" error={self.spectrum_distribution_error}"
                if self.spectrum_distribution_error is not None
                else ""
            )
            print(f"spectrum_distribution=unavailable{suffix}")
            return

        mode, edges, counts = self.spectrum_distribution
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

    def _spectral_mode_bin_means(self) -> np.ndarray:
        """Return binned RMS lengths of x - 1/2 in the eigenvector basis."""
        if self._exact_eigenvectors is None or self._animation_bin_indices is None:
            raise RuntimeError("spectral animation has not been initialized")

        centered = np.asarray(self.primal, dtype=np.float64) - 0.5
        coefficients = centered @ self._exact_eigenvectors
        mode_lengths = np.sqrt(np.mean(coefficients ** 2, axis=0))
        sums = np.bincount(
            self._animation_bin_indices,
            weights=mode_lengths,
            minlength=self.spectral_animation_bins,
        )
        counts = np.bincount(
            self._animation_bin_indices,
            minlength=self.spectral_animation_bins,
        )
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    def _request_spectral_animation_step(self, _event=None) -> None:
        if not self._animation_run_continuous:
            self._animation_step_requested = True

    def _toggle_spectral_animation_run(self, _event=None) -> None:
        self._animation_run_continuous = not self._animation_run_continuous
        if self._animation_run_button is not None:
            label = 'Pause' if self._animation_run_continuous else 'Run'
            self._animation_run_button.label.set_text(label)
        if self._animation_figure is not None:
            self._animation_figure.canvas.draw_idle()

    def _wait_for_spectral_animation_step(self) -> float:
        wait_start = time.perf_counter()
        while (
            self._animation_figure is not None
            and not self._animation_run_continuous
            and not self._animation_step_requested
        ):
            figure = self._animation_figure
            if not self._animation_plt.fignum_exists(figure.number):
                self._animation_figure = None
                break
            self._animation_plt.pause(0.05)

        self._animation_step_requested = False
        return time.perf_counter() - wait_start

    def _setup_spectral_animation(self) -> None:
        if self._exact_eigenvalues is None or self._exact_eigenvectors is None:
            raise RuntimeError("spectral animation requires an exact eigendecomposition")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
        except ImportError as error:
            raise RuntimeError(
                'spectral animation requires matplotlib; install with pip install -e ".[visualization]"'
            ) from error

        edges = self._histogram_edges(
            float(self._exact_eigenvalues[0]),
            float(self._exact_eigenvalues[-1]),
            self.spectral_animation_bins,
        )
        bin_indices = np.searchsorted(edges, self._exact_eigenvalues, side='right') - 1
        self._animation_bin_edges = edges
        self._animation_bin_indices = np.clip(
            bin_indices,
            0,
            self.spectral_animation_bins - 1,
        )

        plt.ion()
        figure, axes = plt.subplots(figsize=(11, 5.5))
        figure.subplots_adjust(bottom=0.18 if self.spectral_animation_manual else 0.12)
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title('PDBO spectral mode amplitudes')
        heights = self._spectral_mode_bin_means()
        bars = axes.bar(
            edges[:-1],
            heights,
            width=np.diff(edges),
            align='edge',
        )
        axes.set_xlim(float(edges[0]), float(edges[-1]))
        axes.set_ylim(0.0, max(float(np.max(heights)) * 1.15, 1e-12))
        axes.set_xlabel('Eigenvalue')
        axes.set_ylabel('Mean projection length')
        axes.grid(axis='y', alpha=0.25)

        if self.spectral_animation_manual:
            next_axes = figure.add_axes((0.68, 0.035, 0.12, 0.06))
            run_axes = figure.add_axes((0.82, 0.035, 0.12, 0.06))
            self._animation_next_button = Button(next_axes, 'Next step')
            self._animation_run_button = Button(run_axes, 'Run')
            self._animation_next_button.on_clicked(self._request_spectral_animation_step)
            self._animation_run_button.on_clicked(self._toggle_spectral_animation_run)

        self._animation_figure = figure
        self._animation_axes = axes
        self._animation_bars = bars
        self._animation_plt = plt
        figure.show()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()

    def _update_spectral_animation(
            self,
            phase: int,
            phases: int,
            phase_iteration: int,
    ) -> None:
        figure = self._animation_figure
        if figure is None or not self._animation_plt.fignum_exists(figure.number):
            self._animation_figure = None
            return

        heights = self._spectral_mode_bin_means()
        for bar, height in zip(self._animation_bars, heights):
            bar.set_height(float(height))

        current_upper = float(self._animation_axes.get_ylim()[1])
        required_upper = max(float(np.max(heights)) * 1.15, 1e-12)
        if required_upper > current_upper:
            self._animation_axes.set_ylim(0.0, required_upper)
        self._animation_axes.set_title(
            f'round {phase}/{phases} | iteration {phase_iteration}/{self.max_iters}'
        )
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        self._animation_plt.pause(0.001)

    def _finish_spectral_animation(self, phase: int, phases: int, phase_iteration: int) -> None:
        if self._animation_figure is None:
            return
        self._update_spectral_animation(phase, phases, phase_iteration)
        if self.spectral_animation_hold and self._animation_figure is not None:
            self._animation_plt.ioff()
            self._animation_plt.show()

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

    @property
    def ObjVal(self):
        return self.objVal

    @property
    def X(self):
        return self.incumbent

    @property
    def Runtime(self):
        return self.solving_time

    @property
    def obj_val(self):
        return self.objVal

    @property
    def objective_history(self):
        return self.objVal_record

    @property
    def timing_history(self):
        return self.timing_record

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
        _, objective_gradient = self._objective_values_and_gradients(old_primal)
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
        self.dual = (old_dual + self.dual_lr * constraint_value).astype(np.float32, copy=False)
        self.perturbation_count += int(np.count_nonzero(trigger))

    def _update_incumbent(self) -> bool:
        rounded = np.rint(self.primal).astype(np.float32)
        candidates = np.concatenate((rounded, self.incumbent[np.newaxis, :]), axis=0)
        values = self._score_candidates(candidates)
        index = int(np.argmin(values))
        candidate_value = float(values[index])
        improved = candidate_value < self.objVal - self.min_delta
        self.objVal = candidate_value
        self.incumbent = candidates[index].copy()
        return improved

    def _restart_from_incumbent(self) -> None:
        """Restart both variables from the first phase's best rounded solution."""
        self.primal[...] = self.incumbent[np.newaxis, :]
        self.dual.fill(self.dual_init)

    def optimize(self):
        self.stop_reason = None
        phases = 2 if self.restart else 1
        if (
            (self.verbose or self.spectral_animation)
            and self.spectrum_distribution is None
        ):
            try:
                self.spectrum_distribution = self._compute_spectrum_distribution()
            except (sparse_linalg.ArpackError, np.linalg.LinAlgError, ValueError) as error:
                self.spectrum_distribution_error = type(error).__name__
                if self.spectral_animation:
                    raise RuntimeError(
                        "failed to compute eigenvectors for spectral animation"
                    ) from error
        if self.verbose and self.spectral_window is None:
            try:
                self.spectral_window = self._compute_spectral_window()
            except (sparse_linalg.ArpackError, np.linalg.LinAlgError, ValueError) as error:
                self.spectral_window_error = type(error).__name__
        if self.verbose:
            self._log_spectrum_distribution()
        if self.spectral_animation and self._animation_figure is None:
            self._setup_spectral_animation()
        self.start_time = time.perf_counter()
        total_step = 0
        last_phase = 1
        last_phase_iteration = 0
        for phase in range(phases):
            current_phase = phase + 1
            last_phase = current_phase
            last_phase_iteration = 0
            if self.verbose:
                self._log_spectral_window(current_phase, phases)
            if self.spectral_animation:
                self._update_spectral_animation(current_phase, phases, 0)
            last_improvement_step = 0
            phase_stop_reason = None
            completed_phase_iterations = 0
            for phase_step in range(self.max_iters):
                if self.spectral_animation and self.spectral_animation_manual:
                    manual_wait = self._wait_for_spectral_animation_step()
                    self.start_time += manual_wait
                step = total_step
                total_step += 1
                step_start = time.perf_counter()
                self._paper_step()
                improved = self._update_incumbent()

                if improved:
                    self.objVal_record.append(self.objVal)
                    self.timing_record.append(time.perf_counter() - self.start_time)
                    last_improvement_step = phase_step

                step_time = time.perf_counter() - step_start
                phase_iteration = phase_step + 1
                completed_phase_iterations = phase_iteration
                last_phase = current_phase
                last_phase_iteration = phase_iteration
                if self.verbose and phase_iteration % self.check_every == 0:
                    self.log(
                        time.perf_counter() - self.start_time,
                        phase + 1,
                        phases,
                        phase_iteration,
                    )
                if (
                    self.spectral_animation
                    and (
                        self.spectral_animation_manual
                        or phase_iteration % self.spectral_animation_every == 0
                    )
                ):
                    self._update_spectral_animation(
                        current_phase,
                        phases,
                        phase_iteration,
                    )
                if self.step_callback is not None:
                    self.step_callback(step, step_time, self.objVal, self.incumbent)

                if (
                    self.timelimit is not None
                    and time.perf_counter() - self.start_time >= self.timelimit
                ):
                    phase_stop_reason = 'timelimit'
                    break
                if (
                    self.patience is not None
                    and phase_step - last_improvement_step >= self.patience
                ):
                    phase_stop_reason = 'patience'
                    break

            if (
                self.spectral_animation
                and not self.spectral_animation_manual
                and completed_phase_iterations % self.spectral_animation_every != 0
            ):
                self._update_spectral_animation(
                    current_phase,
                    phases,
                    completed_phase_iterations,
                )

            if phase_stop_reason == 'timelimit':
                self.stop_reason = phase_stop_reason
                break
            if phase == 0 and self.restart:
                self._restart_from_incumbent()
                continue
            if phase_stop_reason is not None:
                self.stop_reason = phase_stop_reason
                break

        if self.stop_reason is None:
            self.stop_reason = 'max_iters'

        if self.rounding_samples > 0:
            sampled = (
                self.rng.random((self.rounding_samples, self.batch_size, self.n)) < self.primal
            ).astype(np.float32).reshape((-1, self.n))
            candidates = np.concatenate((sampled, self.incumbent[np.newaxis, :]), axis=0)
            values = self._score_candidates(candidates)
            index = int(np.argmin(values))
            candidate_value = float(values[index])
            if candidate_value < self.objVal - self.min_delta:
                self.objVal = candidate_value
                self.incumbent = candidates[index].copy()
                self.objVal_record.append(self.objVal)
                self.timing_record.append(time.perf_counter() - self.start_time)

        self.solving_time = time.perf_counter() - self.start_time
        if self.spectral_animation:
            self._finish_spectral_animation(
                last_phase,
                phases,
                last_phase_iteration,
            )
        return PDBOResult(
            objective=float(self.objVal),
            incumbent=np.asarray(self.incumbent, dtype=np.int32),
            objective_history=list(self.objVal_record),
            timing_history=list(self.timing_record),
            runtime=self.solving_time,
            stop_reason=self.stop_reason,
        )


PDQUBO_CPU = PDBO_CPU
