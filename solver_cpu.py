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
            primal_init: str = 'uniform',
            step_callback: Optional[Callable] = None,
            patience: Optional[int] = None,
            min_delta: float = 0.0,
            check_every: int = 100,
            quadratic_backend: str = 'sparse',
            rounding_samples: int = 0,
            g_type: str = 'quadratic',
            restart: bool = False,
    ):
        if primal_init not in {'uniform', 'half', 'binary'}:
            raise ValueError("primal_init must be 'uniform', 'half', or 'binary'")
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

        self.primal, self.dual = self._init_variables(dual_init, primal_init)
        self.incumbent = np.zeros(self.n, dtype=np.float32)
        self.objVal = float(self._score_candidates(self.incumbent[np.newaxis, :])[0])
        self.objVal_record = [self.objVal]
        self.timing_record = [0.0]
        self.start_time = None
        self.solving_time = None
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _init_variables(self, dual_init: float, primal_init: str) -> Tuple[np.ndarray, np.ndarray]:
        shape = (self.batch_size, self.n)
        if primal_init == 'half':
            primal = np.full(shape, 0.5, dtype=np.float32)
        elif primal_init == 'binary':
            primal = self.rng.integers(0, 2, size=shape).astype(np.float32)
        else:
            primal = self.rng.random(shape, dtype=np.float32)
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
        self.start_time = time.perf_counter()
        self.stop_reason = None
        phases = 2 if self.restart else 1
        total_step = 0
        for phase in range(phases):
            last_improvement_step = 0
            phase_stop_reason = None
            for phase_step in range(self.max_iters):
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
                if self.verbose and phase_iteration % self.check_every == 0:
                    self.log(
                        time.perf_counter() - self.start_time,
                        phase + 1,
                        phases,
                        phase_iteration,
                    )
                if self.step_callback is not None:
                    self.step_callback(step, step_time, self.objVal, self.incumbent)

                if self.timelimit is not None and time.perf_counter() - self.start_time >= self.timelimit:
                    phase_stop_reason = 'timelimit'
                    break
                if self.patience is not None and phase_step - last_improvement_step >= self.patience:
                    phase_stop_reason = 'patience'
                    break

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
        return PDBOResult(
            objective=float(self.objVal),
            incumbent=np.asarray(self.incumbent, dtype=np.int32),
            objective_history=list(self.objVal_record),
            timing_history=list(self.timing_record),
            runtime=self.solving_time,
            stop_reason=self.stop_reason,
        )


PDQUBO_CPU = PDBO_CPU
