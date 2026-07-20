#!/usr/bin/env python3
"""Landscape and trajectory diagnostics for PDBO on Max-Cut.

The module deliberately keeps the experiment driver independent of ``src/solver``.
That makes the ablations comparable: every method receives the same float32 initial
states, uses the same archive rule, and consumes one simultaneous old-state update
per iteration.  The ``pdbo`` update mirrors :meth:`PDBO_CPU._paper_step` (including
the optional centre kick), while the other methods change only the requested part
of the mechanism.

The command line driver writes a long-form trajectory CSV and a JSON sidecar.  It
is intended for small exact graphs and sparse Gset trajectories; full dense spectra
are used only below ``--exact-limit``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem_parser import parse_gset, random_graph


METHODS = (
    "direct_matched",
    "direct_tuned",
    "static_convex",
    "scalar_dual",
    "pdbo",
)


@dataclass(frozen=True)
class MaxCutProblem:
    """Sparse Max-Cut representation used by all methods."""

    n: int
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    W: sparse.csr_matrix
    degree: np.ndarray
    total_weight: float
    name: str = "graph"

    @classmethod
    def from_graph(cls, graph: nx.Graph, name: str = "graph") -> "MaxCutProblem":
        edges = []
        for u, v, data in graph.edges(data=True):
            if u == v:
                # A self-loop never contributes to a cut.  Ignoring it also keeps
                # the zero diagonal convention used by the spectral theory.
                continue
            edges.append((int(u), int(v), float(data.get("weight", 1.0))))
        n = graph.number_of_nodes()
        if n == 0:
            raise ValueError("Max-Cut graph must contain at least one node")
        rows = np.asarray([u for u, _, _ in edges], dtype=np.int64)
        cols = np.asarray([v for _, v, _ in edges], dtype=np.int64)
        weights = np.asarray([w for _, _, w in edges], dtype=np.float64)
        if rows.size:
            w_rows = np.concatenate((rows, cols))
            w_cols = np.concatenate((cols, rows))
            w_values = np.concatenate((weights, weights))
            matrix = sparse.coo_matrix(
                (w_values, (w_rows, w_cols)), shape=(n, n), dtype=np.float64
            ).tocsr()
        else:
            matrix = sparse.csr_matrix((n, n), dtype=np.float64)
        degree = np.asarray(matrix.sum(axis=1)).ravel()
        return cls(
            n=n,
            rows=rows,
            cols=cols,
            weights=weights,
            W=matrix,
            degree=degree,
            total_weight=float(np.sum(weights)),
            name=name,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any], name: str = "graph") -> "MaxCutProblem":
        indices = np.asarray(data["Q_indices"], dtype=np.int64)
        values = np.asarray(data["Q_values"], dtype=np.float64)
        if indices.shape[0] != 2:
            raise ValueError("Q_indices must have shape (2, m)")
        # ``generate_Max_cut`` stores both orientations.  Keeping one orientation
        # is useful for cut evaluation and avoids double counting total weight.
        seen: dict[tuple[int, int], float] = {}
        for i, j, value in zip(indices[0], indices[1], values):
            if i == j:
                continue
            key = (min(int(i), int(j)), max(int(i), int(j)))
            seen[key] = seen.get(key, 0.0) + float(value)
        # Each undirected edge occurs twice in the generated Q, hence divide by 2.
        edges = [(i, j, value / 2.0) for (i, j), value in seen.items()]
        graph = nx.Graph()
        graph.add_nodes_from(range(int(data["num_vars"])))
        graph.add_weighted_edges_from(edges)
        return cls.from_graph(graph, name=name)


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mode: str
    lambda_min: float
    lambda_max: float
    exact: bool


@dataclass
class LandscapeConfig:
    """Runtime controls shared by every trajectory."""

    steps: int = 300
    batch: int = 8
    beta: float = 0.02
    dual_init: Optional[float] = None
    psd_margin: float = 0.1
    spectral_fraction: float = 0.5
    direct_tuned_fraction: float = 0.5
    rho: float = 0.05
    delta: float = 1e-8
    sample_every: int = 1
    hessian_every: int = 10
    exact_limit: int = 2000
    mode_count: int = 16
    seed: int = 0
    kick: bool = True
    init: str = "center_uniform"
    start_radius: float = 0.0
    bad_local_starts: int = 0

    def validate(self) -> None:
        if self.steps < 0 or self.batch < 1:
            raise ValueError("steps must be non-negative and batch must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < self.spectral_fraction < 1.0:
            raise ValueError("spectral_fraction must lie in (0, 1)")
        if self.direct_tuned_fraction <= 0.0:
            raise ValueError("direct_tuned_fraction must be positive")
        if not 0.0 <= self.rho < 0.5:
            raise ValueError("rho must lie in [0, 0.5)")
        if not 0 < self.delta < 0.5:
            raise ValueError("delta must lie in (0, 0.5)")
        if self.sample_every < 1 or self.hessian_every < 1 or self.exact_limit < 1 or self.mode_count < 1:
            raise ValueError("sample_every, hessian_every, exact_limit and mode_count must be positive")
        if self.init not in {"center_uniform", "uniform", "half", "binary"}:
            raise ValueError("unsupported initialization")
        if not 0.0 <= self.start_radius < 0.5:
            raise ValueError("start_radius must lie in [0, 0.5)")
        if self.bad_local_starts < 0:
            raise ValueError("bad_local_starts must be non-negative")


def compute_spectrum(
    problem: MaxCutProblem,
    exact_limit: int = 2000,
    mode_count: int = 16,
    seed: int = 0,
) -> Spectrum:
    """Return an exact small spectrum or sparse extremal Ritz modes.

    For a large graph the returned basis is intentionally only a diagnostic basis;
    ``lambda_min``/``lambda_max`` are computed with conservative residual padding.
    """

    n = problem.n
    if n <= exact_limit:
        values, vectors = np.linalg.eigh(problem.W.toarray())
        return Spectrum(
            eigenvalues=np.asarray(values, dtype=np.float64),
            eigenvectors=np.asarray(vectors, dtype=np.float64),
            mode="exact",
            lambda_min=float(values[0]),
            lambda_max=float(values[-1]),
            exact=True,
        )

    k = min(max(2, mode_count), n - 1)
    low_k = min(max(1, k // 2), n - 1)
    high_k = min(max(1, k - low_k), n - 1)
    rng = np.random.default_rng(seed)
    low_values, low_vectors = sparse_linalg.eigsh(
        problem.W.astype(np.float64), k=low_k, which="SA", v0=rng.standard_normal(n), tol=1e-7
    )
    high_values, high_vectors = sparse_linalg.eigsh(
        problem.W.astype(np.float64), k=high_k, which="LA", v0=rng.standard_normal(n), tol=1e-7
    )
    values = np.concatenate((low_values, high_values))
    vectors = np.concatenate((low_vectors, high_vectors), axis=1)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    # Ritz residual padding keeps the reported extrema conservative for step sizing.
    low_residual = np.linalg.norm(problem.W @ vectors[:, 0] - values[0] * vectors[:, 0])
    high_residual = np.linalg.norm(problem.W @ vectors[:, -1] - values[-1] * vectors[:, -1])
    return Spectrum(
        eigenvalues=values,
        eigenvectors=vectors,
        mode="sparse-extrema",
        lambda_min=float(values[0] - low_residual),
        lambda_max=float(values[-1] + high_residual),
        exact=False,
    )


def initial_states(
    n: int,
    batch: int,
    rho: float,
    seed: int,
    init: str = "center_uniform",
) -> np.ndarray:
    """Generate a reproducible common random initial batch."""

    rng = np.random.default_rng(seed)
    if init == "half":
        return np.full((batch, n), 0.5, dtype=np.float32)
    if init == "binary":
        return rng.integers(0, 2, size=(batch, n), dtype=np.int8).astype(np.float32)
    if init == "uniform":
        return rng.random((batch, n), dtype=np.float32)
    low = np.float32(0.5 - rho)
    high = np.float32(0.5 + rho)
    return (low + (high - low) * rng.random((batch, n), dtype=np.float32)).astype(np.float32)


def cut_values(problem: MaxCutProblem, bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    if not problem.weights.size:
        return np.zeros(bits.shape[0], dtype=np.float64)
    return np.sum(
        problem.weights[None, :]
        * (bits[:, problem.rows] != bits[:, problem.cols]),
        axis=1,
        dtype=np.float64,
    )


def expected_cut(problem: MaxCutProblem, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    if not problem.weights.size:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    p = probabilities
    return np.sum(
        problem.weights[None, :]
        * (p[:, problem.rows] + p[:, problem.cols] - 2.0 * p[:, problem.rows] * p[:, problem.cols]),
        axis=1,
        dtype=np.float64,
    )


def flip_gains(problem: MaxCutProblem, bits: np.ndarray) -> np.ndarray:
    """Return exact cut gains for all one-bit flips."""

    bits = np.asarray(bits, dtype=np.float64).ravel()
    signs = 2.0 * bits - 1.0
    return signs * np.asarray(problem.W @ signs).ravel()


def greedy_one_flip(
    problem: MaxCutProblem,
    bits: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float, int]:
    """Apply best-improvement one-flip search to a Max-Cut candidate."""

    current = np.asarray(bits, dtype=np.int8).ravel().copy()
    current_cut = float(cut_values(problem, current)[0])
    steps = 0
    while True:
        gains = flip_gains(problem, current)
        index = int(np.argmax(gains))
        gain = float(gains[index])
        if gain <= tolerance:
            break
        current[index] = 1 - current[index]
        current_cut += gain
        steps += 1
    return current, current_cut, steps


def enumerate_cuts(
    problem: MaxCutProblem,
    max_n: int = 24,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    """Enumerate all cuts and one-flip local optima for a small graph."""

    if problem.n > max_n:
        raise ValueError(f"exact enumeration is limited to n <= {max_n}")
    total = 1 << problem.n
    values = np.empty(total, dtype=np.float64)
    bit_positions = np.arange(problem.n, dtype=np.uint64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        states = np.arange(start, stop, dtype=np.uint64)
        bits = ((states[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
        values[start:stop] = cut_values(problem, bits)
    optimum = float(np.max(values))
    optimal_indices = np.flatnonzero(np.isclose(values, optimum, rtol=0.0, atol=1e-9))
    local_mask = np.ones(total, dtype=bool)
    states = np.arange(total, dtype=np.uint64)
    for index in range(problem.n):
        local_mask &= values >= values[states ^ (np.uint64(1) << np.uint64(index))] - 1e-12
    local_indices = np.flatnonzero(local_mask)
    return {
        "n": problem.n,
        "state_count": total,
        "optimum_cut": optimum,
        "optimal_indices": optimal_indices.astype(np.int64),
        "optimal_count": int(optimal_indices.size),
        "local_optima_indices": local_indices.astype(np.int64),
        "local_optima_count": int(local_indices.size),
        "suboptimal_local_indices": local_indices[values[local_indices] < optimum - 1e-9].astype(np.int64),
        "cut_values": values,
    }


def enumerate_local_optima(problem: MaxCutProblem, max_n: int = 24) -> dict[str, Any]:
    """Compatibility-facing alias for the exact local-optimum analysis."""

    return enumerate_cuts(problem, max_n=max_n)


def _json_number(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(v) for v in value]
    return value


class _Trajectory:
    """Stateful common-update runner used internally and by tests."""

    def __init__(
        self,
        problem: MaxCutProblem,
        x0: np.ndarray,
        method: str,
        config: LandscapeConfig,
        spectrum: Spectrum,
        y0: float,
        alpha: float,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        self.problem = problem
        self.x = np.asarray(x0, dtype=np.float32).copy()
        if self.x.ndim != 2 or self.x.shape[1] != problem.n:
            raise ValueError("x0 must have shape (batch, n)")
        self.method = method
        self.config = config
        self.spectrum = spectrum
        self.reference_y0 = float(y0)
        # Direct f(x) baselines have no dual term.  Keep the spectral y0 used to
        # choose the matched step as a separate reference parameter, otherwise
        # their Hessian diagnostics would accidentally analyze W+Diag(y0).
        self.y0 = 0.0 if method in {"direct_matched", "direct_tuned"} else float(y0)
        self.alpha = float(alpha)
        self.beta = float(config.beta)
        self.y = np.full_like(self.x, self.y0, dtype=np.float32)
        self.steps = 0
        self.perturbations = 0
        self.first_clip_step = -1
        self._candidate_hashes: set[str] = set()
        initial_bits = np.rint(self.x).astype(np.float32)
        candidates = np.concatenate((initial_bits, np.zeros((1, problem.n), dtype=np.float32)))
        candidate_cuts = cut_values(problem, candidates)
        index = int(np.argmax(candidate_cuts))
        self.archive_bits = candidates[index].copy().astype(np.int8)
        self.archive_cut = float(candidate_cuts[index])
        self.archive_step = 0
        self.archive_batch = index if index < self.x.shape[0] else -1
        self._observe_candidates(initial_bits, step=0)
        self.rows: list[dict[str, Any]] = []
        self._cached_hessian: Optional[tuple[float, float, float, int]] = None
        self._cached_hessian_step = -1

    @staticmethod
    def _candidate_hash(bits: np.ndarray) -> str:
        packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little")
        return packed.tobytes().hex()

    def _observe_candidates(self, bits: np.ndarray, step: int) -> None:
        candidates = np.asarray(bits, dtype=np.float32)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        for candidate in candidates:
            self._candidate_hashes.add(self._candidate_hash(candidate))
        candidate_cuts = cut_values(self.problem, candidates)
        index = int(np.argmax(candidate_cuts))
        if float(candidate_cuts[index]) > self.archive_cut + 1e-12:
            self.archive_cut = float(candidate_cuts[index])
            self.archive_bits = candidates[index].astype(np.int8).copy()
            self.archive_step = int(step)
            self.archive_batch = index

    def _archive_neighbor_count(self) -> int:
        covered = 0
        for index in range(self.problem.n):
            neighbor = self.archive_bits.copy()
            neighbor[index] = 1 - neighbor[index]
            covered += int(self._candidate_hash(neighbor) in self._candidate_hashes)
        return covered

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        z = x.astype(np.float64) - 0.5
        return (2.0 * (self.problem.W @ z.T).T).astype(np.float32)

    def _update(self) -> dict[str, Any]:
        old_x = self.x.copy()
        old_y = self.y.copy()
        objective_gradient = self._objective_gradient(old_x)
        if self.method in {"static_convex", "scalar_dual", "pdbo"}:
            lagrangian_gradient = objective_gradient + old_y * (2.0 * old_x - 1.0)
        else:
            lagrangian_gradient = objective_gradient
        raw = old_x - np.float32(self.alpha) * lagrangian_gradient
        clipped = np.logical_or(raw < 0.0, raw > 1.0)
        projected = np.clip(raw, 0.0, 1.0)

        if self.method == "pdbo" and self.config.kick:
            at_center = np.abs(old_x - 0.5) <= max(self.config.delta, float(np.spacing(np.float32(0.5))))
            trigger = at_center & (np.abs(lagrangian_gradient) <= 2.0 * self.config.delta) & (old_y <= 0.0)
            perturbed = np.where(old_x <= 0.5, 0.5 - self.config.delta, 0.5 + self.config.delta)
            new_x = np.where(trigger, perturbed, projected).astype(np.float32)
            self.perturbations += int(np.count_nonzero(trigger))
        else:
            trigger = np.zeros_like(old_x, dtype=bool)
            new_x = projected.astype(np.float32)

        constraint = old_x * old_x - old_x
        if self.method == "pdbo":
            new_y = old_y + np.float32(self.beta) * constraint
        elif self.method == "scalar_dual":
            scalar = np.mean(old_y + np.float32(self.beta) * constraint, axis=1, keepdims=True)
            new_y = np.broadcast_to(scalar, old_y.shape).copy()
        else:
            new_y = old_y

        self.x = new_x
        self.y = np.asarray(new_y, dtype=np.float32)
        self.steps += 1
        if self.first_clip_step < 0 and np.any(clipped):
            self.first_clip_step = self.steps
        self._observe_candidates(np.rint(self.x).astype(np.float32), step=self.steps)
        return {
            "old_x": old_x,
            "old_y": old_y,
            "raw": raw,
            "clipped": clipped,
            "trigger": trigger,
            "constraint": constraint,
        }

    def _hessian_metrics(self) -> tuple[float, float, float, int]:
        """Return mean min eigenvalue, max eigenvalue, negative index and mode."""

        # Uniform duals (including direct baselines with y=0) inherit the known
        # spectrum exactly, so no eigensolver is needed at every sample.
        uniform = np.max(np.ptp(self.y.astype(np.float64), axis=1)) <= 1e-12
        if uniform:
            shifts = np.mean(self.y.astype(np.float64), axis=1)
            minima = self.spectrum.lambda_min + shifts
            maxima = self.spectrum.lambda_max + shifts
            if self.spectrum.exact:
                negatives = [
                    int(np.count_nonzero(self.spectrum.eigenvalues + shift < -1e-10))
                    for shift in shifts
                ]
            else:
                negatives = [-1 for _ in shifts]
            return (
                2.0 * float(np.mean(minima)),
                2.0 * float(np.mean(maxima)),
                float(np.mean(minima)),
                int(round(float(np.mean(negatives)))) if negatives else -1,
            )

        # Exact coordinate-wise Hessians are useful for tiny landscape studies,
        # but become needlessly expensive for a large batch.  For larger problems
        # track the row with the largest dual anisotropy and use sparse extrema.
        if self.spectrum.exact and self.problem.n <= 64:
            rows = self.y.astype(np.float64)
        else:
            anisotropy = np.std(self.y.astype(np.float64), axis=1)
            rows = self.y[np.argsort(anisotropy)[-1:]].astype(np.float64)
        minima = []
        maxima = []
        negative = []
        for row in rows:
            shifted = self.problem.W + sparse.diags(row, format="csr")
            if self.spectrum.exact and self.problem.n <= 64:
                values = np.linalg.eigvalsh(shifted.toarray())
                minima.append(float(values[0]))
                maxima.append(float(values[-1]))
                negative.append(int(np.count_nonzero(values < -1e-10)))
            else:
                low, _ = sparse_linalg.eigsh(shifted, k=1, which="SA", tol=1e-5)
                high, _ = sparse_linalg.eigsh(shifted, k=1, which="LA", tol=1e-5)
                minima.append(float(low[0]))
                maxima.append(float(high[0]))
                negative.append(-1)
        min_value = float(np.mean(minima)) if minima else float("nan")
        max_value = float(np.mean(maxima)) if maxima else float("nan")
        neg_index = int(round(float(np.mean(negative)))) if negative else -1
        return 2.0 * min_value, 2.0 * max_value, min_value, neg_index

    def _metrics(self, step: int, update: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        bits = np.rint(self.x).astype(np.float32)
        current_cuts = cut_values(self.problem, bits)
        current_index = int(np.argmax(current_cuts))
        expected = expected_cut(self.problem, self.x)
        expected_index = int(np.argmax(expected))
        self._observe_candidates(bits, step=step)
        z = self.x.astype(np.float64) - 0.5
        fractionality = 1.0 - 4.0 * np.mean(z * z, axis=1)
        coefficients = z @ self.spectrum.eigenvectors
        mode_energy = np.mean(coefficients * coefficients, axis=0)
        represented = float(np.sum(mode_energy))
        total_z_energy = float(np.mean(np.sum(z * z, axis=1)))
        if represented > 0.0:
            mode_probability = mode_energy / represented
            entropy = float(-np.sum(mode_probability[mode_probability > 0.0] * np.log(mode_probability[mode_probability > 0.0])))
        else:
            entropy = 0.0
        # This is the actual quadratic Rayleigh quotient, rather than the
        # quotient of a truncated low/high Ritz basis (which is biased on Gset).
        wz = (self.problem.W @ z.T).T
        z_norm = np.sum(z * z, axis=1)
        rayleigh_values = np.divide(
            np.sum(z * wz, axis=1),
            z_norm,
            out=np.full(z_norm.shape, np.nan, dtype=np.float64),
            where=z_norm > 1e-30,
        )
        rayleigh = float(np.nanmean(rayleigh_values)) if np.any(np.isfinite(rayleigh_values)) else float("nan")
        low_count = min(5, mode_energy.size)
        low_energy = float(np.sum(mode_energy[:low_count]) / represented) if represented > 0.0 else 0.0
        if self._cached_hessian is None or step == 0 or step % self.config.hessian_every == 0:
            self._cached_hessian = self._hessian_metrics()
            self._cached_hessian_step = int(step)
        h_min, h_max, shifted_min, h_index = self._cached_hessian
        dual_mean = np.mean(self.y, axis=1)
        dual_anisotropy = self.y.astype(np.float64) - dual_mean[:, None]
        gains = flip_gains(self.problem, self.archive_bits)
        neighbor_count = self._archive_neighbor_count()
        if step == self.config.steps:
            _, refined_cut, refinement_steps = greedy_one_flip(
                self.problem,
                self.archive_bits,
            )
        else:
            refined_cut = float("nan")
            refinement_steps = 0
        row: dict[str, Any] = {
            "problem": self.problem.name,
            "n": self.problem.n,
            "method": self.method,
            "step": int(step),
            "batch": self.x.shape[0],
            "alpha": self.alpha,
            "beta": self.beta,
            "dual_init": self.y0,
            "reference_dual_init": self.reference_y0,
            "spectral_mode": self.spectrum.mode,
            "lambda_min": self.spectrum.lambda_min,
            "lambda_max": self.spectrum.lambda_max,
            "archive_cut": self.archive_cut,
            "current_cut": float(current_cuts[current_index]),
            "current_expected_cut": float(expected[expected_index]),
            # Keep short aliases in the long-form output so downstream notebooks
            # need not know whether the value is the best current batch member.
            "expected_cut": float(expected[expected_index]),
            "mean_expected_cut": float(np.mean(expected)),
            "fractionality": float(np.mean(fractionality)),
            "fractionality_max": float(np.max(fractionality)),
            "saturation_fraction": float(np.mean((self.x <= 0.0) | (self.x >= 1.0))),
            "clip_fraction": float(np.mean(update["clipped"])) if update is not None else 0.0,
            "kick_fraction": float(np.mean(update["trigger"])) if update is not None else 0.0,
            "dual_mean": float(np.mean(dual_mean)),
            "dual_std": float(np.std(self.y)),
            "dual_anisotropy_rms": float(np.sqrt(np.mean(dual_anisotropy ** 2))),
            "dual_min": float(np.min(self.y)),
            "dual_max": float(np.max(self.y)),
            "modal_energy_fraction_low5": low_energy,
            "modal_energy_fraction_represented": represented / max(total_z_energy, 1e-15),
            "modal_entropy": entropy,
            "modal_effective_rank": float(np.exp(entropy)),
            "rayleigh": rayleigh,
            "hessian_min": h_min,
            "hessian_max": h_max,
            "hessian_shifted_min": shifted_min,
            "hessian_negative_index": h_index,
            "hessian_index": h_index,
            "hessian_evaluated_step": self._cached_hessian_step,
            "max_flip_gain": float(np.max(gains)) if gains.size else 0.0,
            "mean_flip_gain": float(np.mean(gains)) if gains.size else 0.0,
            "positive_flip_count": int(np.count_nonzero(gains > 1e-12)),
            "refined_cut": refined_cut,
            "refinement_gain": refined_cut - self.archive_cut,
            "refinement_steps": int(refinement_steps),
            "archive_step": int(self.archive_step),
            "archive_batch": int(self.archive_batch),
            "distinct_candidates": len(self._candidate_hashes),
            "archive_neighbor_count": neighbor_count,
            "archive_neighbor_coverage": neighbor_count / max(self.problem.n, 1),
            "first_clip_step": int(self.first_clip_step),
            "steps": int(self.steps),
            "perturbations": int(self.perturbations),
        }
        for index, value in enumerate(mode_energy[: min(8, mode_energy.size)]):
            row[f"mode_energy_{index}"] = float(value)
        return row

    def run(self) -> list[dict[str, Any]]:
        self.rows.append(self._metrics(0))
        for step in range(1, self.config.steps + 1):
            update = self._update()
            if step % self.config.sample_every == 0 or step == self.config.steps:
                self.rows.append(self._metrics(step, update))
        return self.rows


def _resolve_parameters(
    spectrum: Spectrum,
    config: LandscapeConfig,
) -> tuple[float, float]:
    y0 = config.dual_init
    if y0 is None:
        y0 = max(-spectrum.lambda_min + config.psd_margin, np.finfo(np.float32).tiny)
    denominator = spectrum.lambda_max + float(y0)
    if denominator <= 0.0:
        denominator = max(abs(spectrum.lambda_min), abs(spectrum.lambda_max), 1.0)
    alpha = config.spectral_fraction / denominator
    return float(y0), float(alpha)


def run_trajectory(
    problem: MaxCutProblem,
    method: str,
    config: LandscapeConfig,
    x0: Optional[np.ndarray] = None,
    spectrum: Optional[Spectrum] = None,
    spectral_fraction: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Run one method from a supplied common state and return sampled rows."""

    config.validate()
    if spectrum is None:
        spectrum = compute_spectrum(problem, config.exact_limit, config.mode_count, config.seed)
    if x0 is None:
        x0 = initial_states(problem.n, config.batch, config.rho, config.seed, config.init)
    y0, alpha = _resolve_parameters(spectrum, config)
    if spectral_fraction is not None:
        if not 0.0 < spectral_fraction < 1.0:
            raise ValueError("spectral_fraction must lie in (0, 1)")
        alpha = float(spectral_fraction) / max(spectrum.lambda_max + y0, 1e-12)
    if method == "direct_tuned":
        scale = max(abs(spectrum.lambda_min), abs(spectrum.lambda_max), 1e-12)
        alpha = config.direct_tuned_fraction / scale
    trajectory = _Trajectory(problem, x0, method, config, spectrum, y0, alpha)
    rows = trajectory.run()
    for row in rows:
        row["seed"] = int(config.seed)
        if method == "direct_tuned":
            row["spectral_fraction"] = None
            row["fraction_label"] = "tuned"
        else:
            row["spectral_fraction"] = float(spectral_fraction if spectral_fraction is not None else config.spectral_fraction)
            row["fraction_label"] = f"{row['spectral_fraction']:g}"
    return rows


def run_bad_local_starts(
    problem: MaxCutProblem,
    enumeration: Mapping[str, Any],
    config: LandscapeConfig,
    methods: Sequence[str] = METHODS,
    count: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Start trajectories near suboptimal one-flip optima and measure escape."""

    indices = np.asarray(enumeration.get("suboptimal_local_indices", []), dtype=np.int64)
    cuts = np.asarray(enumeration["cut_values"], dtype=np.float64)
    if indices.size:
        indices = indices[np.argsort(cuts[indices], kind="stable")]
    if count is None:
        count = config.bad_local_starts
    if count and indices.size > count:
        indices = indices[:count]
    if not indices.size:
        return []
    spectrum = compute_spectrum(problem, config.exact_limit, config.mode_count, config.seed)
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 100003)
    for start_index in indices:
        bits = ((np.uint64(start_index) >> np.arange(problem.n, dtype=np.uint64)) & 1).astype(np.float32)
        # Keep the same start for every method while moving just inside the box.
        x_start = 0.5 + (2.0 * bits - 1.0) * (0.5 - config.start_radius)
        x_start = np.broadcast_to(x_start, (config.batch, problem.n)).copy()
        if config.batch > 1:
            x_start += rng.normal(0.0, config.start_radius * 0.1, size=x_start.shape).astype(np.float32)
            x_start = np.clip(x_start, 0.0, 1.0)
        start_cut = float(cuts[int(start_index)])
        for method in methods:
            rows = run_trajectory(problem, method, config, x0=x_start, spectrum=spectrum)
            # For an escape experiment the incumbent must be the supplied bad
            # vertex, rather than the generic all-zero candidate used by the
            # production archive.  Otherwise a signed instance could report an
            # artificial "escape" before taking a step.
            running_best = start_cut
            for row in rows:
                running_best = max(running_best, float(row["archive_cut"]))
                row["archive_cut"] = running_best
            escaped = [row for row in rows if row["archive_cut"] > start_cut + 1e-9]
            final = rows[-1]
            output.append(
                {
                    "problem": problem.name,
                    "seed": int(config.seed),
                    "start_index": int(start_index),
                    "start_cut": start_cut,
                    "method": method,
                    "escaped": int(bool(escaped)),
                    "escape_step": int(escaped[0]["step"]) if escaped else -1,
                    "best_cut": float(max(row["archive_cut"] for row in rows)),
                    "final_cut": float(final["archive_cut"]),
                    "global_optimum": float(enumeration["optimum_cut"]),
                    "hit_global": int(final["archive_cut"] >= float(enumeration["optimum_cut"]) - 1e-9),
                }
            )
    return output


def run_experiment(
    problem: MaxCutProblem,
    config: LandscapeConfig,
    methods: Sequence[str] = METHODS,
    spectral_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    exact: bool = True,
) -> dict[str, Any]:
    """Run paired methods and return rows, summaries and optional exact data."""

    config.validate()
    spectral_fractions = tuple(float(value) for value in spectral_fractions)
    if not spectral_fractions:
        raise ValueError("spectral_fractions must not be empty")
    if any(not 0.0 < value < 1.0 for value in spectral_fractions):
        raise ValueError("every spectral fraction must lie in (0, 1)")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
    spectrum = compute_spectrum(problem, config.exact_limit, config.mode_count, config.seed)
    x0 = initial_states(problem.n, config.batch, config.rho, config.seed, config.init)
    rows: list[dict[str, Any]] = []
    # Use a single pre-generated x0 for every method/fraction.  The direct tuned
    # method changes only its scale, never the random state or archive policy.
    for fraction in spectral_fractions:
        for method in methods:
            # direct_tuned is scale-tuned against |lambda| and does not depend on
            # the continuation fraction.  Run it once and label it explicitly.
            if method == "direct_tuned" and fraction != spectral_fractions[0]:
                continue
            method_rows = run_trajectory(problem, method, config, x0=x0, spectrum=spectrum, spectral_fraction=fraction)
            rows.extend(method_rows)
    summary: list[dict[str, Any]] = []
    for method in methods:
        summary_fractions = (spectral_fractions[:1] if method == "direct_tuned" else spectral_fractions)
        for fraction in summary_fractions:
            selected = [
                row for row in rows
                if row["method"] == method
                and ((method == "direct_tuned" and row["fraction_label"] == "tuned")
                     or row["spectral_fraction"] == float(fraction))
            ]
            if not selected:
                continue
            final = selected[-1]
            summary.append(
                {
                    "problem": problem.name,
                    "seed": int(config.seed),
                    "method": method,
                    "spectral_fraction": None if method == "direct_tuned" else float(fraction),
                    "fraction_label": "tuned" if method == "direct_tuned" else f"{float(fraction):g}",
                    "initial_cut": float(selected[0]["archive_cut"]),
                    "final_archive_cut": float(final["archive_cut"]),
                    "final_current_cut": float(final["current_cut"]),
                    "final_expected_cut": float(final["current_expected_cut"]),
                    "final_fractionality": float(final["fractionality"]),
                    "final_rayleigh": float(final["rayleigh"]),
                    "final_hessian_min": float(final["hessian_min"]),
                    "final_max_flip_gain": float(final["max_flip_gain"]),
                    "refined_cut": float(final["refined_cut"]),
                    "refinement_gain": float(final["refinement_gain"]),
                    "refinement_steps": int(final["refinement_steps"]),
                    "archive_step": int(final["archive_step"]),
                    "distinct_candidates": int(final["distinct_candidates"]),
                    "archive_neighbor_count": int(final["archive_neighbor_count"]),
                    "archive_neighbor_coverage": float(final["archive_neighbor_coverage"]),
                    "first_clip_step": int(final["first_clip_step"]),
                    "perturbations": int(final["perturbations"]),
                }
            )
    exact_data: Optional[dict[str, Any]] = None
    escape_data: list[dict[str, Any]] = []
    if exact and problem.n <= min(config.exact_limit, 24):
        exact_data = enumerate_cuts(problem)
        if config.bad_local_starts:
            escape_data = run_bad_local_starts(problem, exact_data, config, methods)
        # The full cut table is useful in Python but too large for the JSON sidecar.
        exact_data = {key: value for key, value in exact_data.items() if key != "cut_values"}
    return {
        "metadata": {
            "problem": problem.name,
            "n": problem.n,
            "total_weight": problem.total_weight,
            "steps": config.steps,
            "batch": config.batch,
            "beta": config.beta,
            "rho": config.rho,
            "hessian_every": config.hessian_every,
            "seed": config.seed,
            "methods": list(methods),
            "spectral_fractions": [float(value) for value in spectral_fractions],
            "spectrum_mode": spectrum.mode,
            "lambda_min": spectrum.lambda_min,
            "lambda_max": spectrum.lambda_max,
            "alpha_rule": "c/(lambda_max+y0)",
        },
        "rows": rows,
        "summary": summary,
        "exact": exact_data,
        "escape": escape_data,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: _json_number(row.get(key, "")) for key in fieldnames} for row in rows])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_number(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiny", "gset"), default="tiny")
    parser.add_argument("--gset-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dual-init", type=float, default=None)
    parser.add_argument("--psd-margin", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=1e-8)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--hessian-every", type=int, default=10)
    parser.add_argument("--exact-limit", type=int, default=2000)
    parser.add_argument("--mode-count", type=int, default=16)
    parser.add_argument("--spectral-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--direct-tuned-fraction", type=float, default=0.5)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--bad-local-starts", type=int, default=0)
    parser.add_argument("--start-radius", type=float, default=0.0)
    parser.add_argument("--no-kick", action="store_true")
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("research/results/maxcut_landscape"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = LandscapeConfig(
        steps=args.steps,
        batch=args.batch,
        beta=args.beta,
        dual_init=args.dual_init,
        psd_margin=args.psd_margin,
        rho=args.rho,
        delta=args.delta,
        sample_every=args.sample_every,
        hessian_every=args.hessian_every,
        exact_limit=args.exact_limit,
        mode_count=args.mode_count,
        direct_tuned_fraction=args.direct_tuned_fraction,
        seed=args.seeds[0],
        kick=not args.no_kick,
        start_radius=args.start_radius,
        bad_local_starts=args.bad_local_starts,
    )
    config.validate()
    payloads = []
    if args.mode == "tiny":
        graph = random_graph(args.n, args.degree, seed=args.graph_seed)
        payloads.append((MaxCutProblem.from_graph(graph, f"random-{args.n}-{args.degree}-{args.graph_seed}"), args.seeds))
    else:
        for gset_id in dict.fromkeys(args.gset_ids):
            graph = parse_gset(str(gset_id))
            payloads.append((MaxCutProblem.from_graph(graph, f"G{gset_id}"), args.seeds))
    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_escape: list[dict[str, Any]] = []
    exact_payload: dict[str, Any] = {}
    for problem, seeds in payloads:
        for seed in dict.fromkeys(seeds):
            run_config = LandscapeConfig(**{**config.__dict__, "seed": int(seed)})
            result = run_experiment(problem, run_config, args.methods, args.spectral_fractions, exact=not args.no_exact)
            all_rows.extend(result["rows"])
            all_summary.extend(result["summary"])
            all_escape.extend(result["escape"])
            if result["exact"] is not None:
                exact_payload[f"{problem.name}/seed={seed}"] = result["exact"]
            print(f"{problem.name} seed={seed}: {len(result['rows'])} trajectory rows")
    prefix = args.output
    csv_path = prefix if prefix.suffix == ".csv" else prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    summary_path = prefix.with_name(prefix.name + "-summary.csv")
    escape_path = prefix.with_name(prefix.name + "-escape.csv")
    _write_csv(csv_path, all_rows)
    _write_csv(summary_path, all_summary)
    if all_escape:
        _write_csv(escape_path, all_escape)
    _write_json(json_path, {"summary": all_summary, "escape": all_escape, "exact": exact_payload})
    print(f"saved {csv_path}")
    print(f"saved {summary_path}")
    if all_escape:
        print(f"saved {escape_path}")
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
