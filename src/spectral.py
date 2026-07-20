"""Sparse spectral utilities used by PDBO diagnostics and initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


EXACT_SPECTRUM_LIMIT = 2000
EXACT_EIGENPAIR_LIMIT = 256
SPECTRUM_BINS = 20
SLQ_PROBES = 4
SLQ_STEPS = 40
DEFAULT_ANIMATION_MODES = 128


@dataclass(frozen=True)
class SpectrumDistribution:
    mode: str
    edges: np.ndarray
    counts: np.ndarray
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None


@dataclass(frozen=True)
class HsSpectrumCertificate:
    distribution: SpectrumDistribution
    lambda_min: float
    gap_upper_bound: float
    eigen_residual: float


def smallest_eigenpair(
    matrix: sparse.csr_matrix,
    n: int,
    v0: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, float]:
    """Return a conservative estimate of a symmetric matrix's minimum eigenvalue."""
    matrix64 = matrix.astype(np.float64)
    if n <= EXACT_EIGENPAIR_LIMIT:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix64.toarray())
        return float(eigenvalues[0]), eigenvectors[:, 0], 0.0

    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        matrix64,
        k=1,
        which="SA",
        return_eigenvectors=True,
        v0=v0,
        tol=1e-9,
    )
    eigenvalue = float(eigenvalues[0])
    eigenvector = eigenvectors[:, 0]
    residual = float(np.linalg.norm(matrix64 @ eigenvector - eigenvalue * eigenvector))
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(eigenvalue))
    return eigenvalue - max(residual, roundoff), eigenvector, residual


def largest_eigenvalue(matrix: sparse.csr_matrix, n: int) -> Tuple[float, float]:
    """Return a conservative estimate of a symmetric matrix's maximum eigenvalue."""
    matrix64 = matrix.astype(np.float64)
    if n <= EXACT_EIGENPAIR_LIMIT:
        return float(np.linalg.eigvalsh(matrix64.toarray())[-1]), 0.0

    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        matrix64,
        k=1,
        which="LA",
        return_eigenvectors=True,
        tol=1e-9,
    )
    estimate = float(eigenvalues[0])
    eigenvector = eigenvectors[:, 0]
    residual = float(np.linalg.norm(matrix64 @ eigenvector - estimate * eigenvector))
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(estimate))
    return estimate + max(residual, roundoff), residual


def compute_spectral_window(
    matrix: sparse.csr_matrix,
    n: int,
    seed: int,
    exact_eigenvalues: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, int]]:
    """Return ``(lambda_1, lambda_(r+1), r)`` for a symmetric matrix."""
    if n < 2:
        return None

    if exact_eigenvalues is not None:
        eigenvalues = exact_eigenvalues
    elif n <= EXACT_EIGENPAIR_LIMIT:
        eigenvalues = np.linalg.eigvalsh(matrix.toarray().astype(np.float64))
    else:
        matrix64 = matrix.astype(np.float64)
        max_k = min(64, n - 1)
        k = min(8, max_k)
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(n)

        while True:
            eigenvalues = np.sort(
                sparse_linalg.eigsh(
                    matrix64,
                    k=k,
                    which="SA",
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

    multiplicity = int(distinct[0])
    return lambda_1, float(eigenvalues[multiplicity]), multiplicity


def histogram_edges(lambda_min: float, lambda_max: float, bins: int) -> np.ndarray:
    if np.isclose(lambda_min, lambda_max):
        padding = 0.5 * max(1.0, abs(lambda_min))
        lambda_min -= padding
        lambda_max += padding
    return np.linspace(lambda_min, lambda_max, bins + 1, dtype=np.float64)


def _lanczos_decomposition(
    matrix: sparse.csr_matrix,
    probe: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Ritz values/vectors and the Lanczos basis for one probe."""
    n = matrix.shape[0]
    basis = np.empty((n, steps), dtype=np.float64)
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

        coefficients = basis[:, : step + 1].T @ z
        z -= basis[:, : step + 1] @ coefficients

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
        off_diagonal = betas[: actual_steps - 1]
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, eigenvectors = np.linalg.eigh(tridiagonal)
    return nodes, eigenvectors, basis[:, :actual_steps]


def _lanczos_spectral_measure(
    matrix: sparse.csr_matrix,
    probe: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes, eigenvectors, _ = _lanczos_decomposition(matrix, probe, steps)
    return nodes, eigenvectors[0, :] ** 2


def compute_animation_modes(
    matrix: sparse.csr_matrix,
    n: int,
    seed: int,
    modes: int = DEFAULT_ANIMATION_MODES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate modes across a large sparse spectrum with Lanczos-Ritz vectors."""
    if modes < 1:
        raise ValueError("animation modes must be positive")
    steps = min(modes, n)
    rng = np.random.default_rng(seed + 2)
    probe = rng.standard_normal(n)
    nodes, projected_vectors, basis = _lanczos_decomposition(
        matrix.astype(np.float64),
        probe,
        steps,
    )
    ritz_vectors = basis @ projected_vectors
    return nodes, ritz_vectors.astype(np.float32)


def compute_spectrum_distribution(
    matrix: sparse.csr_matrix,
    n: int,
    seed: int,
    include_eigenvectors: bool = False,
    bins: int = SPECTRUM_BINS,
    spectral_range: Optional[Tuple[float, float]] = None,
) -> SpectrumDistribution:
    """Build a full-spectrum histogram, exact when dense decomposition is feasible."""
    if bins < 1:
        raise ValueError("spectrum histogram bins must be positive")
    matrix64 = matrix.astype(np.float64)
    if n <= EXACT_SPECTRUM_LIMIT:
        if include_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix64.toarray())
        else:
            eigenvalues = np.linalg.eigvalsh(matrix64.toarray())
            eigenvectors = None
        edges = histogram_edges(float(eigenvalues[0]), float(eigenvalues[-1]), bins)
        counts, _ = np.histogram(eigenvalues, bins=edges)
        return SpectrumDistribution(
            mode="exact",
            edges=edges,
            counts=counts.astype(np.float64),
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    if spectral_range is None:
        lambda_min = float(
            sparse_linalg.eigsh(matrix64, k=1, which="SA", return_eigenvectors=False)[0]
        )
        lambda_max = float(
            sparse_linalg.eigsh(matrix64, k=1, which="LA", return_eigenvectors=False)[0]
        )
    else:
        lambda_min, lambda_max = map(float, spectral_range)
    edges = histogram_edges(lambda_min, lambda_max, bins)
    normalized_counts = np.zeros(bins, dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    steps = min(SLQ_STEPS, n)

    for _ in range(SLQ_PROBES):
        probe = rng.choice((-1.0, 1.0), size=n)
        nodes, weights = _lanczos_spectral_measure(matrix64, probe, steps)
        probe_counts, _ = np.histogram(nodes, bins=edges, weights=weights)
        normalized_counts += probe_counts

    return SpectrumDistribution(
        mode="approx-slq",
        edges=edges,
        counts=normalized_counts * (n / SLQ_PROBES),
    )


def compute_h_s_spectrum_certificate(
    matrix: sparse.csr_matrix,
    sign: np.ndarray,
    seed: int,
    bins: int = SPECTRUM_BINS,
) -> HsSpectrumCertificate:
    """Return the spectrum of ``H_s`` and its Max-Cut additive-gap certificate."""
    matrix64 = matrix.astype(np.float64).tocsr()
    n = matrix64.shape[0]
    if matrix64.shape != (n, n):
        raise ValueError("H_s certificate requires a square matrix")

    sign64 = np.asarray(sign, dtype=np.float64).reshape(-1)
    if sign64.shape != (n,) or not np.all(np.isin(sign64, (-1.0, 1.0))):
        raise ValueError("H_s certificate requires a sign vector in {-1, +1}^n")

    gamma = -sign64 * (matrix64 @ sign64)
    h_s = (matrix64 + sparse.diags(gamma, format="csr")).tocsr()

    if n <= EXACT_SPECTRUM_LIMIT:
        distribution = compute_spectrum_distribution(
            h_s,
            n,
            seed,
            bins=bins,
        )
        if distribution.eigenvalues is None:
            raise RuntimeError("exact H_s spectrum did not return eigenvalues")
        lambda_min = float(distribution.eigenvalues[0])
        residual = 0.0
    else:
        lambda_min, _, residual = smallest_eigenpair(h_s, n)
        lambda_max, _ = largest_eigenvalue(h_s, n)
        distribution = compute_spectrum_distribution(
            h_s,
            n,
            seed,
            bins=bins,
            spectral_range=(lambda_min, lambda_max),
        )

    # H_s s = 0 analytically, so its true minimum eigenvalue cannot be positive.
    lambda_min = min(lambda_min, 0.0)
    return HsSpectrumCertificate(
        distribution=distribution,
        lambda_min=lambda_min,
        gap_upper_bound=-0.25 * n * lambda_min,
        eigen_residual=residual,
    )
