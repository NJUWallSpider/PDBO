"""Local-search refinements for PDBO incumbents."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefinementResult:
    bits: np.ndarray
    objective: float
    steps: int
    seconds: float


def qubo_objective(bits, data):
    x = np.asarray(bits, dtype=np.float32)
    qx = data["Q_sparse"].dot(x)
    return float(x.dot(qx) + np.dot(data["c"], x) + data.get("objective_offset", 0.0))


def one_flip_search(bits, score_fn, max_passes=None):
    """Greedy one-flip local search for minimization objectives."""
    import time

    start = time.perf_counter()
    current = np.asarray(bits, dtype=np.int8).copy()
    best_obj = float(score_fn(current))
    steps = 0
    passes = 0

    while max_passes is None or passes < max_passes:
        improved = False
        passes += 1
        for idx in range(current.size):
            current[idx] = 1 - current[idx]
            candidate_obj = float(score_fn(current))
            if candidate_obj < best_obj:
                best_obj = candidate_obj
                steps += 1
                improved = True
            else:
                current[idx] = 1 - current[idx]
        if not improved:
            break

    return RefinementResult(
        bits=current.astype(np.int32),
        objective=best_obj,
        steps=steps,
        seconds=time.perf_counter() - start,
    )


def incremental_qubo_one_flip(bits, data, max_passes=None, tolerance=1e-9):
    """Greedy sequential one-flip search with sparse incremental gains."""
    import time

    start = time.perf_counter()
    current = np.asarray(bits, dtype=np.int8).copy()
    q = data["Q_sparse"].astype(np.float64).tocsr()
    diagonal = np.asarray(q.diagonal(), dtype=np.float64)
    interaction = (q + q.transpose()).tocsc()
    interaction.setdiag(0.0)
    interaction.eliminate_zeros()
    linear = np.asarray(data["c"], dtype=np.float64) + diagonal
    coefficients = linear + interaction @ current.astype(np.float64)
    steps = 0
    passes = 0

    while max_passes is None or passes < max_passes:
        improved = False
        passes += 1
        for index in range(current.size):
            direction = 1.0 - 2.0 * float(current[index])
            gain = direction * coefficients[index]
            if gain >= -tolerance:
                continue
            current[index] = 1 - current[index]
            start_index = interaction.indptr[index]
            end_index = interaction.indptr[index + 1]
            neighbors = interaction.indices[start_index:end_index]
            coefficients[neighbors] += (
                interaction.data[start_index:end_index] * direction
            )
            steps += 1
            improved = True
        if not improved:
            break

    return RefinementResult(
        bits=current.astype(np.int32),
        objective=qubo_objective(current, data),
        steps=steps,
        seconds=time.perf_counter() - start,
    )


def refine_binary_incumbent(task, bits, data, max_passes=None):
    if task == "labs":
        from src.problem_parser import evaluate_LABS_bits

        n = data["num_x_vars"]
        initial = np.asarray(bits[:n], dtype=np.int8)
        return one_flip_search(initial, evaluate_LABS_bits, max_passes=max_passes)

    return incremental_qubo_one_flip(bits, data, max_passes=max_passes)
