#!/usr/bin/env python3
"""Standard-library checks for the PDBO theory report.

These small computations verify identities, enumerate tight examples, and
reproduce counterexamples.  They do not replace the proofs in the report and
require neither JAX nor a GPU nor third-party Python packages.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Optional


Vector = list[float]
Matrix = list[list[float]]


def mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in matrix]


def quadratic_value(matrix: Matrix, vector: Vector) -> float:
    product = mat_vec(matrix, vector)
    return sum(vector[i] * product[i] for i in range(len(vector)))


def maxcut_value(weights: Matrix, bits: list[int]) -> float:
    return sum(
        weights[i][j]
        for i in range(len(bits))
        for j in range(i + 1, len(bits))
        if bits[i] != bits[j]
    )


def maxcut_flip_gains(weights: Matrix, bits: list[int]) -> Vector:
    current = maxcut_value(weights, bits)
    gains = []
    for i in range(len(bits)):
        neighbor = bits.copy()
        neighbor[i] = 1 - neighbor[i]
        gains.append(maxcut_value(weights, neighbor) - current)
    return gains


def qubo_value(matrix: Matrix, linear: Vector, bits: Vector, constant: float = 0.0) -> float:
    return constant + quadratic_value(matrix, bits) + sum(a * x for a, x in zip(linear, bits))


def paper_pd_step(
    weights: Matrix,
    primal: Vector,
    dual: Vector,
    alpha: float,
    beta: float,
    delta: float,
) -> tuple[Vector, Vector]:
    """One simultaneous step of paper Algorithm 1 for Max-Cut.

    The dual update deliberately uses old primal, unlike solver_jax.py.
    """
    degree = [sum(row) for row in weights]
    wx = mat_vec(weights, primal)
    gradient = [
        2.0 * wx[i] - degree[i] + dual[i] * (2.0 * primal[i] - 1.0)
        for i in range(len(primal))
    ]
    next_primal = []
    for x_i, y_i, grad_i in zip(primal, dual, gradient):
        trigger = abs(x_i - 0.5) <= delta and abs(grad_i) <= 2.0 * delta and y_i <= 0.0
        if trigger:
            next_primal.append(0.5 - delta if x_i <= 0.5 else 0.5 + delta)
        else:
            next_primal.append(max(0.0, min(1.0, x_i - alpha * grad_i)))
    next_dual = [y_i + beta * (x_i * x_i - x_i) for x_i, y_i in zip(primal, dual)]
    return next_primal, next_dual


def check_flip_gradient_identity() -> None:
    rng = random.Random(20260711)
    for n in range(2, 8):
        for _ in range(100):
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    value = float(rng.randint(-4, 4))
                    matrix[i][j] = matrix[j][i] = value
            linear = [float(rng.randint(-4, 4)) for _ in range(n)]
            bits = [float(rng.randint(0, 1)) for _ in range(n)]
            qz = mat_vec(matrix, bits)
            gradient = [2.0 * qz[i] + linear[i] for i in range(n)]
            current = qubo_value(matrix, linear, bits)
            for i in range(n):
                neighbor = bits.copy()
                neighbor[i] = 1.0 - neighbor[i]
                direct = qubo_value(matrix, linear, neighbor) - current
                predicted = (1.0 - 2.0 * bits[i]) * gradient[i]
                assert math.isclose(direct, predicted, abs_tol=1e-10)


def enumerate_unweighted_local_optima() -> tuple[float, Matrix, list[int]]:
    worst_ratio = math.inf
    witness_weights: Optional[Matrix] = None
    witness_bits: Optional[list[int]] = None
    for n in range(2, 5):
        edges = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(edges)):
            weights = [[0.0] * n for _ in range(n)]
            for bit, (i, j) in enumerate(edges):
                if (mask >> bit) & 1:
                    weights[i][j] = weights[j][i] = 1.0
            cuts = [
                (maxcut_value(weights, list(bits)), list(bits))
                for bits in itertools.product((0, 1), repeat=n)
            ]
            optimum = max(value for value, _ in cuts)
            if optimum == 0.0:
                continue
            for value, bits in cuts:
                if max(maxcut_flip_gains(weights, bits)) <= 1e-12:
                    ratio = value / optimum
                    if ratio < worst_ratio:
                        worst_ratio = ratio
                        witness_weights = [row.copy() for row in weights]
                        witness_bits = bits.copy()
    assert witness_weights is not None and witness_bits is not None
    assert math.isclose(worst_ratio, 0.5)
    return worst_ratio, witness_weights, witness_bits


def check_c4_tight_examples() -> None:
    weights = [[0.0] * 4 for _ in range(4)]
    cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in cycle:
        weights[i][j] = weights[j][i] = 1.0
    bits = [0, 0, 1, 1]
    assert maxcut_value(weights, bits) == 2.0
    assert all(math.isclose(gain, 0.0) for gain in maxcut_flip_gains(weights, bits))
    assert max(maxcut_value(weights, list(z)) for z in itertools.product((0, 1), repeat=4)) == 4.0

    eta = 0.01
    strict = [[0.0] * 4 for _ in range(4)]
    for i, j in cycle:
        weight = 1.0 + eta if bits[i] != bits[j] else 1.0
        strict[i][j] = strict[j][i] = weight
    assert all(gain < 0.0 for gain in maxcut_flip_gains(strict, bits))
    strict_optimum = max(maxcut_value(strict, list(z)) for z in itertools.product((0, 1), repeat=4))
    strict_ratio = maxcut_value(strict, bits) / strict_optimum
    assert 0.5 < strict_ratio < 0.51


def check_signed_maxcut_counterexample(eta: float = 0.1) -> None:
    weights = [[0.0] * 4 for _ in range(4)]
    for i, j in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        weights[i][j] = weights[j][i] = 1.0
    weights[0][1] = weights[1][0] = -(2.0 + eta)
    weights[2][3] = weights[3][2] = -(2.0 + eta)
    bits = [0, 0, 0, 0]
    assert maxcut_value(weights, bits) == 0.0
    assert all(gain < 0.0 for gain in maxcut_flip_gains(weights, bits))
    optimum = max(maxcut_value(weights, list(z)) for z in itertools.product((0, 1), repeat=4))
    assert math.isclose(optimum, 4.0)


def check_qubo_local_gap(M: float = 100.0) -> None:
    matrix = [[0.0, -(M + 1.0) / 2.0], [-(M + 1.0) / 2.0, 0.0]]
    linear = [1.0, 1.0]
    values = {
        bits: qubo_value(matrix, linear, [float(x) for x in bits], constant=M)
        for bits in itertools.product((0, 1), repeat=2)
    }
    assert values[(0, 0)] == M
    assert values[(1, 0)] == values[(0, 1)] == M + 1.0
    assert values[(1, 1)] == 1.0


def check_binary_nonconvergence() -> None:
    alpha = 0.1
    dual = 11.0
    primal = 0.0
    trajectory = []
    for _ in range(6):
        trajectory.append(primal)
        gradient = dual * (2.0 * primal - 1.0)
        next_primal = max(0.0, min(1.0, primal - alpha * gradient))
        dual += 0.2 * (primal * primal - primal)
        primal = next_primal
    assert trajectory == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def check_convex_initialization_bad_trajectory() -> int:
    weights = [[0.0, 1.0], [1.0, 0.0]]
    primal = [0.5, 0.5]
    dual = [1.1, 1.1]
    alpha, beta, delta = 0.1, 0.4, 0.1
    assert dual[0] > 1.0  # W + diag(y0) is positive definite on K2.
    hit = None
    rounded_values = []
    for iteration in range(500):
        rounded = [int(x >= 0.5) for x in primal]
        rounded_values.append(maxcut_value(weights, rounded))
        primal, dual = paper_pd_step(weights, primal, dual, alpha, beta, delta)
        if all(math.isclose(x, 0.0, abs_tol=1e-12) for x in primal):
            hit = iteration + 1
            break
    assert hit is not None
    assert all(math.isclose(value, 0.0) for value in rounded_values)
    for _ in range(10):
        primal, dual = paper_pd_step(weights, primal, dual, alpha, beta, delta)
        assert all(math.isclose(x, 0.0, abs_tol=1e-12) for x in primal)
    assert maxcut_value(weights, [0, 0]) == 0.0
    assert maxcut_value(weights, [0, 1]) == 1.0
    return hit


def check_g1_parameter_certificates() -> tuple[float, float]:
    path = Path(__file__).resolve().parents[1] / "instance" / "Gset" / "G1.txt"
    lines = path.read_text().splitlines()
    n, m = map(int, lines[0].split())
    assert (n, m) == (800, 19176)
    assert all(int(line.split()[2]) == 1 for line in lines[1:])

    average_degree = 2.0 * m / n
    assert math.isclose(average_degree, 47.94)
    # Rayleigh with the all-ones vector: lambda_max >= average degree.
    assert 0.02 * (average_degree + 5.0) > 1.0

    # The user's reported G1 cut value itself gives a min-eigenvalue certificate.
    reported_cut = 11624
    signed_rayleigh = 2.0 * (m - 2.0 * reported_cut) / n
    assert signed_rayleigh < -5.0
    return average_degree, signed_rayleigh


def main() -> None:
    check_flip_gradient_identity()
    ratio, weights, bits = enumerate_unweighted_local_optima()
    check_c4_tight_examples()
    check_signed_maxcut_counterexample()
    check_qubo_local_gap()
    check_binary_nonconvergence()
    hit = check_convex_initialization_bad_trajectory()
    average_degree, signed_rayleigh = check_g1_parameter_certificates()
    edge_count = int(sum(sum(row) for row in weights) / 2.0)
    print("all checks passed")
    print(f"worst unweighted n<=4 one-flip ratio: {ratio:.3f}")
    print(f"enumerated witness bits: {''.join(map(str, bits))}")
    print(f"enumerated witness edges: {edge_count}")
    print(f"K2 convex-init bad trajectory reached 00 at iteration: {hit}")
    print(f"G1 average-degree lower bound on lambda_max: {average_degree:.2f}")
    print(f"G1 reported-cut Rayleigh upper bound on lambda_min: {signed_rayleigh:.2f}")


if __name__ == "__main__":
    main()
