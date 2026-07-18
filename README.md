# PDBO

PDBO is a CPU implementation of the original primal-dual algorithm from
**Smoothing Binary Optimization: A Primal-Dual Perspective**.

The solver follows paper Algorithm 1 directly: fixed-step projected gradient
descent, a simultaneous dual update using the old primal iterate, and the
paper's deterministic centre perturbation. It does not use RMSProp/Adam,
optimizer momentum, or the previous Gauss-Seidel-style new-primal dual update.


## Installation

```bash
pip install -e .
```

To run smoke tests:

```bash
pip install -e ".[test]"
pytest
```

## Quick Start

完整的中文命令行参数说明、适用范围、调参建议和保存规则见
[`docs/PDBO_solver_parameters_zh.md`](docs/PDBO_solver_parameters_zh.md)。

Run all bundled small examples:

```bash
python examples/run_examples.py
```

Run PDBO on a random regular MIS instance:

```bash
python src/main.py --task mis --graph reg --n 1000 --d 3 --batch 10
```

Run PDBO on a Gset Max-Cut instance:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --batch 100 --dual_init 6
```

For a spectrum-scaled Max-Cut run, initialize at the PSD frontier and choose
the primal step so that the initial highest-mode multiplier remains positive:

```bash
python src/main.py --task mc --graph Gset --Gset_id 70 --batch 10 \
  --max_iters 3000 --lr_y 0.02 --rho 0.05 \
  --dual_init_mode spectral --dual_burn_in 0 \
  --primal_lr_mode spectral --spectral_step_fraction 0.5 \
  --conditional_rounding --refine
```

The automatic parameters are

```text
dual_init = -lambda_min(W) + dual_psd_margin + dual_lr * dual_burn_in / 4
primal_lr = spectral_step_fraction / (2 * (lambda_max(W) + dual_init))
```

Thus `dual_burn_in` measures the idealized number of centre steps before the
PSD boundary, while `0 < spectral_step_fraction < 1` guarantees a positive
initial multiplier for every eigenmode. `conditional_rounding` retains the
best multilinear/Bernoulli expectation seen over the full trajectory and
derandomizes it at the end. The final one-flip refinement is sparse and
incremental.

Run PDBO on a LABS instance:

```bash
python src/main.py --task labs --labs_n 47 --labs_penalty 10000 --batch 100
```

The default iteration budget is `max_iters=5000`.

The default primal initialization samples each coordinate uniformly around the
centre:

```text
x ~ Uniform(0.5 - rho, 0.5 + rho),  rho=0.05
```

Set the radius with `--rho`, or restore the original full-box initialization
with `--primal_init uniform`:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --rho 0.05
```

Before solving, verbose mode prints a histogram of the full spectrum of
`W = (Q + Q.T) / 2`, followed by the spectral selection window for each phase.
The histogram is exact up to 2000 variables and uses stochastic Lanczos
quadrature (`approx-slq`) for larger sparse problems:

```text
spectrum_distribution=exact n=800 bins=20 range=[...]
  [...] count=... pct=... ####
round=1/2 spectral_window=(-lambda_(r+1), -lambda_1) lambda_1=... lambda_(r+1)=... r=...
```

Animate the RMS projection length of `x - 0.5` along the eigenvectors of `W`,
alongside histograms of the current dual variables `y` and centered primal
variables `z = x - 0.5`, plus the batch-mean values of `L(x, y)` and `f(x)`:

```bash
pip install -e ".[visualization]"
python src/main.py --task mc --graph Gset --Gset_id 1 --spectral_animation
```

The chart uses 50 equal-width bins by default. Within each eigenvalue bin it
plots the mean mode length across eigenvectors, after taking the RMS across the
primal batch. Two panels show the value counts for `y` and `z`; the fourth
plots the batch-mean `L(x, y)` and `f(x)` against the global iteration `t`.
Graphs with `n <= 2000` use the exact eigendecomposition. Larger
graphs use 128 Lanczos-Ritz modes across the spectrum by default; set
`--spectral_animation_modes` to trade initialization and refresh cost against
spectral resolution. Continuous execution refreshes every 100 iterations by
default; set `--spectral_animation_every` to change the interval. All four
panels refresh together, so the objective curves use the same sampled
iterations; manual stepping still refreshes every step. The final window
remains open until it is closed.

For example, G64 can be animated without constructing a dense `7000 x 7000`
eigenvector matrix:

```bash
python src/main.py --task mc --graph Gset --Gset_id 64 \
  --spectral_animation --spectral_animation_modes 128
```

For manual stepping, start the animation in paused mode:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 \
  --spectral_animation
```

`Next step` advances exactly one solver iteration. `Run` switches to continuous
execution and becomes `Pause` while running. Time spent waiting for manual input
is excluded from the solver runtime and `timelimit` accounting.

## Python API

For quadratic objectives, PDBO solves

```text
minimize x^T Q x + c^T x,  x in {0, 1}^n.
```

```python
from src.problem_parser import generate_MIS, random_graph
from src.solver import PDBO_CPU

graph = random_graph(n=1000, d=3, seed=0)
data = generate_MIS(graph, penalty=4)

solver = PDBO_CPU(
    n_vars=data["num_vars"],
    Q_indices=data["Q_indices"],
    Q_values=data["Q_values"],
    c=data["c"],
    batch_size=10,
    primal_lr=0.02,
    dual_lr=0.02,
    tolerance=1e-8,  # Algorithm 1 tolerance delta
    dual_init=5,
    rounding_samples=8,
    seed=0,
)
result = solver.optimize()
print(result.objective, result.incumbent)
```

## Options

Early stopping based on rounded incumbent stagnation:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --patience 1000 --check_every 10
```

Extra rounded candidates sampled from the final relaxed batch:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --rounding_samples 8
```

Restart once from the first phase's best rounded solution. The primal batch is
reset to that solution and the dual variables are restored to `dual_init`
before a second phase with the same `max_iters` budget:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --restart
```

Greedy one-flip local-search refinement:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --refine
```

Set the deterministic centre-perturbation tolerance from paper Algorithm 1:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --delta 1e-4
```

Experiment with the alternative integrality function
`g(x) = |x - 1/2| - 1/2`. The original `g(x) = x^2 - x` remains the default.
In the alternative mode, perturbation is applied only when a coordinate is
exactly `x = 1/2`:

```bash
python src/main.py --task mc --graph Gset --Gset_id 1 --g_type absolute
```

Saved results for this mode use `result/pdbo_absolute/`, separate from the
default algorithm's results.

## Citation

If you use this code, please cite:

```bibtex
@misc{liu2026smoothingbinaryoptimizationprimaldual,
      title={Smoothing Binary Optimization: A Primal-Dual Perspective}, 
      author={Wenbo Liu and Akang Wang and Dun Ma and Hongyi Jiang and Jianghua Wu and Wenguo Yang},
      year={2026},
      eprint={2509.21064},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2509.21064}, 
}
```

## Repository Layout

- `pdbo/`: public package API.
- `src/solver.py`: paper-aligned CPU PDBO implementation.
- `src/spectral.py`: sparse eigensolver and spectrum-distribution utilities.
- `src/spectral_animation.py`: interactive spectral-mode visualization.
- `src/problem_parser.py`: problem parsers and QUBO builders.
- `src/main.py`: command-line entry point.
- `examples/`: small runnable examples for each supported problem type.
- `instance/`: minimal bundled example instances (`Gset/G1.txt` and one small 3-SAT CNF).
- `scripts/pdqubo/`: optional PDBO-only run scripts.

Baseline solvers, large benchmark collections, and paper-reproduction artifacts
are intentionally not included in this clean public package.
