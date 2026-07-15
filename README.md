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

Run all bundled small examples:

```bash
python examples/run_examples.py
```

Run PDBO on a random regular MIS instance:

```bash
python main.py --task mis --graph reg --n 1000 --d 3 --batch 10
```

Run PDBO on a Gset Max-Cut instance:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --batch 100 --dual_init 6
```

Run PDBO on a LABS instance:

```bash
python main.py --task labs --labs_n 47 --labs_penalty 10000 --batch 100
```

The default iteration budget is `max_iters=5000`.

The default primal initialization samples each coordinate uniformly around the
centre:

```text
x ~ Uniform(0.5 - rho, 0.5 + rho),  rho=0.1
```

Set the radius with `--rho`, or restore the original full-box initialization
with `--primal_init uniform`:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --rho 0.05
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

Animate the RMS projection length of `x - 0.5` along the eigenvectors of `W`:

```bash
pip install -e ".[visualization]"
python main.py --task mc --graph Gset --Gset_id 1 --spectral_animation
```

The chart uses 50 equal-width eigenvalue bins by default. Within each bin it
plots the mean mode length across eigenvectors, after taking the RMS across the
primal batch. It refreshes every 100 iterations; use
`--spectral_animation_every 10` for more frequent updates. The exact
eigenvectors required by the animation limit it to `n <= 2000`. The final
window remains open until it is closed; pass `--no-spectral_animation_hold` to
exit immediately after solving.

For manual stepping, start the animation in paused mode:

```bash
python main.py --task mc --graph Gset --Gset_id 1 \
  --spectral_animation --spectral_animation_manual
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
from pdbo import PDBOSolver, generate_mis, random_graph

graph = random_graph(n=1000, d=3, seed=0)
data = generate_mis(graph, penalty=4)

solver = PDBOSolver(
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
print(solver.ObjVal, solver.X)
print(result.objective, result.incumbent)
```

`PDQuboSolver` remains available as an alias for `PDBOSolver`.

## Options

Early stopping based on rounded incumbent stagnation:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --patience 1000 --check_every 10
```

Extra rounded candidates sampled from the final relaxed batch:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --rounding_samples 8
```

Restart once from the first phase's best rounded solution. The primal batch is
reset to that solution and the dual variables are restored to `dual_init`
before a second phase with the same `max_iters` budget:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --restart
```

Greedy one-flip local-search refinement:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --refine
```

Set the deterministic centre-perturbation tolerance from paper Algorithm 1:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --delta 1e-4
```

Experiment with the alternative integrality function
`g(x) = |x - 1/2| - 1/2`. The original `g(x) = x^2 - x` remains the default.
In the alternative mode, perturbation is applied only when a coordinate is
exactly `x = 1/2`:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --g_type absolute
```

Saved results for this mode use `result/pdbo_absolute/`, separate from the
default algorithm's results.

For quadratic objectives, `--quadratic_backend sparse` is the default. Use
`--quadratic_backend edge` to evaluate the QUBO directly from edge indices.

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
- `solver_cpu.py`: paper-aligned CPU PDBO implementation.
- `problem_parser.py`: problem parsers and QUBO builders.
- `main.py`: command-line entry point.
- `examples/`: small runnable examples for each supported problem type.
- `instance/`: minimal bundled example instances (`Gset/G1.txt` and one small 3-SAT CNF).
- `scripts/pdqubo/`: optional PDBO-only run scripts.

Baseline solvers, large benchmark collections, and paper-reproduction artifacts
are intentionally not included in this clean public package.
