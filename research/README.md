# PDBO research diagnostics

## Spectrum-scaled improvement study

Run paired seeds for fixed and spectrum-scaled initialization, with optional
trajectory conditional-expectation rounding and one-flip refinement:

```bash
conda run -n pdbo python research/pdbo_improvement_study.py \
  --gset-ids 1 11 14 22 70 --seeds 0 1 2 3 4 \
  --configured-duals 15 --burn-ins 0 250 500 \
  --steps 3000 --batch 10 --alpha 0.002 --beta 0.02 --rho 0.05 \
  --output research/results/improvement_study/initialization.csv
```

Use `--alpha-mode spectral --spectral-step-fraction 0.5` to scale both the
dual initialization and primal step. Add `--conditional-rounding` to retain
and derandomize the best multilinear expectation over the whole trajectory.
The script records raw/refined cuts, effective parameters, perturbation count,
and separate initialization, solve, and refinement times.

`trajectory_kernel_study.py` is the corresponding experimental Gram/SDP
kernel prototype. It is retained as a negative-result diagnostic: on the
tested Gset instances its valid Gaussian hyperplane rounding was weaker than
trajectory conditional-expectation rounding plus one-flip.

## Per-mode Duhamel experiment

Run the G3 filtering experiment with the parameters used in the spectral theory
report:

```bash
python research/g3_spectral_dynamics.py \
  --gset-id 3 --steps 3000 --sample-every 10 --batch 10 \
  --alpha 0.002 --beta 0.02 --dual-init 15 --rho 0.05 --seed 0 \
  --trajectory-index 0 --major-low-modes 8 --major-energy-modes 8 \
  --output research/results/g3_filtering --no-plot
```

The experiment selects the lowest eigenmodes together with the modes having the
largest peak energy on the tracked trajectory. It writes:

- `*-duhamel.csv`: sampled per-mode values of the actual coefficient,
  homogeneous prediction, net forcing injection, forcing radius `R`, relative
  remainder `epsilon`, and numerical diagnostics.
- `*-duhamel-summary.csv`: one final summary row per selected mode.
- `*-scalars.csv`: sampled global dynamics and archive cut history.
- `*.json`: run parameters and aggregate diagnostics.
- `*.npz`: the full arrays needed for follow-up analysis.

`net_forcing_injection` is evaluated stably as `actual - homogeneous`. The
separately accumulated Duhamel sum can become ill-conditioned when two very
large terms cancel; `identity_residual`, `relative_identity_residual`, and
`cancellation_condition` expose that numerical effect. The modal forcing used
for `R` is computed from consecutive implemented states. Its agreement with
the coordinate-space forcing is reported as `max_forcing_projection_discrepancy`.

Run the same diagnostic across several Gset instances and create a combined
comparison CSV:

```bash
python research/gset_duhamel_sweep.py \
  --gset-ids 1 2 3 4 5 \
  --output-dir research/results/gset_sweep
```

The sweep writes each instance's detailed CSV/JSON artifacts and
`gset-duhamel-sweep.csv`. Full NPZ arrays are disabled by default for sweeps;
pass `--save-npz` when they are needed. Use `--reuse-existing` to rebuild only
the combined summary from completed per-instance artifacts.

## Minimum-eigenvector warm start

To skip the initial contraction phase without constraining later iterates to a
fixed eigenspace, initialize only the primal state along `v1` and choose the
uniform dual so the largest raw coordinate reaches the box in the requested
number of homogeneous steps:

```bash
python research/gset_duhamel_sweep.py \
  --gset-ids 1 2 3 4 5 --batch 1 \
  --initialization v1 --v1-max-abs 0.49 \
  --v1-dual-mode clip-threshold --v1-clip-steps 1 \
  --output-dir research/results/gset_v1_clip_b1
```

After initialization, the full PDBO update runs without projection back onto
`v1`. `clip-threshold` generally starts outside the PSD cone; use
`--v1-dual-mode psd-boundary` to retain initial dual feasibility instead of
targeting immediate clipping.

For 10,000--20,000 vertex Gset instances, use the sparse-Lanczos comparison:

```bash
python research/large_gset_v1_comparison.py \
  --gset-ids 67 70 72 77 81 --rho 0.05 --psd-margin 0.1 \
  --output-dir research/results/large_gset_v1
```

It compares center-uniform and `v1` primal starts under the same
`y_init = -lambda_1 + 0.1`, plus the more aggressive one-step clipping start.
