# Wilson-Cowan Reservoir Ensemble Plan

## Scientific question

The experiment separates four possible sources of one-step EEG prediction
performance:

| Ablation | Reservoir | Isolated contribution |
| --- | --- | --- |
| A | `deterministic_poles` | fitted WC time scales |
| B | `distributed_poles` | high-dimensional diversity around those time scales |
| C | `independent_nonlinear_wc` | complete local nonlinear WC dynamics |
| D | `coupled_nonlinear_wc` | sparse cross-block recurrent dynamics |

A and B use only the local Jacobian spectrum. C and D integrate the nonlinear
vector field. This distinction is part of the scientific interpretation, not
only an implementation detail.

## Scope and invariants

- Use the two-population WC excitatory/inhibitory equations as the only neural
  mass model.
- Use the existing local EDF dataset and channel `Oz`; never download data.
- Split each recording chronologically before fitting detrending,
  normalization, resampling-sensitive processing, WC parameters, or readout
  parameters.
- Use one subject-local WC fit as the common center for all four modes in an
  ablation run.
- Keep all reservoir dynamics fixed. Train only the existing ridge readout.
- Evaluate the same teacher-forced one-step task and persistence baseline in
  every mode.
- Keep subjects, modes, configurations, seeds, fitted parameters, predictions,
  diagnostics, and metrics in isolated artifact directories.
- Preserve the original deterministic reservoir and the legacy mode aliases
  `deterministic` and `gaussian_eigenvalue_cloud`.
- Keep seconds as the canonical time unit. At 250 Hz the EEG sample interval is
  0.004 seconds; continuous-pole imaginary parts are in rad/s.

## Shared WC model

The uncoupled driven block is

```text
tau_E[k] dE[k]/dt = -E[k] + S(w_EE[k] E[k] - w_EI[k] I[k] + P[k] + B_E[k] u(t))
tau_I[k] dI[k]/dt = -I[k] + S(w_IE[k] E[k] - w_II[k] I[k] + Q[k] + B_I[k] u(t))
```

The coupled mode adds only the normalized excitatory graph current in its first
implementation:

```text
+ gamma sum_j A[k,j] E[j]
```

One canonical implementation of these equations must be shared by single-model
simulation, fitting, and the vectorized ensemble.

## Incremental implementation

### Phase 1 — Common interface without baseline drift

- Introduce canonical mode names and retain legacy aliases.
- Preserve the deterministic two-pole numerical path, input-weight seed, ridge
  readout, washout, warmup, and teacher-forced prediction alignment.
- Refactor the pipeline into shared preparation and per-mode execution.
- Verify preprocessing leakage tests and the original reservoir tests before
  adding nonlinear behavior.

Acceptance gate: existing deterministic tests and a fixed-seed regression pass.

### Phase 2 — Distributed linear poles

- Sample in continuous-time eigenvalue space around the fitted conjugate pair.
- Reject nonnegative real parts, preserve exact conjugates, and map with
  `mu = exp(lambda * sample_dt_s)`.
- Record both continuous and discrete stability diagnostics.

Acceptance gate: requested dimension, seed behavior, conjugacy, continuous
stability, and `abs(mu) < 1` are tested.

### Phase 3 — Independent nonlinear WC ensemble

- Sample `K` bounded parameter vectors around the fitted center.
- Interpret scalar jitter as a fractional scale; allow parameter-specific
  overrides. Use log-space perturbations for positive time constants and
  couplings and bounded additive perturbations for signed drives.
- Drive WC input currents with EEG; never equate EEG with `E` or `I`.
- Integrate all blocks with vectorized fixed-step RK4 and piecewise-constant EEG
  input. The step interval comes from the processed sampling rate and is divided
  by `rk4_substeps`.
- Run a short autonomous boundedness preflight and resample invalid blocks.

Acceptance gate: deterministic construction, valid samples, `2K` state
dimension, finite zero-input and driven trajectories, input sensitivity,
block-local parameter sensitivity, and RK4 reference agreement are tested.

### Phase 4 — Weak sparse coupling

- Generate a reproducible fixed graph with no self-edges and approximately the
  configured degree.
- Normalize each row and retain one global strength `gamma`.
- Keep graph randomness independent from parameter and input randomness.
- Bypass the graph term when strength is zero.

Acceptance gate: zero strength reproduces Phase 3, nonzero strength changes the
trajectory, graph sparsity/connectivity is correct, seeds reproduce the graph,
and normalized coupling remains comparable across reservoir sizes.

### Phase 5 — Optional hybrid fitting objective

- Retain PSD loss with weight 1.0 as the default.
- Add a training-only normalized log-STFT term with configurable window and
  overlap; keep its default weight at zero.
- Keep local temporal loss optional and default it to zero.
- Avoid dynamic time warping in the first implementation.

Acceptance gate: zero optional weights reproduce the PSD-only objective, test
data cannot affect any loss component, and enabled STFT loss is finite and
sensitive to local spectral changes.

### Phase 6 — Four-mode ablation

- Load, split, preprocess, and fit WC once for `sub-001`.
- Run A–D from the same prepared context and save each mode separately.
- Produce one compact comparison containing RMSE, MAE, Pearson correlation,
  persistence metrics, state dimension, block or mode count, seed, runtime,
  fitted parameters, effective configuration, and stability/boundedness data.
- Profile the 50- and 100-block nonlinear modes on a normal CPU before adding
  acceleration or dependencies.

Acceptance gate: all four modes complete on synthetic data; every saved summary
identifies the canonical architecture; mode directories do not overwrite one
another; only the readout changes during training.

## Validity rules

Linear and nonlinear validity must never be conflated:

```text
linear stability != nonlinear boundedness
```

- A and B require `Re(lambda) < 0`, hence `abs(mu) < 1`.
- C and D require finite RK4 trajectories within the configured population
  bounds. A nonnegative equilibrium eigenvalue is not itself a rejection reason,
  because a bounded limit cycle near a Hopf bifurcation is allowed.
- Numerical clipping must not be used to disguise a failed nonlinear preflight.

## Artifact contract

The intended layout is:

```text
artifacts/<subject>/<canonical-mode>/
```

Each mode directory is self-describing and includes the effective configuration,
shared WC fit, predictions, metrics, runtime, and mode-relevant diagnostics. The
machine-readable ablation summary and compact Markdown comparison live at the
subject level. Existing summary fields and artifact filenames remain available
where practical, with new structured fields added rather than silently replacing
old ones.

## Deferred work

- Multichannel and multi-region models.
- Automatic fitting across `sub-001` through `sub-010`.
- Trainable recurrent weights, unrelated neural architectures, or GPU
  frameworks.
- Bayesian calibration of the WC parameter population.
- Autonomous long-horizon prediction claims.

The ten-subject milestone remains deferred until the four `sub-001` modes and
their scientific diagnostics have been verified.
