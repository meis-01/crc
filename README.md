# Wilson-Cowan Reservoir Ensemble for EEG

This repository is a configuration-driven experiment framework for asking which
parts of a Wilson-Cowan (WC) prior help one-step EEG prediction. It compares four
progressively richer reservoirs while keeping the chronological split, fixed
physics dynamics, ridge readout, and evaluation protocol the same:

1. deterministic WC poles;
2. distributed WC poles;
3. independent nonlinear WC blocks;
4. a weakly coupled nonlinear WC network.

Only the readout is trained. The fitted WC parameters, sampled reservoir
parameters, input projections, and coupling graph remain fixed during readout
training.

## Reservoir modes

### `deterministic_poles`

The original linearized physics reservoir is preserved. For a fitted WC
equilibrium $x^*$, its Jacobian supplies two continuous-time eigenvalues
$\lambda_k$. They are mapped to discrete poles at the EEG sample interval
$\Delta t$:

$$
\mu_k = \exp(\lambda_k\Delta t), \qquad
z_{t+1} = \mathrm{diag}(\mu_k)z_t + W_{\mathrm{in}}u_t.
$$

This mode represents local time scales near one equilibrium; it is not a
simulation of the nonlinear WC vector field. The legacy mode name
`deterministic` remains accepted.

### `distributed_poles`

This linear mode samples a configurable population around the fitted continuous
poles before discretization:

$$
\widetilde{\lambda}_j = (\mathrm{Re}\,\lambda_k+\epsilon_r) + i(\mathrm{Im}\,\lambda_k+\epsilon_i),
$$
$$

$$
\epsilon_r\sim\mathcal N(0,\sigma_r^2), \qquad
\epsilon_i\sim\mathcal N(0,\sigma_i^2), \qquad
\widetilde\mu_j=\exp(\widetilde\lambda_j\Delta t).
$$

Sampling occurs in continuous-time eigenvalue space, conjugate pairs are kept
exactly, and samples must satisfy
$\Re\widetilde\lambda_j<0$, equivalently
$|\widetilde\mu_j|<1$. The previous name
`gaussian_eigenvalue_cloud` remains accepted as a compatibility alias.

### `independent_nonlinear_wc`

This mode retains the complete WC vector field in $K$ independent blocks. The
z-scored EEG value $u_t$ drives each block as an input current; it is not
identified with either population state:

$$
\tau_E^{(k)}\dot E_k = -E_k + S\!\left(
w_{EE}^{(k)}E_k-w_{EI}^{(k)}I_k+P_k+B_{E,k}u_t\right),
$$

$$
\tau_I^{(k)}\dot I_k = -I_k + S\!\left(
w_{IE}^{(k)}E_k-w_{II}^{(k)}I_k+Q_k+B_{I,k}u_t\right).
$$

The readout feature vector is
$h_t=[E_1,I_1,\ldots,E_K,I_K]$, so its state dimension is $2K$.
The EEG input is piecewise constant during each sample interval, and a vectorized
fixed-step RK4 integrator uses the configured number of substeps.

### `coupled_nonlinear_wc`

This is a separate ablation from the independent ensemble. A fixed sparse graph
adds normalized excitatory cross-block recurrence:

$$
\tau_E^{(k)}\dot E_k = -E_k + S\!\left(
w_{EE}^{(k)}E_k-w_{EI}^{(k)}I_k+P_k+B_{E,k}u_t
+\gamma\sum_j A_{kj}E_j\right).
$$

The inhibitory equation remains local. The graph has no self-edges, is generated
from its own seed, and is row-normalized so changing $K$ does not multiply the
coupling magnitude. The first implementation is a directed fixed-in-degree graph:
`A[k, j]` carries source block `j` into target block `k`. Setting
`coupling.strength: 0.0` reproduces the independent nonlinear mode exactly for
the same reservoir seed. `coupling.enabled: false` is an equivalent diagnostic
bypass inside this mode; selecting `independent_nonlinear_wc` remains the
explicit uncoupled architecture.

## WC fitting and preprocessing

The raw recording is split chronologically before any learned or global
preprocessing statistics are calculated. Linear detrending and z-score
statistics are fitted on the training partition and reused for the test
partition. Train and test partitions are resampled independently, so the
polyphase filter cannot access samples across the split.

The default WC objective remains the original training-only PSD shape loss. An
optional hybrid objective is configured as

$$
L=\alpha L_{\mathrm{PSD}}+\beta L_{\mathrm{STFT}}+\eta L_{\mathrm{temporal}}.
$$

`fit.loss.stft_weight: 0.0` and `temporal_weight: 0.0` preserve PSD-only fitting.
When enabled, the STFT term uses training data only, one-second windows, 50%
overlap, and normalized log magnitude by default. It compares local spectral
structure without requiring agreement in global Fourier phase.

## Parameter population and validity

The fitted parameter vector is the center of the nonlinear ensemble. The scalar
`parameter_jitter` is a fractional scale rather than one shared absolute
Gaussian standard deviation. A mapping keyed by WC parameter can override it,
for example:

```yaml
nonlinear_reservoir:
  parameter_jitter:
    tau_e: 0.03
    tau_i: 0.05
    w_ee: 0.08
    p: 0.02
```

Positive time constants and positive coupling parameters are perturbed
multiplicatively in log space. Signed drives `p` and `q` use bounded additive
perturbations scaled to their configured range. A nonnegative coupling centered
exactly at zero cannot be sampled in log space, so it uses a one-sided
half-normal perturbation scaled to its allowed range. A mapping may include a
`default` key for unspecified parameters; without it, unspecified entries have
zero jitter. Samples are constrained to the same physiological/numerical bounds
as fitting; sigmoid gain and threshold remain fixed unless explicitly supported
by a later experiment.

The fixed input-current coefficients $B_E$ and $B_I$ are independent
zero-mean Gaussian draws scaled by `input_scale`. Parameter draws and input
draws use separate deterministic random streams, and the coupling graph has its
own seed, so changing graph construction cannot silently change the local WC
population.

The validity criteria deliberately differ by architecture:

- Linear modes require continuous stability
  $\Re(\lambda)<0$ and discrete stability $|\mu|<1$.
- Nonlinear modes require a finite, bounded RK4 preflight trajectory inside the
  configured state bounds. They are not rejected merely because an equilibrium
  has a nonnegative eigenvalue real part, since bounded oscillations around a
  Hopf bifurcation are scientifically valid.

In short, **linear stability is not nonlinear boundedness**.

The nonlinear preflight is autonomous (zero EEG current) and never clips a
trajectory back into range. During driven reservoir execution, every RK4
substep is also checked for finite values and the configured state bounds.

## Units

- EEG sampling rates are in hertz and all time values (`tau_e`, `tau_i`, `dt`,
  durations, and STFT windows) are in seconds.
- The default resampled rate is 250 Hz, giving $\Delta t=0.004$ seconds.
- Continuous-pole real parts and `eigenvalue_sigma_real` are in s$^{-1}$.
- Continuous-pole imaginary parts and `eigenvalue_sigma_imag` are angular
  frequencies in rad/s. Discrete poles $\mu$ are dimensionless.
- $E$ and $I$ are dimensionless activity fractions with a theoretical range
  of $[0,1]$. A small configurable numerical tolerance is used in preflight.
- The preprocessed EEG input is dimensionless after training-fitted z-scoring;
  `input_scale` maps it into the WC input-current or linear reservoir scale.

The nonlinear reservoir always advances over the processed EEG sample interval;
`rk4_substeps` divides that interval. The configured fitting `dt` is separately
recorded and should match the sample interval when comparing fitted poles with
sampled reservoir dynamics.

## Runtime considerations

The nonlinear equations are vectorized over blocks. Coupling uses SciPy CSR
sparse matrix-vector products, so each derivative evaluation scales with the
number of graph edges (approximately $K$ times `degree`) rather than $K^2$.
On the development CPU, a 60,000-sample synthetic run took about 3.2–3.5 seconds
for 50–100 independent blocks and 4.0–4.2 seconds for degree-4 coupled blocks
with one RK4 substep. These figures are implementation checks, not portable
benchmarks.

## Evaluation and artifacts

Evaluation is chronological, teacher-forced one-step prediction. To predict
sample $u_{t+1}$, the reservoir is driven by observed $u_t$. This is not an
autonomous multistep rollout. The same washout, training partition, held-out test
partition, ridge readout, and persistence predictor $\widehat u_{t+1}=u_t$ are
used in every mode.

Each summary records RMSE, MAE, Pearson correlation, persistence performance,
runtime, seed, fitted WC parameters, effective configuration, readout state
dimension, WC block count, and stability or boundedness diagnostics. Linear
summaries distinguish the number of complex modes from the real-valued feature
dimension used by the readout.

Mode outputs are isolated so one architecture cannot overwrite another:

```text
artifacts/
  sub-001/
    deterministic_poles/
    distributed_poles/
    independent_nonlinear_wc/
    coupled_nonlinear_wc/
    ablation_summary.json
    ablation_report.md
```

Each mode directory contains its summary/configuration and the corresponding
PSD, WC fit, simulation, prediction, and exact fixed reservoir-dynamics
artifacts. The latter stores sampled poles or WC parameter arrays, input
projections, and the CSR graph realization needed to audit a seeded run.

## Running experiments

The local EDF dataset must already exist; no download logic is included. Set
`data_root` in `config/defaults.yaml`. The default experiment remains numerically
equivalent to the previous 100-mode Gaussian eigenvalue cloud, now named
`distributed_poles`.

Run the configured default:

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml
```

Run each architecture explicitly:

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode deterministic_poles
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode distributed_poles
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode independent_nonlinear_wc
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode coupled_nonlinear_wc
```

Run the four-mode ablation while reusing one preprocessing result and WC fit:

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --all-modes
```

Use a separate artifact root if desired:

```powershell
python scripts/run_sub001_pipeline.py --all-modes --output-dir artifacts/experiment-01
```

Run the test suite with:

```powershell
python -m pytest
```

## Current limitations

- The validated workflow targets one `Oz` channel from `sub-001`; it is not yet
  a multichannel, multi-region, or cross-subject model.
- Results are one-step teacher-forced predictions, not evidence of stable
  autonomous long-horizon forecasting.
- The WC fit is low-budget by default (`maxiter: 3`) for a practical proof of
  concept. The unforced fitted trajectory may emphasize transient spectral
  structure, so PSD/STFT agreement should not be interpreted as parameter
  identifiability.
- Nonlinear parameter populations are bounded perturbations around one fitted
  subject-level center, not posterior samples from a calibrated physiological
  inference procedure.
- The linear modes encode only local Jacobian dynamics. They must not be
  described as complete nonlinear physics-informed reservoirs.
