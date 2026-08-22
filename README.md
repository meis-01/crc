# Wilson-Cowan Reservoir Ensemble for EEG

A configuration-driven research framework for testing which parts of a Wilson-Cowan (WC) neural-population prior are useful for one-step EEG prediction.

The repository compares four progressively richer fixed reservoirs:

1. `deterministic_poles` — fitted local WC time scales;
2. `distributed_poles` — a heterogeneous linear population around those time scales;
3. `independent_nonlinear_wc` — independent nonlinear Wilson-Cowan blocks;
4. `coupled_nonlinear_wc` — a weakly coupled nonlinear Wilson-Cowan network.

Only the **linear ridge readout is trained**. The fitted WC parameters, sampled reservoir parameters, input projections, and coupling graph remain fixed during readout training.

The main scientific objective is therefore not simply to maximize prediction accuracy, but to determine **which dynamical ingredients add useful predictive representation**.

---

## Scientific question

The four modes form an incremental ablation:

| Mode | WC-derived time scales | Heterogeneity | Full WC nonlinearity | Cross-block recurrence |
| --- | :---: | :---: | :---: | :---: |
| `deterministic_poles` | ✓ | — | — | — |
| `distributed_poles` | ✓ | ✓ | — | — |
| `independent_nonlinear_wc` | ✓ | ✓ | ✓ | — |
| `coupled_nonlinear_wc` | ✓ | ✓ | ✓ | ✓ |

This isolates three main transitions:

1. **Deterministic → distributed:** does diversity of decay rates and oscillation frequencies help?
2. **Distributed → independent nonlinear:** does the complete nonlinear WC vector field add useful information beyond a rich linear filter bank?
3. **Independent → coupled nonlinear:** do recurrent interactions between nonlinear WC populations add further predictive structure?

The experiment is deliberately controlled: all modes use the same chronological data split, the same subject-local WC fit, the same teacher-forced one-step prediction task, the same ridge readout family, and the same evaluation protocol.

---

## Current scope

The validated workflow currently targets:

- dataset already available locally in EDF/BIDS form;
- subject `sub-001`;
- EEG channel `Oz`;
- task `resteyesc`;
- default resampling to 250 Hz;
- chronological train/test evaluation;
- one-step-ahead EEG prediction;
- fixed reservoir dynamics;
- ridge regression as the only trained component.

The current work is a proof of concept for controlled reservoir ablation. It is not yet a multichannel, cross-subject, or autonomous long-horizon forecasting system.

---

## Wilson-Cowan model

The nonlinear reservoirs use the two-population Wilson-Cowan model with excitatory activity $E$ and inhibitory activity $I$.

For one driven block,

$$
\tau_E \dot E = -E + S\left(w_{EE}E-w_{EI}I+P+B_Eu(t)\right),
$$

$$
\tau_I \dot I = -I + S\left(w_{IE}E-w_{II}I+Q+B_Iu(t)\right).
$$

The sigmoid used by the implementation is

$$
S(x)=\frac{1}{1+\exp\left[-g(x-\theta)\right]}.
$$

Here:

- $\tau_E,\tau_I$ are excitatory and inhibitory time constants;
- $w_{EE},w_{EI},w_{IE},w_{II}$ are within-block coupling coefficients;
- $P,Q$ are constant drives;
- $g$ is sigmoid gain;
- $\theta$ is sigmoid threshold;
- $u(t)$ is the preprocessed EEG input;
- $B_E,B_I$ map that input into excitatory and inhibitory currents.

The EEG observation is an **external driving signal**. It is not identified with either $E$ or $I$.

The theoretical WC population activities lie in $[0,1]$. The nonlinear reservoir implementation uses configurable numerical bounds during validity checks.

---

# Reservoir modes

## 1. `deterministic_poles`

This is the most constrained reservoir and serves as the linear physics-informed baseline.

After fitting the WC parameters, the implementation finds an equilibrium

$$
x^*=[E^*,I^*]^T
$$

and evaluates the Jacobian of the WC vector field at that point:

$$
J=\left.\frac{\partial F}{\partial x}\right|_{x=x^*}.
$$

The two continuous-time eigenvalues

$$
\lambda_k \in \mathrm{eig}(J)
$$

describe the local behavior of small perturbations around that equilibrium.

If an eigenvalue is written as

$$
\lambda_k=a_k+i b_k,
$$

then:

- $a_k<0$ determines the local decay rate;
- $b_k$ determines the local angular oscillation frequency.

The continuous-time eigenvalues are mapped to discrete-time poles at the processed EEG sample interval $\Delta t$:

$$
\mu_k=\exp(\lambda_k\Delta t).
$$

The reservoir then evolves as

$$
z_{t+1}=\mathrm{diag}(\mu_k)z_t+W_{\mathrm{in}}u_t.
$$

The resulting complex reservoir states are converted to real-valued readout features by concatenating their real and imaginary parts.

### Interpretation

This reservoir behaves like a small bank of linear dynamical filters whose memory and oscillation characteristics are inherited from the fitted WC equilibrium.

It asks:

> **Are the local time scales implied by the fitted Wilson-Cowan model already useful for predicting EEG?**

This mode does **not** simulate the complete nonlinear WC vector field. It represents only the local linearization around one equilibrium.

The legacy mode name `deterministic` remains accepted.

---

## 2. `distributed_poles`

The deterministic model contains only the small set of modes supplied by one fitted Jacobian. `distributed_poles` keeps the model linear but expands those modes into a configurable heterogeneous population.

For a fitted eigenvalue $\lambda_k$, nearby continuous-time modes are sampled as

$$
\widetilde{\lambda}_j=(\mathrm{Re}\,\lambda_k+\epsilon_r)+i(\mathrm{Im}\,\lambda_k+\epsilon_i),
$$

with

$$
\epsilon_r\sim\mathcal{N}(0,\sigma_r^2),
$$

$$
\epsilon_i\sim\mathcal{N}(0,\sigma_i^2).
$$

Each sampled continuous-time mode is then discretized:

$$
\widetilde{\mu}_j=\exp(\widetilde{\lambda}_j\Delta t).
$$

Sampling is performed in **continuous-time eigenvalue space**, rather than perturbing the discrete poles directly. This keeps the perturbations interpretable in terms of decay rates and angular frequencies.

Every sampled mode must satisfy

$$
\mathrm{Re}\,\widetilde{\lambda}_j<0,
$$

which implies

$$
|\widetilde{\mu}_j|<1.
$$

When oscillatory modes are present, conjugate pairs are preserved exactly. This gives the reservoir a real-consistent representation of oscillatory dynamics.

### Why this adds capacity

The deterministic model may expose only one or two characteristic time scales. The distributed model creates many nearby modes:

- some decay faster;
- some decay more slowly;
- some oscillate at slightly higher frequencies;
- some oscillate at slightly lower frequencies.

The ridge readout can combine these fixed responses to form a richer temporal basis.

### Interpretation

This mode asks:

> **Is a high-dimensional population of nearby WC-inspired time scales more useful than the small deterministic set from one fitted equilibrium?**

It is still a **linear reservoir**. Therefore, improvement over `deterministic_poles` should be interpreted as a benefit of **dynamical diversity and state dimension**, not as evidence for nonlinear computation.

The legacy name `gaussian_eigenvalue_cloud` remains accepted as a compatibility alias.

---

## 3. `independent_nonlinear_wc`

This mode makes the main transition from local linear dynamics to the complete nonlinear Wilson-Cowan vector field.

The reservoir consists of $K$ independent WC blocks:

$$
\tau_E^{(k)}\dot E_k=-E_k+S\left(w_{EE}^{(k)}E_k-w_{EI}^{(k)}I_k+P_k+B_{E,k}u_t\right),
$$

$$
\tau_I^{(k)}\dot I_k=-I_k+S\left(w_{IE}^{(k)}E_k-w_{II}^{(k)}I_k+Q_k+B_{I,k}u_t\right).
$$

Each block contains two dynamical states:

$$
(E_k,I_k).
$$

The readout state is

$$
h_t=[E_1,I_1,E_2,I_2,\ldots,E_K,I_K],
$$

so the real-valued reservoir state dimension is

$$
\mathrm{dim}(h_t)=2K.
$$

### Heterogeneous parameter population

The fitted WC parameter vector acts as the center of the nonlinear population.

Each block receives a fixed bounded perturbation of selected WC parameters. Therefore, the ensemble is not simply $K$ identical copies of the same neural-mass system.

The parameter population allows slightly different:

- excitatory and inhibitory time constants;
- recurrent coupling strengths;
- cross-population coupling strengths;
- constant excitatory and inhibitory drives.

The sigmoid gain and threshold remain fixed unless a later experiment explicitly supports perturbing them.

### Input projections

Each block also receives fixed input-current coefficients

$$
B_{E,k},B_{I,k}.
$$

These are independent zero-mean Gaussian draws scaled by `input_scale`.

As a result, different WC blocks can respond differently to the same EEG sample even when their local parameter sets are similar.

### Numerical integration

The EEG input is treated as piecewise constant over each processed sample interval.

The nonlinear equations are integrated with a vectorized fixed-step fourth-order Runge-Kutta method. If the processed EEG interval is $\Delta t$ and `rk4_substeps = m`, each RK4 substep uses approximately

$$
\frac{\Delta t}{m}.
$$

### What nonlinearity adds

Unlike the pole reservoirs, the response of a nonlinear WC block depends on its current state. The reservoir can therefore express effects such as:

- state-dependent gain;
- nonlinear saturation;
- nonlinear interaction between excitation and inhibition;
- amplitude-dependent responses;
- nonlinear oscillatory dynamics;
- trajectories that cannot be represented by a fixed linear filter bank.

The $K$ blocks remain mutually uncoupled. Their states are combined only by the final ridge readout.

### Interpretation

This mode asks:

> **Does the complete nonlinear Wilson-Cowan vector field provide useful predictive structure beyond a heterogeneous linear population of WC-derived time scales?**

The comparison

`distributed_poles` → `independent_nonlinear_wc`

is therefore the key ablation for studying the contribution of **nonlinear local dynamics**.

---

## 4. `coupled_nonlinear_wc`

The final reservoir extends the independent nonlinear ensemble by allowing recurrent interactions between WC blocks.

The excitatory equation becomes

$$
\tau_E^{(k)}\dot E_k=-E_k+S\left(w_{EE}^{(k)}E_k-w_{EI}^{(k)}I_k+P_k+B_{E,k}u_t+\gamma\sum_j A_{kj}E_j\right).
$$

The inhibitory equation remains local:

$$
\tau_I^{(k)}\dot I_k=-I_k+S\left(w_{IE}^{(k)}E_k-w_{II}^{(k)}I_k+Q_k+B_{I,k}u_t\right).
$$

The coupling term is

$$
\gamma\sum_j A_{kj}E_j.
$$

Here:

- $A_{kj}$ represents a directed connection from source block $j$ to target block $k$;
- $\gamma$ is the global coupling strength.

### Graph construction

The current implementation uses a fixed sparse directed graph with:

- no self-edges;
- configurable in-degree;
- deterministic generation from its own random seed;
- row normalization.

Row normalization prevents the effective coupling magnitude from automatically increasing when the reservoir size or graph degree changes.

Graph randomness is kept separate from parameter-population randomness and input-projection randomness. This makes seeded ablations easier to audit.

### Relation to the independent mode

With the same reservoir seed,

$$
\gamma=0
$$

removes cross-block recurrence and reproduces the independent nonlinear trajectory.

Therefore,

`independent_nonlinear_wc` → `coupled_nonlinear_wc`

isolates the contribution of **network interaction and collective recurrence** rather than changing the local WC equations.

### Interpretation

This mode asks:

> **Do weak recurrent interactions between heterogeneous nonlinear WC populations create useful temporal representations beyond independent nonlinear blocks?**

---

## Reading the four-mode ablation

The intended interpretation is:

```text
deterministic WC poles
        |
        | add diversity of time scales
        v
distributed WC poles
        |
        | add full nonlinear WC dynamics
        v
independent nonlinear WC blocks
        |
        | add sparse recurrent interaction
        v
coupled nonlinear WC network
```

A gain at each transition has a different scientific meaning:

| Comparison | Main added ingredient | Suggested interpretation |
| --- | --- | --- |
| deterministic → distributed | more WC-centered modes | benefit from temporal diversity / dimensionality |
| distributed → independent nonlinear | nonlinear E-I dynamics | benefit from nonlinear local state evolution |
| independent → coupled nonlinear | recurrent graph interaction | benefit from collective network dynamics |

These are controlled comparisons, but they should still be interpreted carefully. In particular, the distributed reservoir can have a different state dimension from the deterministic reservoir, so improvement is not automatically attributable to physiology alone.

---

# Data and preprocessing

The repository assumes the EEG dataset already exists locally. It does not download the dataset.

The default configuration points to:

```yaml
data_root: "C:/Users/meisa/Data/eeg/ds003775_10"
subject_id: "sub-001"
target_channel: "Oz"
task: "resteyesc"
resample_hz: 250.0
```

Change `data_root` in `config/defaults.yaml` to match your local dataset.

## Chronological split

The recording is split chronologically **before** fitting global or learned preprocessing quantities.

This is important because the task is temporal prediction rather than random-sample regression.

The workflow keeps training and held-out test information separated when computing:

- linear detrending parameters;
- z-score statistics;
- resampling-sensitive preprocessing;
- WC fitting;
- ridge-readout fitting.

Train and test partitions are resampled independently so that the polyphase resampling filter cannot access samples across the train/test boundary.

## Default spectral range

The default PSD range is

```yaml
psd:
  fmin_hz: 0.5
  fmax_hz: 45.0
```

---

# Wilson-Cowan fitting

One subject-local WC parameter set is fitted on the **training partition only** and is reused as the common physical center for all four reservoir modes in an ablation run.

The fitted parameters are:

- `tau_e`
- `tau_i`
- `w_ee`
- `w_ei`
- `w_ie`
- `w_ii`
- `p`
- `q`

The sigmoid gain and threshold are taken from configuration rather than optimized in the current fitting routine.

## Default objective

The default objective matches the normalized log-PSD shape of the autonomous WC simulation to the training EEG spectrum.

The generalized objective is

$$
L=\alpha L_{\mathrm{PSD}}+\beta L_{\mathrm{STFT}}+\eta L_{\mathrm{temporal}}.
$$

The default configuration is

```yaml
fit:
  loss:
    psd_weight: 1.0
    stft_weight: 0.0
    temporal_weight: 0.0
```

so the current default remains PSD-only.

### PSD term

The PSD term compares centered log spectral shapes over the configured frequency range.

This emphasizes spectral shape rather than absolute signal power.

### Optional STFT term

When enabled, the STFT term compares standardized log-magnitude time-frequency representations.

The default settings use:

- 1-second windows;
- 50% overlap;
- a Hann window;
- the configured EEG frequency range.

The term is calculated using training data only.

### Optional temporal term

When enabled, the temporal term compares standardized observed and simulated trajectories over a common aligned time interval.

It is disabled by default.

## Optimizer

The WC fit uses SciPy differential evolution within fixed physiological/numerical parameter bounds.

The default optimization budget is intentionally small:

```yaml
fit:
  maxiter: 3
  population_size: 4
  polish: false
```

This is suitable for a proof-of-concept workflow but should not be interpreted as strong parameter identification.

---

# Parameter populations

For nonlinear reservoirs, the fitted parameter vector is the center of the ensemble.

A scalar `parameter_jitter` is interpreted as a **fractional perturbation scale**:

```yaml
nonlinear_reservoir:
  parameter_jitter: 0.05
```

Parameter-specific scales can also be provided:

```yaml
nonlinear_reservoir:
  parameter_jitter:
    tau_e: 0.03
    tau_i: 0.05
    w_ee: 0.08
    p: 0.02
```

A mapping may include a `default` value for unspecified parameters.

## Perturbation rules

Different parameter types are perturbed differently:

- positive time constants and positive coupling parameters use multiplicative perturbations in log space;
- signed drives `p` and `q` use bounded additive perturbations;
- a nonnegative parameter centered exactly at zero uses a one-sided perturbation rather than taking a logarithm of zero.

All sampled parameters remain inside the same supported numerical/physiological bounds used by the WC implementation.

Parameter draws, input projections, and graph generation use separate deterministic random streams.

---

# Stability and validity

Linear and nonlinear reservoirs require different validity criteria.

## Linear modes

`deterministic_poles` and `distributed_poles` require continuous-time stability:

$$
\mathrm{Re}\,\lambda<0.
$$

After discretization this implies

$$
|\mu|<1.
$$

## Nonlinear modes

`independent_nonlinear_wc` and `coupled_nonlinear_wc` are checked by integrating a short autonomous trajectory before the driven experiment.

A nonlinear parameter sample is accepted only if the RK4 trajectory remains:

- finite;
- numerically valid;
- inside the configured state bounds.

The preflight does not clip a failed nonlinear trajectory back into range.

A nonlinear block is **not** rejected merely because an equilibrium has a nonnegative eigenvalue real part. Such a system can still possess bounded nonlinear oscillations, for example near a Hopf bifurcation.

The distinction is fundamental:

```text
linear stability != nonlinear boundedness
```

During the actual driven reservoir run, RK4 substeps are also checked for finite values and configured state bounds.

---

# Units

The repository uses the following conventions:

- sampling rates: Hz;
- all time constants and durations: seconds;
- default processed EEG rate: 250 Hz;
- default EEG sample interval: $\Delta t=0.004$ s;
- continuous eigenvalue real parts: s$^{-1}$;
- continuous eigenvalue imaginary parts: rad/s;
- discrete poles $\mu$: dimensionless;
- $E$ and $I$: dimensionless activity fractions;
- z-scored EEG input: dimensionless.

For consistency, the fitting integration interval `fit.dt` should match the processed EEG sample interval when fitted continuous poles are being compared with discrete reservoir dynamics.

---

# Prediction task

The evaluation task is **teacher-forced one-step prediction**.

Given an observed EEG sample $u_t$, the reservoir state is used to predict

$$
u_{t+1}.
$$

The reservoir is therefore continually driven by the observed signal during evaluation.

This is not an autonomous rollout where previous predictions are fed back as future inputs.

## Readout

Only the linear ridge readout is fitted.

For reservoir feature vector $h_t$, the predictor has the form

$$
\widehat{u}_{t+1}=w^T h_t+b.
$$

For complex linear reservoirs, the real and imaginary state components are concatenated before fitting the real-valued ridge readout.

The default ridge regularization is

```yaml
reservoir:
  readout_ridge: 1.0e-4
```

## Washout and warmup

The default prediction configuration is

```yaml
prediction:
  train_fraction: 0.8
  washout_samples: 500
  warmup_samples: 500
```

- `washout_samples` removes the initial reservoir transient from readout fitting;
- `warmup_samples` initializes reservoir state before held-out prediction.

## Persistence baseline

Every mode is compared against the persistence predictor

$$
\widehat{u}_{t+1}=u_t.
$$

This baseline is particularly important for one-step EEG prediction because adjacent samples are strongly autocorrelated.

---

# Evaluation

Each mode reports the same core metrics:

- RMSE;
- MAE;
- Pearson correlation;
- persistence-baseline performance;
- runtime;
- random seed;
- reservoir state dimension;
- mode or WC-block count;
- fitted WC parameters;
- effective configuration;
- stability or boundedness diagnostics.

Linear-mode summaries distinguish the number of complex modes from the number of real-valued features presented to the ridge readout.

The strongest result is not necessarily the mode with the lowest raw error alone. The purpose of the ablation is also to determine **what additional dynamical structure was required to obtain that improvement**.

---

# Installation

The package requires Python 3.10 or newer.

Core dependencies are:

- MNE;
- NumPy;
- SciPy;
- PyYAML.

`pytest` is used for tests.

Clone the repository:

```bash
git clone https://github.com/meis-01/crc.git
cd crc
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Alternatively, install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

---

# Configuration

The main experiment configuration is:

```text
config/defaults.yaml
```

Important sections include:

| Section | Purpose |
| --- | --- |
| top-level data fields | dataset, subject, task, channel, resampling |
| `psd` | spectral analysis range |
| `wilson_cowan` | initial/default WC parameters |
| `fit` | WC fitting and hybrid loss |
| `reservoir` | linear pole reservoirs and ridge readout |
| `nonlinear_reservoir` | nonlinear population and RK4 settings |
| `coupling` | sparse graph configuration |
| `prediction` | chronological split, washout, warmup |
| `ablation` | modes included in the full comparison |

The default reservoir mode is currently:

```yaml
reservoir:
  reservoir_mode: distributed_poles
```

---

# Running experiments

The local EDF dataset must already exist.

## Run the configured default

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml
```

## Run one reservoir mode

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode deterministic_poles
```

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode distributed_poles
```

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode independent_nonlinear_wc
```

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --mode coupled_nonlinear_wc
```

## Run the complete four-mode ablation

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --all-modes
```

The ablation reuses one common preprocessing result and one common WC fit.

## Use another artifact root

```powershell
python scripts/run_sub001_pipeline.py --config config/defaults.yaml --all-modes --output-dir artifacts/experiment-01
```

The script prints the resulting summary as JSON.

---

# Artifacts

Mode outputs are isolated so that one architecture cannot overwrite another.

The intended layout is:

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

Each mode directory stores its own experiment outputs, including the relevant combination of:

- effective configuration;
- shared WC fit information;
- PSD outputs;
- simulation outputs;
- prediction outputs;
- metrics;
- runtime;
- sampled poles or nonlinear WC parameters;
- input projections;
- stability/boundedness diagnostics;
- sparse coupling graph data where applicable.

The fixed reservoir realization is saved so seeded experiments can be audited and reproduced.

---

# Runtime considerations

The nonlinear equations are vectorized across WC blocks.

For the coupled mode, graph recurrence uses SciPy CSR sparse matrix-vector multiplication. The derivative cost therefore scales approximately with the number of graph edges rather than with a dense $K^2$ interaction matrix.

Development CPU checks reported approximately:

- 3.2–3.5 s for 60,000 samples with 50–100 independent blocks;
- 4.0–4.2 s for degree-4 coupled blocks;
- one RK4 substep per EEG sample.

These are implementation checks, not portable benchmarks. Runtime depends on CPU, reservoir size, graph degree, RK4 substeps, and recording length.

---

# Tests

Run the test suite with:

```bash
python -m pytest
```

The repository includes tests for data loading and the core Wilson-Cowan / physics-reservoir behavior.

The design also emphasizes regression tests for:

- deterministic seeded construction;
- pole stability;
- conjugate-pair handling;
- nonlinear state dimension;
- RK4 validity;
- zero-coupling equivalence;
- graph reproducibility;
- preprocessing separation.

---

# Repository structure

```text
crc/
├── config/
│   └── defaults.yaml
├── scripts/
│   └── run_sub001_pipeline.py
├── src/
│   └── physics_esn/
│       ├── analysis/
│       ├── data/
│       ├── fitting/
│       ├── models/
│       ├── config.py
│       └── pipeline.py
├── tests/
├── PROJECT_PLAN.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

At a high level:

- `data/` handles EEG discovery/loading and preprocessing;
- `analysis/` contains signal-analysis utilities such as PSD computation;
- `fitting/` contains WC parameter fitting;
- `models/` contains Wilson-Cowan, ESN, complex-reservoir, and physics-reservoir components;
- `pipeline.py` coordinates preparation, fitting, reservoir execution, evaluation, and artifact generation;
- `scripts/run_sub001_pipeline.py` provides the command-line experiment entry point.

---

# Reproducibility principles

The experiment is designed around several invariants:

1. **Chronological separation:** test data must not influence fitted preprocessing, WC parameters, or readout parameters.
2. **Shared physical center:** all four modes in an ablation use the same subject-local WC fit.
3. **Fixed reservoirs:** only the readout is trained.
4. **Independent random streams:** parameter jitter, input projections, and graph construction are seeded separately.
5. **Mode isolation:** each reservoir writes to its own artifact directory.
6. **Comparable evaluation:** all modes solve the same one-step task and use the same persistence baseline.
7. **Explicit validity criteria:** linear stability and nonlinear boundedness are treated separately.

These constraints are important because the goal is scientific comparison, not merely architecture search.

---

# What the experiment can and cannot show

A successful ablation can support statements such as:

- WC-derived local time scales provide useful temporal features;
- increasing diversity around those time scales improves one-step prediction;
- full nonlinear WC dynamics improve over the distributed linear baseline;
- sparse cross-block recurrence improves over independent nonlinear blocks.

However, the current experiment does **not** by itself establish that:

- fitted WC parameters are physiologically identifiable;
- the nonlinear reservoir states correspond directly to measured neural populations;
- better one-step prediction implies better long-horizon simulation;
- the model generalizes across subjects;
- any particular WC parameter is uniquely responsible for prediction performance.

The experiment is best interpreted as a controlled study of **inductive bias in fixed dynamical reservoirs**.

---

# Current limitations

- The validated workflow currently targets one `Oz` channel from `sub-001`.
- It is not yet multichannel, multi-region, or cross-subject.
- Evaluation is one-step teacher-forced prediction rather than autonomous long-horizon forecasting.
- The default WC fitting budget is intentionally small (`maxiter: 3`).
- PSD or STFT agreement should not be interpreted as WC parameter identifiability.
- Nonlinear parameter populations are bounded perturbations around one fitted center, not samples from a calibrated posterior distribution.
- The linear modes encode only local Jacobian dynamics and must not be described as complete nonlinear WC simulations.
- Reservoir dimensions differ across some modes, so capacity and dynamical assumptions are not perfectly orthogonal.
- Current results should be treated as proof-of-concept evidence until repeated across subjects and seeds.

---

# Planned extensions

The project plan leaves several directions for later experiments:

- repeat the ablation across `sub-001` through `sub-010`;
- extend from one channel to multichannel EEG;
- study cross-region or structured neural-mass coupling;
- increase the WC fitting budget;
- evaluate robustness over multiple seeds;
- perform matched-dimension ablations where useful;
- calibrate nonlinear WC parameter populations probabilistically;
- investigate longer-horizon or autonomous prediction only after one-step behavior is well validated.

Trainable recurrent weights and unrelated neural architectures are intentionally outside the current controlled ablation.

---

## Summary

This repository tests a specific progression:

$$
\text{local WC linearization}
\rightarrow
\text{distributed linear WC modes}
\rightarrow
\text{nonlinear WC ensemble}
\rightarrow
\text{coupled nonlinear WC network}.
$$

The key methodological choice is that the recurrent dynamics remain fixed and only a ridge readout is trained.

That makes the central question explicit:

> **How much predictive value comes from WC-derived time scales, how much from nonlinear neural-population dynamics, and how much from interactions between those dynamics?**
