# Physics-Informed ESN for EEG

This repository now targets a first proof-of-concept for EEG forecasting with a
physics-informed complex reservoir derived from the Wilson-Cowan
excitatory/inhibitory neural population model.

The local EEG dataset is assumed to already exist on disk. No download logic is
included. Configure the dataset root in YAML:

```yaml
data_root: "C:/Users/meisa/Data/eeg/ds003775_10"
```

Implemented proof-of-concept scope:

- complete milestones 1-4 pipeline for `sub-001` only
- `Oz` extraction
- raw inspection
- PSD computation
- Wilson-Cowan simulation
- Wilson-Cowan PSD fitting on the chronological training partition
- equilibrium, Jacobian, continuous eigenvalues `lambda`, and discrete
  reservoir eigenvalues `mu = exp(lambda * dt)`
- construction of a first diagonal complex reservoir from those eigenvalues
- one-step-ahead `Oz` prediction on the held-out temporal partition
- per-subject artifact directories that keep PSD, simulation, fit, eigenvalues,
  and prediction outputs isolated

The repo also includes discovery utilities for all ten local subject folders and
tests for EDF discovery and `Oz` extraction. Milestone 5 is intentionally not run
yet; see `PROJECT_PLAN.md`.

Run tests with:

```powershell
python -m pytest
```

Run the `sub-001` pipeline with:

```powershell
python scripts/run_sub001_pipeline.py
```
