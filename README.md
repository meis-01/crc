# CODE-II ECG reconstruction data preparation

This repository prepares clean real-valued 12-lead CODE-II ECG tracings for a configurable
11-input-leads-to-1-target-lead reconstruction experiment. It does not contain modeling code.

Current acquisition status (2026-08-16): the peer-reviewed paper says CODE-II-open and the
CODE-II-test waveforms are public through PhysioNet, but the paper contains no project URL and
PhysioNet's live catalog returns no CODE-II result. No mirror or older CODE-15% data has been
substituted. See [reports/data_preparation.md](reports/data_preparation.md) for evidence, access
steps, verified properties, and the exact workflow.

The local `data` path is a Windows junction to `D:\CODE-II\data`. At setup time, D: had
1,054.96 GiB free.

## Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_data_layout.ps1
python -m pip install -e ".[test,wfdb]"

# After copying the exact official project URL from PhysioNet:
python scripts/download_code_ii.py `
  --project-url "<OFFICIAL_PHYSIONET_PROJECT_URL>" `
  --destination "D:\CODE-II\data\raw\code-ii"

# Build data/interim/codeii_manifest.csv after confirming the released file schema.
python scripts/inspect_dataset.py `
  --manifest data/interim/codeii_manifest.csv `
  --raw-root data/raw

python scripts/prepare_dataset.py `
  --manifest data/interim/codeii_manifest.csv `
  --raw-root data/raw `
  --output data/processed/codeii_reconstruction.h5 `
  --target-lead V2 `
  --target-rate 400 `
  --baseline-correction none `
  --normalization training_set `
  --split-mode official `
  --seed 2026

python scripts/validate_dataset.py `
  --dataset data/processed/codeii_reconstruction.h5 `
  --target-lead V2 `
  --expected-rate 400

python scripts/plot_diagnostics.py `
  --dataset data/processed/codeii_reconstruction.h5 `
  --target-lead V2
```

Run automated tests with:

```powershell
python -m pytest
```

## Prepared format

The HDF5 file stores all samples contiguously as `float32` in `ecg[total_samples, 12]`.
Per-record `offset` and `length` arrays preserve each tracing's native duration without padding or
cropping. Metadata includes lead names, patient and record identifiers, sampling rates, split,
source order, baseline offsets, and invertible normalization parameters.

