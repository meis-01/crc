# CODE-II data preparation

Status date: 2026-08-16

## Outcome and acquisition status

The repository structure, external D: storage, acquisition guardrails, inspection code, preparation
pipeline, validation checks, diagnostic plotting, and automated tests are complete. Download and
empirical inspection of CODE-II are blocked by the official release state: the published paper says
the open/test signals are available through PhysioNet, but it does not provide a PhysioNet project
URL, and a search of the live PhysioNet catalog for `CODE-II-open` returns no result. Inventing a
slug, using a mirror, or silently substituting the older CODE-15% cohort would be scientifically
incorrect, so no ECG files were downloaded.

Official sources checked:

- [Peer-reviewed CODE-II paper and documentation](https://www.nature.com/articles/s41746-026-02704-4)
- [Published reference PDF containing Methods and Data availability](https://www.nature.com/articles/s41746-026-02704-4_reference.pdf)
- [Live PhysioNet CODE-II-open catalog search](https://physionet.org/content/?topic=CODE-II-open)
- [Authors' official ECG preprocessing repository](https://github.com/antonior92/ecg-preprocessing),
  inspected at commit `439bc1bafde571feee9e79e86f475ae9ee37d9e4` (MIT license)
- [CODE diagnostic-class site named by the paper](https://code.telessaude.hc.ufmg.br/). Its TLS
  certificate did not validate for this hostname during inspection, so certificate verification was
  not bypassed.

### Exact action required for public CODE-II-open / CODE-II-test

1. Open the paper's Data availability section or search PhysioNet for the newly published official
   CODE-II project. Confirm the page is hosted on `https://physionet.org/content/...` and names the
   CODE-II authors/TNMG.
2. Copy that exact URL. Do not use CODE-15%, CODE-I, CODE-Test (2020), Zenodo mirrors, or a third-party
   dataset index as a substitute.
3. Run:

   ```powershell
   python scripts/download_code_ii.py `
     --project-url "<EXACT_OFFICIAL_PHYSIONET_PROJECT_URL>" `
     --destination "D:\CODE-II\data\raw\code-ii"
   ```

   The script rejects non-PhysioNet hosts, discovers the official `/files/` root from the project
   page, streams files to D:, and writes URLs, byte sizes, and SHA-256 hashes to
   `download_manifest.json`.

There is no honest value that can replace the bracketed URL today: neither the published article nor
PhysioNet currently exposes it.

### Exact action required for the full CODE-II cohort

The full cohort is restricted to non-commercial research and requires an appropriate data-use
agreement. Email `telessaude.hc-ufmg@ebserh.gov.br` and copy the three corresponding authors named in
the paper:

- `petrusabreu@ufmg.br`
- `antonio.ribeiro@ebserh.gov.br`
- `antonio.horta.ribeiro@it.uu.se`

State the institution, principal investigator, non-commercial purpose, requested cohort, lead-
reconstruction protocol, storage/security plan, and intended outputs. Complete the DUA,
confidentiality, institutional ethics, and credential steps that TNMG returns. Use only the download
mechanism they authorize. Nothing in this repository attempts to bypass authentication or a DUA.

## Verified dataset documentation

The following are published properties, not local empirical measurements:

| Property | Verified documentation |
|---|---|
| Full CODE-II size | 2,735,269 ECG exams from 2,093,807 adult patients, January 2019–December 2022 |
| CODE-II-open | 15,000 first-exam, unique-patient ECG exams |
| CODE-II-test | 8,475 unique-patient ECG exams; patients do not overlap full CODE-II |
| Signals | Standard 12-lead ECG; up to four raw successive tracings per exam |
| Native rates | 300, 500, 600, or 1000 Hz, depending on device |
| Duration | 7–12 seconds per tracing |
| Leads | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 are expected; the released on-disk order is **not documented and has not been assumed** |
| Identifiers | Patient ID, exam ID, and a sequential tracing index/acquisition order |
| Open metadata | Exam date, date of birth, sex, reported comorbidities, clinical indication, report upload date, original/revised report type, ECG measurements, and 66 diagnostic class IDs/labels |
| ECG measurements | Heart rate, P-wave duration, PR interval, QRS duration, QRS axis, and QTc are described |
| CODE-II-test metadata | Exam ID/date, patient ID, date of birth, sex, age, specialist IDs, and final diagnoses are used internally; public expert labels are reserved for controlled benchmarking |
| Internal format | TNMG converts captures to a custom internal format |
| Public on-disk format | Not stated in the paper and not inspectable until the PhysioNet record exists |
| Open/test license | CC BY-NC-SA 4.0 |
| Full-cohort terms | Restricted, non-commercial research, individual review, DUA required; no general open license is asserted |

The paper's baseline classifier resampled to 400 Hz, trimmed/padded to 4096 samples, used only eight
essential leads, and applied baseline/powerline filters. Those classification choices are not copied
here: reconstruction keeps all 12 leads and variable durations, never pads or crops, and defaults to no
filtering. The `--target-rate 400` command is used only because documented native rates are mixed;
polyphase resampling is applied to all leads together.

## Storage and repository structure

D: had 1,054.96 GiB free during setup. The repository `data` path is a junction to
`D:\CODE-II\data`, keeping large files out of OneDrive and Git.

```text
crc/
├── data/ -> D:\CODE-II\data
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reports/
│   ├── data_preparation.md
│   └── figures/
├── scripts/
│   ├── build_manifest.py
│   ├── download_code_ii.py
│   ├── inspect_dataset.py
│   ├── plot_diagnostics.py
│   ├── prepare_dataset.py
│   ├── setup_data_layout.ps1
│   └── validate_dataset.py
├── src/data/
└── tests/
```

`.gitignore` excludes raw/interim/processed data, waveform/container formats, generated figures, and
inspection outputs.

## Manifest and exact preparation commands

Preparation consumes a trace-level CSV manifest. Required fields are `path`, `patient_id`, and
`record_id`; supported fields include `source_format`, `exam_id`, `tracing_index`, `dataset_key`,
`index`, `sampling_rate`, `lead_names`, `axis_order`, `amplitude_unit`, `official_split`, and
`dataset_part`. This explicit contract prevents an undocumented HDF5 axis or lead order from being
guessed.

If the official release is a three-dimensional HDF5 matrix with a companion trace-level metadata CSV,
build the manifest only after reading its official README and inspecting its attributes:

```powershell
python scripts/build_manifest.py `
  --signals data/raw/code-ii/signals.h5 `
  --dataset-key "<OFFICIAL_DATASET_KEY>" `
  --metadata-csv data/raw/code-ii/metadata.csv `
  --output data/interim/codeii_manifest.csv `
  --lead-names "<ACTUAL_ORDER_FROM_RELEASE>" `
  --axis-order "<records_samples_leads_OR_records_leads_samples>"
```

The placeholders cannot be resolved before the released schema exists. After a manifest has been
built, the rest of the workflow is exact:

```powershell
python scripts/inspect_dataset.py `
  --manifest data/interim/codeii_manifest.csv `
  --raw-root data/raw `
  --report-json reports/dataset_inspection.json `
  --issues-csv reports/data_quality_issues.csv

python scripts/prepare_dataset.py `
  --manifest data/interim/codeii_manifest.csv `
  --raw-root data/raw `
  --output data/processed/codeii_reconstruction.h5 `
  --target-lead V2 `
  --target-rate 400 `
  --baseline-correction none `
  --normalization training_set `
  --split-mode official `
  --seed 2026 `
  --summary-json reports/preparation_summary.json

python scripts/validate_dataset.py `
  --dataset data/processed/codeii_reconstruction.h5 `
  --target-lead V2 `
  --expected-rate 400 `
  --report-json reports/validation.json

python scripts/plot_diagnostics.py `
  --dataset data/processed/codeii_reconstruction.h5 `
  --target-lead V2 `
  --output-dir reports/figures `
  --per-split 1
```

For the official design, combine CODE-II-open's published train/validation assignments with
CODE-II-test records forced to `test`. The preparer refuses a partial official split. If CODE-II-test
is still unavailable, an explicitly requested deterministic `--split-mode random --seed 2026` is the
fallback, but it is not the preferred reported experiment.

## Processing decisions and output schema

- Source lead names and order are loaded per tracing and reported before reordering.
- Output order is fixed to `I, II, III, aVR, aVL, aVF, V1–V6` only after all 12 names are verified.
- Missing/duplicate leads, unreadable records, invalid rates, insufficient samples, NaNs, and infinities
  are removed and reported. Unexpected 7–12 second duration is a warning, not an automatic deletion.
- Values are stored as `float32`; no Hilbert or complex representation is generated.
- Default baseline correction is `none`. Optional `median` correction removes only a constant per-lead
  offset and stores that offset for inversion; it is not a morphology-changing high-pass filter.
- Default normalization is per-lead training-set mean/std computed only from training patients. The
  parameters are stored globally and per record for inversion. Per-record normalization is supported,
  but its target-lead statistics would be unavailable in a deployment setting and should only be used
  for controlled comparisons.
- Every record is a tracing. Exams and patients are retained separately, so reports distinguish patients,
  exams, and waveform records.

HDF5 schema:

```text
ecg                              float32 [total_samples, 12]
lead_names                       UTF-8 [12]
metadata/offset                  int64 [records]
metadata/length                  int64 [records]
metadata/patient_id              UTF-8 [records]
metadata/record_id               UTF-8 [records]
metadata/exam_id                 UTF-8 [records]
metadata/tracing_index           UTF-8 [records]
metadata/sampling_rate           float32 [records]
metadata/source_sampling_rate    float32 [records]
metadata/source_num_samples      int64 [records]
metadata/source_lead_order       UTF-8 [records]
metadata/amplitude_unit          UTF-8 [records]
metadata/split                   UTF-8 [records]
metadata/baseline_offset         float32 [records, 12]
metadata/normalization_center    float32 [records, 12]
metadata/normalization_scale     float32 [records, 12]
```

The target is selected at load time. For `--target-lead V2`, `inputs` has shape
`[samples, 11]` and `target` has shape `[samples]`, sliced from the same time-aligned array.

## Leakage and ECG sanity checks

Official documentation gives CODE-II-open 12,000 train and 3,000 validation patients/exams, while
CODE-II-test supplies 8,475 non-overlapping test patients/exams. Exact tracing counts cannot be stated
until the up-to-four records per exam are downloaded and counted. Random fallback splits operate on
unique sorted patient IDs and store the seed, ratios, and SHA-256 assignment digest. Validation fails if
one patient appears in multiple splits.

The limb leads have deterministic ideal relationships:

```text
II  = I + III                 (Einthoven)
III = II - I
aVR = -(I + II) / 2
aVL = I - II / 2
aVF = II - I / 2
```

Many ECG devices derive rather than independently measure some limb leads. The validator reports RMSE
residuals for these equations after inverting normalization. Consequently, a model reconstructing III,
aVR, aVL, or aVF from the other limb leads may mostly learn algebra. V1–V6 are marked as the primary
scientifically interesting targets; V2 is the default example.

Automated validation checks the required schema, exact output lead count/order, offset integrity,
finite values, expected frequency, patient-disjoint and reproducible splits, 11/1 temporal alignment,
and exclusion of the target from inputs. Diagnostic plotting selects representative records per split
and highlights the target in red.

## Dataset statistics and data-quality results

No local dataset statistics, split tracing counts, amplitude ranges, actual lead order, duplicates,
missing leads, or diagnostic plots are reported because no official waveform files were accessible.
The inspection script will produce all of these from the downloaded release. Published cohort counts are
listed above and are intentionally not mislabeled as observed results.

The paper reports that full-cohort curation excluded 292,332 exam IDs: 23,358 for invalid clinical data,
83,790 for ECG technical problems, 18,016 already flagged for removal, and 167,168 pediatric cases.
The public tracings were additionally selected by an automated pipeline that excluded corrupted or
structurally inconsistent signals, but local validation remains necessary.

## Verification performed without CODE-II

The implementation was exercised with synthetic ECGs containing exact limb-lead relationships, mixed
300/500 Hz sampling, and two different source lead orders. `python -m pytest` passes five tests covering
inspection, deterministic patient splitting, resampling, preparation, float32 output, official partition
handling, finite values, lead order, target exclusion, alignment, and normalization inversion. Synthetic
statistics and plots are not presented as CODE-II results.

