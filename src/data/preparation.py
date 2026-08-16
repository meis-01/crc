from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import CANONICAL_LEADS, normalize_target_lead
from .io import ManifestRow, load_record
from .processing import RunningStats, normalization_parameters, process_record
from .splits import resolve_splits, split_assignment_digest


def prepare_dataset(
    rows: list[ManifestRow],
    output_path: str | Path,
    raw_root: str | Path | None = None,
    target_lead: str = "V2",
    target_rate: float | None = None,
    baseline_correction: str = "none",
    normalization: str = "training_set",
    split_mode: str = "auto",
    split_seed: int = 2026,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    compression: str | None = "lzf",
) -> dict[str, Any]:
    target = normalize_target_lead(target_lead)
    if normalization not in {"none", "per_record", "training_set"}:
        raise ValueError("normalization must be none, per_record, or training_set")
    assignments, resolved_split_mode = resolve_splits(rows, split_mode, split_seed, split_ratios)

    record_counts = Counter(row.record_id for row in rows)
    duplicates = [record_id for record_id, count in record_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate record_id values in manifest: {duplicates[:10]}")

    # The discovery pass validates records and determines whether native rates are mixed.
    valid_rows: list[ManifestRow] = []
    invalid: list[dict[str, str]] = []
    native_rates: set[float] = set()
    for row in rows:
        try:
            loaded = load_record(row, raw_root)
            process_record(loaded, target_rate=None, baseline_correction="none")
            native_rates.add(float(loaded.sampling_rate))
            valid_rows.append(row)
        except Exception as exc:
            invalid.append({"record_id": row.record_id, "reason": f"{type(exc).__name__}: {exc}"})

    if not valid_rows:
        raise ValueError("No valid ECG records remain after validation")
    if target_rate is None and len(native_rates) > 1:
        raise ValueError(
            f"Mixed native sampling rates {sorted(native_rates)} require an explicit --target-rate"
        )
    effective_rate = float(target_rate) if target_rate is not None else next(iter(native_rates))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    metadata: dict[str, list[Any]] = defaultdict(list)
    train_stats = RunningStats(len(CANONICAL_LEADS))
    utf8 = h5py.string_dtype("utf-8")
    chunk_rows = 16_384

    try:
        with h5py.File(temp_output, "w") as handle:
            ecg_dataset = handle.create_dataset(
                "ecg",
                shape=(0, len(CANONICAL_LEADS)),
                maxshape=(None, len(CANONICAL_LEADS)),
                chunks=(chunk_rows, len(CANONICAL_LEADS)),
                dtype="f4",
                compression=compression,
            )
            handle.create_dataset("lead_names", data=np.asarray(CANONICAL_LEADS, dtype=utf8))
            total_samples = 0
            for row in valid_rows:
                loaded = load_record(row, raw_root)
                processed = process_record(loaded, target_rate=effective_rate, baseline_correction=baseline_correction)
                values = processed.ecg
                split = assignments[row.patient_id]
                if normalization == "per_record":
                    center, scale = normalization_parameters(values)
                    stored = (values - center) / scale
                else:
                    center = np.zeros(len(CANONICAL_LEADS), dtype=np.float32)
                    scale = np.ones(len(CANONICAL_LEADS), dtype=np.float32)
                    stored = values
                if normalization == "training_set" and split == "train":
                    train_stats.update(values)

                length = int(stored.shape[0])
                ecg_dataset.resize(total_samples + length, axis=0)
                ecg_dataset[total_samples : total_samples + length] = stored.astype(np.float32, copy=False)
                metadata["offset"].append(total_samples)
                metadata["length"].append(length)
                metadata["patient_id"].append(row.patient_id)
                metadata["record_id"].append(row.record_id)
                metadata["exam_id"].append(row.exam_id)
                metadata["tracing_index"].append(row.tracing_index)
                metadata["split"].append(split)
                metadata["sampling_rate"].append(processed.sampling_rate)
                metadata["source_sampling_rate"].append(loaded.sampling_rate)
                metadata["source_num_samples"].append(loaded.ecg.shape[0])
                metadata["source_lead_order"].append("|".join(map(str, loaded.lead_names)))
                metadata["amplitude_unit"].append(loaded.amplitude_unit)
                metadata["baseline_offset"].append(processed.baseline_offset)
                metadata["normalization_center"].append(center)
                metadata["normalization_scale"].append(scale)
                total_samples += length

            if normalization == "training_set":
                global_center, global_scale = train_stats.parameters()
                for start in range(0, total_samples, chunk_rows):
                    stop = min(start + chunk_rows, total_samples)
                    ecg_dataset[start:stop] = (ecg_dataset[start:stop] - global_center) / global_scale
                metadata["normalization_center"] = [global_center] * len(valid_rows)
                metadata["normalization_scale"] = [global_scale] * len(valid_rows)
            elif normalization == "none":
                global_center = np.zeros(len(CANONICAL_LEADS), dtype=np.float32)
                global_scale = np.ones(len(CANONICAL_LEADS), dtype=np.float32)
            else:
                global_center = np.full(len(CANONICAL_LEADS), np.nan, dtype=np.float32)
                global_scale = np.full(len(CANONICAL_LEADS), np.nan, dtype=np.float32)

            group = handle.create_group("metadata")
            for key in (
                "patient_id",
                "record_id",
                "exam_id",
                "tracing_index",
                "split",
                "source_lead_order",
                "amplitude_unit",
            ):
                group.create_dataset(key, data=np.asarray(metadata[key], dtype=utf8))
            for key, dtype in (
                ("offset", "i8"),
                ("length", "i8"),
                ("sampling_rate", "f4"),
                ("source_sampling_rate", "f4"),
                ("source_num_samples", "i8"),
            ):
                group.create_dataset(key, data=np.asarray(metadata[key], dtype=dtype))
            for key in ("baseline_offset", "normalization_center", "normalization_scale"):
                group.create_dataset(key, data=np.asarray(metadata[key], dtype="f4"))

            normalization_group = handle.create_group("normalization")
            normalization_group.create_dataset("training_center", data=global_center)
            normalization_group.create_dataset("training_scale", data=global_scale)
            handle.attrs.update(
                {
                    "format": "codeii_reconstruction_flat_v1",
                    "target_lead": target,
                    "input_leads": json.dumps([lead for lead in CANONICAL_LEADS if lead != target]),
                    "normalization": normalization,
                    "baseline_correction": baseline_correction,
                    "split_mode": resolved_split_mode,
                    "split_seed": int(split_seed),
                    "split_ratios": json.dumps(split_ratios),
                    "split_assignment_sha256": split_assignment_digest(assignments),
                    "sampling_rate": effective_rate,
                }
            )
            handle.flush()
        temp_output.replace(output)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    split_patients: dict[str, set[str]] = defaultdict(set)
    split_records: Counter[str] = Counter()
    for row in valid_rows:
        split = assignments[row.patient_id]
        split_patients[split].add(row.patient_id)
        split_records[split] += 1
    summary = {
        "output": str(output.resolve()),
        "records_in_manifest": len(rows),
        "valid_records_written": len(valid_rows),
        "invalid_records_removed": len(invalid),
        "invalid_records": invalid,
        "patients_written": len({row.patient_id for row in valid_rows}),
        "exams_written": len({row.exam_id for row in valid_rows if row.exam_id}),
        "native_sampling_rates_hz": sorted(native_rates),
        "output_sampling_rate_hz": effective_rate,
        "lead_names": list(CANONICAL_LEADS),
        "target_lead": target,
        "input_leads": [lead for lead in CANONICAL_LEADS if lead != target],
        "normalization": normalization,
        "baseline_correction": baseline_correction,
        "split_mode": resolved_split_mode,
        "split_seed": split_seed,
        "split_assignment_sha256": split_assignment_digest(assignments),
        "split_counts": {
            split: {"patients": len(split_patients[split]), "records": split_records[split]}
            for split in ("train", "validation", "test")
        },
    }
    return summary
