from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CANONICAL_LEADS, normalize_lead_name
from .io import ManifestRow, load_record
from .processing import canonicalize_record


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def inspect_manifest(
    rows: list[ManifestRow],
    raw_root: str | Path | None = None,
    published_duration_range: tuple[float, float] = (7.0, 12.0),
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    record_ids = Counter(row.record_id for row in rows)
    patient_ids = {row.patient_id for row in rows}
    exam_ids = {row.exam_id for row in rows if row.exam_id}
    sampling_rates: Counter[str] = Counter()
    source_orders: Counter[str] = Counter()
    units: Counter[str] = Counter()
    sample_counts: list[float] = []
    durations: list[float] = []
    lead_min = np.full(len(CANONICAL_LEADS), np.inf, dtype=np.float64)
    lead_max = np.full(len(CANONICAL_LEADS), -np.inf, dtype=np.float64)
    nan_values = 0
    infinite_values = 0
    invalid_records: set[str] = set()
    missing_lead_records = 0
    load_error_records = 0
    nonfinite_records = 0
    duration_warning_records = 0
    waveform_hash_first: dict[str, str] = {}
    exact_duplicate_pairs: list[dict[str, str]] = []

    for record_id, count in record_ids.items():
        if count > 1:
            invalid_records.add(record_id)
            issues.append(_issue(record_id, "error", "duplicate_record_id", f"record_id occurs {count} times"))

    for row in rows:
        try:
            loaded = load_record(row, raw_root)
        except Exception as exc:  # keep inspecting after corrupt/unrecognized records
            load_error_records += 1
            invalid_records.add(row.record_id)
            issues.append(_issue(row.record_id, "error", "load_error", f"{type(exc).__name__}: {exc}"))
            continue

        units[loaded.amplitude_unit or "unknown"] += 1
        source_orders["|".join(map(str, loaded.lead_names))] += 1
        sampling_rates[f"{loaded.sampling_rate:g}"] += 1
        sample_counts.append(float(loaded.ecg.shape[0]))
        if np.isfinite(loaded.sampling_rate) and loaded.sampling_rate > 0:
            duration = loaded.ecg.shape[0] / loaded.sampling_rate
            durations.append(float(duration))
            if not (published_duration_range[0] <= duration <= published_duration_range[1]):
                duration_warning_records += 1
                issues.append(
                    _issue(
                        row.record_id,
                        "warning",
                        "duration_outside_published_range",
                        f"{duration:.6g}s is outside {published_duration_range[0]}-{published_duration_range[1]}s",
                    )
                )
        else:
            invalid_records.add(row.record_id)
            issues.append(_issue(row.record_id, "error", "invalid_sampling_rate", str(loaded.sampling_rate)))

        array = np.asarray(loaded.ecg)
        row_nan = int(np.isnan(array).sum())
        row_inf = int(np.isinf(array).sum())
        nan_values += row_nan
        infinite_values += row_inf
        if row_nan or row_inf:
            nonfinite_records += 1
            invalid_records.add(row.record_id)
            issues.append(
                _issue(row.record_id, "error", "nonfinite_values", f"NaN={row_nan}, infinite={row_inf}")
            )

        try:
            canonical, normalized_order = canonicalize_record(loaded)
        except Exception as exc:
            missing_lead_records += 1
            invalid_records.add(row.record_id)
            issues.append(_issue(row.record_id, "error", "lead_set_invalid", str(exc)))
            continue

        if array.shape[0] < 2:
            invalid_records.add(row.record_id)
            issues.append(_issue(row.record_id, "error", "insufficient_samples", str(array.shape[0])))
            continue
        if row_nan or row_inf:
            continue
        lead_min = np.minimum(lead_min, np.min(canonical, axis=0))
        lead_max = np.maximum(lead_max, np.max(canonical, axis=0))
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(canonical, dtype="<f4").tobytes(order="C"))
        digest.update(f"|{loaded.sampling_rate:g}|{'|'.join(CANONICAL_LEADS)}".encode())
        key = digest.hexdigest()
        if key in waveform_hash_first:
            exact_duplicate_pairs.append(
                {"record_id": row.record_id, "duplicate_of": waveform_hash_first[key]}
            )
            issues.append(
                _issue(row.record_id, "warning", "exact_waveform_duplicate", waveform_hash_first[key])
            )
        else:
            waveform_hash_first[key] = row.record_id

    ranges = {
        lead: {
            "min": None if not np.isfinite(lead_min[index]) else float(lead_min[index]),
            "max": None if not np.isfinite(lead_max[index]) else float(lead_max[index]),
        }
        for index, lead in enumerate(CANONICAL_LEADS)
    }
    invalid_row_count = sum(record_ids[record_id] for record_id in invalid_records)
    summary: dict[str, Any] = {
        "manifest_records": len(rows),
        "patients": len(patient_ids),
        "exams": len(exam_ids) if exam_ids else None,
        "valid_records": len(rows) - invalid_row_count,
        "invalid_records": invalid_row_count,
        "load_error_records": load_error_records,
        "missing_or_invalid_lead_records": missing_lead_records,
        "nonfinite_records": nonfinite_records,
        "nan_values": nan_values,
        "infinite_values": infinite_values,
        "duration_warning_records": duration_warning_records,
        "exact_waveform_duplicate_records": len(exact_duplicate_pairs),
        "exact_waveform_duplicate_pairs": exact_duplicate_pairs,
        "sampling_rate_counts_hz": dict(sorted(sampling_rates.items())),
        "samples_per_record": _distribution(sample_counts),
        "duration_seconds": _distribution(durations),
        "source_lead_order_counts": dict(source_orders.most_common()),
        "canonical_output_lead_order": list(CANONICAL_LEADS),
        "amplitude_units": dict(units.most_common()),
        "amplitude_ranges_by_lead": ranges,
        "issue_counts": dict(Counter(issue["issue"] for issue in issues)),
    }
    return summary, issues


def write_inspection_outputs(
    summary: dict[str, Any],
    issues: list[dict[str, str]],
    report_json: str | Path,
    issues_csv: str | Path,
) -> None:
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    issue_path = Path(issues_csv)
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    with issue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "severity", "issue", "detail"))
        writer.writeheader()
        writer.writerows(issues)


def _issue(record_id: str, severity: str, issue: str, detail: str) -> dict[str, str]:
    return {"record_id": record_id, "severity": severity, "issue": issue, "detail": detail}
