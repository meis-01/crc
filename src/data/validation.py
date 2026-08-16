from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import CANONICAL_LEADS, normalize_target_lead
from .dataset import decode_h5_string, invert_record, lead_names_from_h5, load_reconstruction_sample
from .splits import assign_patient_splits, split_assignment_digest


def validate_prepared_dataset(
    path: str | Path,
    target_lead: str = "V2",
    expected_rate: float | None = None,
    limb_check_records: int = 32,
) -> dict[str, Any]:
    target = normalize_target_lead(target_lead)
    checks: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        required = {
            "ecg",
            "lead_names",
            "metadata/offset",
            "metadata/length",
            "metadata/patient_id",
            "metadata/record_id",
            "metadata/sampling_rate",
            "metadata/split",
            "metadata/normalization_center",
            "metadata/normalization_scale",
        }
        missing = sorted(name for name in required if name not in handle)
        checks["required_fields"] = {"passed": not missing, "missing": missing}
        if missing:
            return _result(checks)

        leads = lead_names_from_h5(handle)
        checks["lead_count_and_order"] = {
            "passed": leads == list(CANONICAL_LEADS),
            "actual": leads,
            "expected": list(CANONICAL_LEADS),
        }
        n_records = int(handle["metadata/offset"].shape[0])
        metadata_lengths = {
            name: int(handle[f"metadata/{name}"].shape[0])
            for name in ("offset", "length", "patient_id", "record_id", "sampling_rate", "split")
        }
        checks["metadata_alignment"] = {
            "passed": len(set(metadata_lengths.values())) == 1,
            "lengths": metadata_lengths,
        }

        offsets = np.asarray(handle["metadata/offset"], dtype=np.int64)
        lengths = np.asarray(handle["metadata/length"], dtype=np.int64)
        expected_offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))) if n_records else np.array([])
        contiguous = bool(
            n_records > 0
            and np.array_equal(offsets, expected_offsets)
            and offsets[-1] + lengths[-1] == handle["ecg"].shape[0]
            and np.all(lengths > 0)
        )
        checks["record_offsets"] = {"passed": contiguous, "records": n_records}

        finite = True
        dataset = handle["ecg"]
        chunk = 65_536
        for start in range(0, dataset.shape[0], chunk):
            if not np.isfinite(dataset[start : start + chunk]).all():
                finite = False
                break
        checks["finite_values"] = {"passed": finite}

        rates = np.asarray(handle["metadata/sampling_rate"], dtype=np.float64)
        expected = float(expected_rate if expected_rate is not None else handle.attrs["sampling_rate"])
        checks["sampling_frequency"] = {
            "passed": bool(np.allclose(rates, expected)),
            "expected_hz": expected,
            "observed_hz": sorted(set(map(float, rates))),
        }

        patient_splits: dict[str, set[str]] = defaultdict(set)
        stored_assignments: dict[str, str] = {}
        for patient_raw, split_raw in zip(handle["metadata/patient_id"], handle["metadata/split"], strict=True):
            patient = decode_h5_string(patient_raw)
            split = decode_h5_string(split_raw)
            patient_splits[split].add(patient)
            stored_assignments.setdefault(patient, split)
        overlaps = {
            "train_validation": sorted(patient_splits["train"] & patient_splits["validation"]),
            "train_test": sorted(patient_splits["train"] & patient_splits["test"]),
            "validation_test": sorted(patient_splits["validation"] & patient_splits["test"]),
        }
        checks["patient_disjoint_splits"] = {
            "passed": not any(overlaps.values()),
            "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        }
        stored_digest = str(handle.attrs.get("split_assignment_sha256", ""))
        digest_matches = split_assignment_digest(stored_assignments) == stored_digest
        if str(handle.attrs.get("split_mode", "")) == "random":
            ratios = tuple(json.loads(str(handle.attrs["split_ratios"])))
            reproduced = assign_patient_splits(
                stored_assignments.keys(), int(handle.attrs["split_seed"]), ratios
            )
            reproducible = reproduced == stored_assignments and digest_matches
        else:
            reproducible = digest_matches
        checks["reproducible_splits"] = {
            "passed": reproducible,
            "mode": str(handle.attrs.get("split_mode", "")),
            "assignment_sha256": stored_digest,
        }

        target_excluded = True
        alignment = True
        input_leads: list[str] = []
        if n_records:
            sample = load_reconstruction_sample(handle, 0, target)
            input_leads = sample["input_lead_names"]
            target_excluded = target not in input_leads and len(input_leads) == 11
            alignment = sample["inputs"].shape[0] == sample["target"].shape[0] == int(lengths[0])
        checks["target_excluded_from_inputs"] = {
            "passed": target_excluded,
            "target": target,
            "input_leads": input_leads,
        }
        checks["temporal_alignment"] = {"passed": alignment}

        residuals = _limb_relationship_residuals(handle, min(n_records, limb_check_records))
        checks["limb_lead_relationships"] = {
            "passed": True,
            "informational_only": True,
            "records_checked": min(n_records, limb_check_records),
            "rmse": residuals,
            "note": "Small residuals show why limb-lead reconstruction can be artificially easy.",
        }
    return _result(checks)


def _limb_relationship_residuals(handle: h5py.File, n_records: int) -> dict[str, float | None]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    index = {lead: CANONICAL_LEADS.index(lead) for lead in CANONICAL_LEADS}
    for record_index in range(n_records):
        ecg = invert_record(handle, record_index, restore_baseline=True).astype(np.float64)
        expressions = {
            "II_minus_I_plus_III": ecg[:, index["II"]] - ecg[:, index["I"]] - ecg[:, index["III"]],
            "III_minus_II_minus_I": ecg[:, index["III"]] - (ecg[:, index["II"]] - ecg[:, index["I"]]),
            "aVR_plus_I_plus_II_over_2": ecg[:, index["aVR"]] + (ecg[:, index["I"]] + ecg[:, index["II"]]) / 2,
            "aVL_minus_I_minus_II_over_2": ecg[:, index["aVL"]] - (ecg[:, index["I"]] - ecg[:, index["II"]] / 2),
            "aVF_minus_II_minus_I_over_2": ecg[:, index["aVF"]] - (ecg[:, index["II"]] - ecg[:, index["I"]] / 2),
        }
        for name, error in expressions.items():
            sums[name] += float(np.sum(error * error))
            counts[name] += int(error.size)
    return {
        name: (float(np.sqrt(sums[name] / counts[name])) if counts[name] else None)
        for name in expressions if n_records
    }


def _result(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failed = [name for name, details in checks.items() if not details.get("passed", False)]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}

