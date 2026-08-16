from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from data.constants import CANONICAL_LEADS


def synthetic_ecg(rate: int, patient_number: int) -> np.ndarray:
    time = np.arange(rate, dtype=np.float64) / rate
    phase = patient_number * 0.07
    lead_i = 0.7 * np.sin(2 * np.pi * 1.2 * time + phase)
    lead_ii = 0.4 * np.sin(2 * np.pi * 1.2 * time + phase) + 0.5 * np.cos(2 * np.pi * 0.8 * time)
    lead_iii = lead_ii - lead_i
    avr = -(lead_i + lead_ii) / 2
    avl = lead_i - lead_ii / 2
    avf = lead_ii - lead_i / 2
    chest = [
        np.sin(2 * np.pi * (1.0 + index * 0.05) * time + phase + index * 0.2)
        + 0.05 * patient_number
        for index in range(6)
    ]
    return np.column_stack((lead_i, lead_ii, lead_iii, avr, avl, avf, *chest)).astype(np.float64)


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest.csv"
    rows = []
    split_by_patient = {
        **{f"p{i}": "train" for i in range(6)},
        **{f"p{i}": "validation" for i in range(6, 9)},
        **{f"p{i}": "test" for i in range(9, 12)},
    }
    for i in range(12):
        rate = 300 if i % 2 == 0 else 500
        ecg = synthetic_ecg(rate, i)
        leads = list(CANONICAL_LEADS)
        if i % 3 == 1:
            permutation = list(reversed(range(12)))
            ecg = ecg[:, permutation]
            leads = [leads[index] for index in permutation]
        path = raw / f"r{i}.npz"
        np.savez(path, ecg=ecg, lead_names=np.asarray(leads), sampling_rate=rate, amplitude_unit="mV")
        rows.append(
            {
                "path": path.name,
                "source_format": "npz",
                "patient_id": f"p{i}",
                "record_id": f"r{i}",
                "exam_id": f"e{i}",
                "tracing_index": "1",
                "dataset_key": "ecg",
                "index": "",
                "sampling_rate": "",
                "lead_names": "",
                "axis_order": "samples_leads",
                "amplitude_unit": "mV",
                "official_split": split_by_patient[f"p{i}"],
                "dataset_part": "CODE-II-open" if i < 9 else "CODE-II-test",
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return manifest, raw

