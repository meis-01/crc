from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from .constants import CANONICAL_LEADS, normalize_target_lead


def decode_h5_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def lead_names_from_h5(handle: h5py.File) -> list[str]:
    return [decode_h5_string(value) for value in handle["lead_names"][...]]


def record_slice(handle: h5py.File, index: int) -> slice:
    offset = int(handle["metadata/offset"][index])
    length = int(handle["metadata/length"][index])
    return slice(offset, offset + length)


def load_reconstruction_sample(
    path_or_handle: str | h5py.File,
    index: int,
    target_lead: str = "V2",
) -> dict[str, Any]:
    owns_handle = not isinstance(path_or_handle, h5py.File)
    handle = h5py.File(path_or_handle, "r") if owns_handle else path_or_handle
    try:
        leads = lead_names_from_h5(handle)
        target = normalize_target_lead(target_lead)
        target_index = leads.index(target)
        input_indices = [i for i, name in enumerate(leads) if name != target]
        region = record_slice(handle, index)
        ecg = np.asarray(handle["ecg"][region, :], dtype=np.float32)
        return {
            "inputs": ecg[:, input_indices],
            "target": ecg[:, target_index],
            "input_lead_names": [leads[i] for i in input_indices],
            "target_lead": target,
            "patient_id": decode_h5_string(handle["metadata/patient_id"][index]),
            "record_id": decode_h5_string(handle["metadata/record_id"][index]),
            "sampling_rate": float(handle["metadata/sampling_rate"][index]),
            "split": decode_h5_string(handle["metadata/split"][index]),
        }
    finally:
        if owns_handle:
            handle.close()


def invert_record(handle: h5py.File, index: int, restore_baseline: bool = True) -> np.ndarray:
    region = record_slice(handle, index)
    ecg = np.asarray(handle["ecg"][region, :], dtype=np.float64)
    center = np.asarray(handle["metadata/normalization_center"][index], dtype=np.float64)
    scale = np.asarray(handle["metadata/normalization_scale"][index], dtype=np.float64)
    ecg = ecg * scale + center
    if restore_baseline:
        ecg += np.asarray(handle["metadata/baseline_offset"][index], dtype=np.float64)
    return ecg.astype(np.float32)


def expected_input_leads(target_lead: str) -> list[str]:
    target = normalize_target_lead(target_lead)
    return [name for name in CANONICAL_LEADS if name != target]

