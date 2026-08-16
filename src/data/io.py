from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class ManifestRow:
    path: str
    source_format: str
    patient_id: str
    record_id: str
    exam_id: str = ""
    tracing_index: str = ""
    dataset_key: str = ""
    index: str = ""
    sampling_rate: str = ""
    lead_names: str = ""
    axis_order: str = "auto"
    amplitude_unit: str = ""
    official_split: str = ""
    dataset_part: str = ""


@dataclass
class LoadedRecord:
    ecg: np.ndarray  # samples x leads
    lead_names: list[str]
    sampling_rate: float
    amplitude_unit: str


MANIFEST_COLUMNS = tuple(ManifestRow.__dataclass_fields__)


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "patient_id", "record_id"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            values = {name: (raw.get(name) or "").strip() for name in MANIFEST_COLUMNS}
            if not values["source_format"]:
                values["source_format"] = Path(values["path"]).suffix.lstrip(".").lower()
            if not values["patient_id"] or not values["record_id"]:
                raise ValueError(f"Empty patient_id or record_id at manifest line {line_number}")
            rows.append(ManifestRow(**values))
    if not rows:
        raise ValueError("Manifest contains no records")
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any] | ManifestRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            data = row.__dict__ if isinstance(row, ManifestRow) else row
            writer.writerow({name: data.get(name, "") for name in MANIFEST_COLUMNS})


def parse_lead_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_decode_scalar(item) for item in value]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    separator = "|" if "|" in text else ("," if "," in text else ";")
    return [item.strip() for item in text.split(separator) if item.strip()]


def resolve_record_path(row: ManifestRow, raw_root: str | Path | None) -> Path:
    path = Path(row.path)
    if not path.is_absolute():
        if raw_root is None:
            raise ValueError(f"Relative path requires --raw-root: {row.path}")
        path = Path(raw_root) / path
    return path.resolve()


def load_record(row: ManifestRow, raw_root: str | Path | None = None) -> LoadedRecord:
    path = resolve_record_path(row, raw_root)
    fmt = row.source_format.lower().replace(".", "")
    requested_leads = parse_lead_names(row.lead_names)
    rate = _optional_float(row.sampling_rate)
    unit = row.amplitude_unit

    if fmt in {"h5", "hdf5"}:
        with h5py.File(path, "r") as handle:
            key = row.dataset_key or _find_hdf5_dataset(handle)
            dataset = handle[key]
            index = _optional_int(row.index)
            array = dataset[index] if index is not None else dataset[...]
            leads = requested_leads or parse_lead_names(
                dataset.attrs.get("lead_names", handle.attrs.get("lead_names"))
            )
            if rate is None:
                rate = _optional_float(
                    dataset.attrs.get("sampling_rate", handle.attrs.get("sampling_rate"))
                )
            if not unit:
                unit = _decode_scalar(
                    dataset.attrs.get("amplitude_unit", handle.attrs.get("amplitude_unit", ""))
                )
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as archive:
            key = row.dataset_key or ("ecg" if "ecg" in archive.files else archive.files[0])
            array = archive[key]
            index = _optional_int(row.index)
            if index is not None:
                array = array[index]
            leads = requested_leads or parse_lead_names(archive.get("lead_names"))
            if rate is None and "sampling_rate" in archive:
                rate = float(np.asarray(archive["sampling_rate"]).reshape(-1)[0])
            if not unit and "amplitude_unit" in archive:
                unit = _decode_scalar(np.asarray(archive["amplitude_unit"]).reshape(-1)[0])
    elif fmt == "npy":
        array = np.load(path, allow_pickle=False)
        index = _optional_int(row.index)
        if index is not None:
            array = array[index]
        leads = requested_leads
    elif fmt == "wfdb":
        try:
            import wfdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("WFDB input requires `pip install wfdb`") from exc
        stem = str(path.with_suffix(""))
        signal, fields = wfdb.rdsamp(stem)
        array = signal
        leads = requested_leads or [str(name) for name in fields["sig_name"]]
        if rate is None:
            rate = float(fields["fs"])
        if not unit:
            units = fields.get("units") or []
            unit = str(units[0]) if units and len(set(units)) == 1 else "|".join(map(str, units))
    else:
        raise ValueError(f"Unsupported source_format {row.source_format!r} for {path}")

    if rate is None:
        raise ValueError(f"Sampling rate is absent for record {row.record_id}")
    if not leads:
        raise ValueError(f"Lead names are absent for record {row.record_id}")
    ecg = _orient_2d(np.asarray(array), leads, row.axis_order)
    return LoadedRecord(ecg=ecg, lead_names=leads, sampling_rate=float(rate), amplitude_unit=unit)


def _orient_2d(array: np.ndarray, leads: list[str], axis_order: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"A record must be two-dimensional, got shape {array.shape}")
    order = (axis_order or "auto").lower().replace("records_", "")
    if order in {"samples_leads", "sample_lead"}:
        oriented = array
    elif order in {"leads_samples", "lead_sample"}:
        oriented = array.T
    elif order == "auto":
        lead_axes = [axis for axis, size in enumerate(array.shape) if size == len(leads)]
        if len(lead_axes) != 1:
            raise ValueError(
                f"Cannot infer lead axis for shape {array.shape} and {len(leads)} lead names; "
                "set axis_order explicitly"
            )
        oriented = array if lead_axes[0] == 1 else array.T
    else:
        raise ValueError(f"Unknown axis_order: {axis_order!r}")
    if oriented.shape[1] != len(leads):
        raise ValueError(
            f"Lead-name count {len(leads)} does not match oriented ECG shape {oriented.shape}"
        )
    return oriented


def _find_hdf5_dataset(handle: h5py.File) -> str:
    for key in ("ecg", "tracings", "signals"):
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return key
    datasets: list[str] = []
    handle.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
    if len(datasets) != 1:
        raise ValueError(f"Set dataset_key explicitly; HDF5 datasets found: {datasets}")
    return datasets[0]


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

