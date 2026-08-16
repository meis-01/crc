from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from .constants import CANONICAL_LEADS, normalize_lead_name
from .io import LoadedRecord


@dataclass
class ProcessedRecord:
    ecg: np.ndarray
    sampling_rate: float
    baseline_offset: np.ndarray
    source_lead_order: list[str]


class RunningStats:
    """Stable per-lead population moments, weighted by individual samples."""

    def __init__(self, n_leads: int):
        self.count = 0
        self.mean = np.zeros(n_leads, dtype=np.float64)
        self.m2 = np.zeros(n_leads, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_mean = np.mean(values, axis=0, dtype=np.float64)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def parameters(self, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot compute normalization statistics from an empty training split")
        scale = np.sqrt(self.m2 / self.count)
        scale = np.where(scale < epsilon, 1.0, scale)
        return self.mean.astype(np.float32), scale.astype(np.float32)


def canonicalize_record(record: LoadedRecord) -> tuple[np.ndarray, list[str]]:
    normalized = [normalize_lead_name(name) for name in record.lead_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate lead names after normalization: {normalized}")
    missing = [name for name in CANONICAL_LEADS if name not in normalized]
    if missing:
        raise ValueError(f"Missing standard leads: {missing}")
    indices = [normalized.index(name) for name in CANONICAL_LEADS]
    return np.asarray(record.ecg[:, indices]), normalized


def process_record(
    record: LoadedRecord,
    target_rate: float | None,
    baseline_correction: str = "none",
) -> ProcessedRecord:
    ecg, source_order = canonicalize_record(record)
    if ecg.shape[0] < 2:
        raise ValueError("ECG has fewer than two samples")
    if not np.isfinite(record.sampling_rate) or record.sampling_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {record.sampling_rate}")
    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains NaN or infinite values")
    ecg = np.asarray(ecg, dtype=np.float64)

    if baseline_correction == "none":
        baseline = np.zeros(len(CANONICAL_LEADS), dtype=np.float64)
    elif baseline_correction == "median":
        baseline = np.median(ecg, axis=0)
        ecg = ecg - baseline
    else:
        raise ValueError("baseline_correction must be none or median")

    output_rate = float(record.sampling_rate)
    if target_rate is not None and not np.isclose(target_rate, output_rate):
        ratio = Fraction(float(target_rate) / output_rate).limit_denominator(10_000)
        ecg = resample_poly(ecg, ratio.numerator, ratio.denominator, axis=0)
        output_rate = float(target_rate)

    return ProcessedRecord(
        ecg=np.asarray(ecg, dtype=np.float32),
        sampling_rate=output_rate,
        baseline_offset=np.asarray(baseline, dtype=np.float32),
        source_lead_order=source_order,
    )


def normalization_parameters(ecg: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(ecg, axis=0, dtype=np.float64)
    scale = np.std(ecg, axis=0, dtype=np.float64)
    scale = np.where(scale < epsilon, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)

