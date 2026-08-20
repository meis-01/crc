from __future__ import annotations

import numpy as np
from scipy import signal


def detrend_signal(values: np.ndarray) -> np.ndarray:
    return signal.detrend(np.asarray(values, dtype=np.float64), type="linear")


def zscore_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    if scale == 0:
        return values - values.mean()
    return (values - values.mean()) / scale


def maybe_resample(
    values: np.ndarray,
    original_rate_hz: float,
    target_rate_hz: float | None,
) -> tuple[np.ndarray, float]:
    if original_rate_hz <= 0.0:
        raise ValueError("original_rate_hz must be positive.")
    if target_rate_hz is not None and target_rate_hz <= 0.0:
        raise ValueError("target_rate_hz must be positive when provided.")
    if target_rate_hz is None or target_rate_hz == original_rate_hz:
        return np.asarray(values, dtype=np.float64), float(original_rate_hz)
    sample_count = int(round(len(values) * target_rate_hz / original_rate_hz))
    if sample_count < 2:
        raise ValueError("Resampling would produce fewer than two samples.")
    resampled = signal.resample(np.asarray(values, dtype=np.float64), sample_count)
    return resampled.astype(np.float64, copy=False), float(target_rate_hz)


def preprocess_oz_signal(
    values: np.ndarray,
    sampling_rate_hz: float,
    target_rate_hz: float | None = None,
) -> tuple[np.ndarray, float]:
    detrended = detrend_signal(values)
    normalized = zscore_signal(detrended)
    return maybe_resample(normalized, sampling_rate_hz, target_rate_hz)
