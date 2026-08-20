from __future__ import annotations

import numpy as np
from scipy import signal


def compute_psd(
    values: np.ndarray,
    sampling_rate_hz: float,
    fmin_hz: float = 0.5,
    fmax_hz: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("PSD input must be a one-dimensional signal with at least two samples.")
    if sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be positive.")
    if not 0.0 <= fmin_hz < fmax_hz:
        raise ValueError("PSD bounds must satisfy 0 <= fmin_hz < fmax_hz.")

    frequencies, power = signal.welch(
        values,
        fs=sampling_rate_hz,
        nperseg=min(2048, values.size),
        detrend="constant",
    )
    mask = (frequencies >= fmin_hz) & (frequencies <= fmax_hz)
    if not np.any(mask):
        raise ValueError("No PSD bins fall within the requested frequency range.")
    return frequencies[mask], power[mask]
