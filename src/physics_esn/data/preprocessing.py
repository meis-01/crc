from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal


_POLYPHASE_FILTER_HALF_LENGTH_FACTOR = 10
_POLYPHASE_WINDOW = ("kaiser", 5.0)


def _as_signal(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Signal values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Signal values must be finite.")
    return array


def _validate_sampling_rates(original_rate_hz: float, target_rate_hz: float | None) -> None:
    if not np.isfinite(original_rate_hz) or original_rate_hz <= 0.0:
        raise ValueError("original_rate_hz must be finite and positive.")
    if target_rate_hz is not None and (
        not np.isfinite(target_rate_hz) or target_rate_hz <= 0.0
    ):
        raise ValueError("target_rate_hz must be finite and positive when provided.")


def _resampling_factors(
    original_rate_hz: float,
    target_rate_hz: float | None,
) -> tuple[int, int]:
    _validate_sampling_rates(original_rate_hz, target_rate_hz)
    if target_rate_hz is None or target_rate_hz == original_rate_hz:
        return 1, 1
    ratio = Fraction(float(target_rate_hz) / float(original_rate_hz)).limit_denominator(10_000)
    return ratio.numerator, ratio.denominator


def _fit_linear_trend(values: np.ndarray) -> tuple[float, float]:
    positions = np.arange(values.size, dtype=np.float64)
    centered_positions = positions - positions.mean()
    denominator = float(centered_positions @ centered_positions)
    if denominator == 0.0:
        return float(values.mean()), 0.0
    slope = float(centered_positions @ (values - values.mean()) / denominator)
    intercept = float(values.mean() - slope * positions.mean())
    return intercept, slope


def detrend_signal(values: np.ndarray) -> np.ndarray:
    values = _as_signal(values)
    intercept, slope = _fit_linear_trend(values)
    positions = np.arange(values.size, dtype=np.float64)
    return values - (intercept + slope * positions)


def zscore_signal(values: np.ndarray) -> np.ndarray:
    values = _as_signal(values)
    mean = float(values.mean())
    scale = float(values.std())
    if scale == 0:
        return values - mean
    return (values - mean) / scale


def maybe_resample(
    values: np.ndarray,
    original_rate_hz: float,
    target_rate_hz: float | None,
) -> tuple[np.ndarray, float]:
    values = _as_signal(values)
    up, down = _resampling_factors(original_rate_hz, target_rate_hz)
    if up == down == 1:
        return values, float(original_rate_hz)
    output_count = (values.size * up + down - 1) // down
    if output_count < 2:
        raise ValueError("Resampling would produce fewer than two samples.")

    # Each partition is resampled independently, so the finite FIR support can
    # reach padding at an edge but can never reach into the other partition.
    resampled = signal.resample_poly(
        values,
        up,
        down,
        window=_POLYPHASE_WINDOW,
        padtype="constant",
    )
    achieved_rate_hz = float(original_rate_hz) * up / down
    return resampled.astype(np.float64, copy=False), achieved_rate_hz


def resample_edge_guard_samples(
    original_rate_hz: float,
    target_rate_hz: float | None,
) -> int:
    """Return the output-edge width touched by resample_poly padding."""
    up, down = _resampling_factors(original_rate_hz, target_rate_hz)
    if up == down == 1:
        return 0
    half_length = _POLYPHASE_FILTER_HALF_LENGTH_FACTOR * max(up, down)
    return int(np.ceil(half_length / down))


@dataclass
class EEGPreprocessor:
    sampling_rate_hz: float
    target_rate_hz: float | None = None
    trend_intercept_: float | None = field(init=False, default=None)
    trend_slope_: float | None = field(init=False, default=None)
    normalization_mean_: float | None = field(init=False, default=None)
    normalization_std_: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        _validate_sampling_rates(self.sampling_rate_hz, self.target_rate_hz)

    @property
    def output_rate_hz(self) -> float:
        up, down = _resampling_factors(self.sampling_rate_hz, self.target_rate_hz)
        return float(self.sampling_rate_hz) * up / down

    @property
    def resample_guard_samples(self) -> int:
        return resample_edge_guard_samples(self.sampling_rate_hz, self.target_rate_hz)

    def fit(self, train_values: np.ndarray) -> EEGPreprocessor:
        values = _as_signal(train_values)
        if values.size < 2:
            raise ValueError("At least two training samples are required.")
        intercept, slope = _fit_linear_trend(values)
        positions = np.arange(values.size, dtype=np.float64)
        detrended = values - (intercept + slope * positions)
        self.trend_intercept_ = intercept
        self.trend_slope_ = slope
        self.normalization_mean_ = float(detrended.mean())
        self.normalization_std_ = float(detrended.std())
        return self

    def transform(self, values: np.ndarray, *, start_sample: int = 0) -> np.ndarray:
        values = _as_signal(values)
        if any(
            parameter is None
            for parameter in (
                self.trend_intercept_,
                self.trend_slope_,
                self.normalization_mean_,
                self.normalization_std_,
            )
        ):
            raise RuntimeError("The preprocessor must be fitted before transform().")
        if not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError("start_sample must be a non-negative integer.")

        positions = start_sample + np.arange(values.size, dtype=np.float64)
        detrended = values - (
            float(self.trend_intercept_) + float(self.trend_slope_) * positions
        )
        scale = float(self.normalization_std_)
        if scale == 0.0:
            normalized = detrended - float(self.normalization_mean_)
        else:
            normalized = (detrended - float(self.normalization_mean_)) / scale
        transformed, _ = maybe_resample(
            normalized,
            self.sampling_rate_hz,
            self.target_rate_hz,
        )
        return transformed

    def fit_transform(self, train_values: np.ndarray) -> np.ndarray:
        return self.fit(train_values).transform(train_values)


@dataclass(frozen=True)
class PreprocessedSignalSplit:
    train: np.ndarray
    test: np.ndarray
    sampling_rate_hz: float
    raw_split_index: int
    preprocessor: EEGPreprocessor


def chronological_split(
    values: np.ndarray,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = _as_signal(values)
    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and between zero and one.")
    split_index = int(values.size * train_fraction)
    train_values = values[:split_index]
    test_values = values[split_index:]
    if train_values.size < 2 or test_values.size < 2:
        raise ValueError(
            "The chronological train/test split produced a partition with fewer than two samples."
        )
    return train_values, test_values, split_index


def preprocess_chronological_split(
    values: np.ndarray,
    sampling_rate_hz: float,
    train_fraction: float,
    target_rate_hz: float | None = None,
) -> PreprocessedSignalSplit:
    raw_train, raw_test, raw_split_index = chronological_split(values, train_fraction)
    preprocessor = EEGPreprocessor(sampling_rate_hz, target_rate_hz).fit(raw_train)
    train = preprocessor.transform(raw_train)
    test = preprocessor.transform(raw_test, start_sample=raw_split_index)
    return PreprocessedSignalSplit(
        train=train,
        test=test,
        sampling_rate_hz=preprocessor.output_rate_hz,
        raw_split_index=raw_split_index,
        preprocessor=preprocessor,
    )


def preprocess_oz_signal(
    values: np.ndarray,
    sampling_rate_hz: float,
    target_rate_hz: float | None = None,
) -> tuple[np.ndarray, float]:
    """Preprocess one already-isolated signal partition.

    Chronological experiments should use ``preprocess_chronological_split`` so
    fitted parameters cannot accidentally include held-out samples.
    """
    preprocessor = EEGPreprocessor(sampling_rate_hz, target_rate_hz)
    return preprocessor.fit_transform(values), preprocessor.output_rate_hz
