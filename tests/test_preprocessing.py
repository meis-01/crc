from __future__ import annotations

import numpy as np

from physics_esn.data.preprocessing import (
    EEGPreprocessor,
    chronological_split,
    preprocess_chronological_split,
)


def _recording_with_test_offset(test_offset: float = 0.0) -> np.ndarray:
    positions = np.arange(200, dtype=np.float64)
    values = 3.0 + 0.02 * positions + np.sin(positions / 5.0)
    values[150:] += test_offset
    return values


def test_test_values_do_not_affect_transformed_training_partition() -> None:
    baseline = preprocess_chronological_split(
        _recording_with_test_offset(),
        sampling_rate_hz=100.0,
        train_fraction=0.75,
        target_rate_hz=50.0,
    )
    changed_test = preprocess_chronological_split(
        _recording_with_test_offset(test_offset=10_000.0),
        sampling_rate_hz=100.0,
        train_fraction=0.75,
        target_rate_hz=50.0,
    )

    assert np.array_equal(baseline.train, changed_test.train)
    assert baseline.preprocessor.trend_intercept_ == changed_test.preprocessor.trend_intercept_
    assert baseline.preprocessor.trend_slope_ == changed_test.preprocessor.trend_slope_
    assert baseline.preprocessor.normalization_mean_ == changed_test.preprocessor.normalization_mean_
    assert baseline.preprocessor.normalization_std_ == changed_test.preprocessor.normalization_std_


def test_detrending_and_normalization_are_fitted_on_training_only() -> None:
    train = _recording_with_test_offset()[:150]
    preprocessor = EEGPreprocessor(sampling_rate_hz=100.0).fit(train)
    expected_slope, expected_intercept = np.polyfit(np.arange(train.size), train, 1)
    transformed_train = preprocessor.transform(train)

    assert np.isclose(preprocessor.trend_intercept_, expected_intercept)
    assert np.isclose(preprocessor.trend_slope_, expected_slope)
    assert np.isclose(transformed_train.mean(), 0.0, atol=1.0e-14)
    assert np.isclose(transformed_train.std(), 1.0)


def test_training_parameters_are_reused_with_absolute_test_positions() -> None:
    recording = _recording_with_test_offset(test_offset=20.0)
    train, test, split_index = chronological_split(recording, train_fraction=0.75)
    preprocessor = EEGPreprocessor(sampling_rate_hz=100.0).fit(train)
    transformed_test = preprocessor.transform(test, start_sample=split_index)

    positions = split_index + np.arange(test.size, dtype=np.float64)
    expected = test - (
        float(preprocessor.trend_intercept_) + float(preprocessor.trend_slope_) * positions
    )
    expected = (expected - float(preprocessor.normalization_mean_)) / float(
        preprocessor.normalization_std_
    )

    assert np.allclose(transformed_test, expected)
    assert not np.isclose(transformed_test.mean(), 0.0)


def test_chronological_split_happens_on_raw_samples() -> None:
    values = np.arange(10, dtype=np.float64)
    train, test, split_index = chronological_split(values, train_fraction=0.6)

    assert split_index == 6
    assert np.array_equal(train, values[:6])
    assert np.array_equal(test, values[6:])
