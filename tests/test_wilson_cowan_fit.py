from __future__ import annotations

import numpy as np

from physics_esn.fitting.wilson_cowan_fit import (
    _interpolated_stft_mse,
    _normalized_log_magnitude_stft,
    fit_wilson_cowan_psd,
)
from physics_esn.models.wilson_cowan import WilsonCowanParameters


def _params() -> WilsonCowanParameters:
    return WilsonCowanParameters(
        tau_e=0.01,
        tau_i=0.02,
        w_ee=10.0,
        w_ei=12.0,
        w_ie=10.0,
        w_ii=0.0,
        p=0.5,
        q=0.0,
        sigmoid_gain=1.5,
        sigmoid_theta=2.5,
    )


def test_zero_optional_loss_weights_reproduce_default_psd_fit() -> None:
    rate = 250.0
    time = np.arange(250, dtype=np.float64) / rate
    observed = np.sin(2.0 * np.pi * 10.0 * time)
    common = {
        "observed_signal": observed,
        "sampling_rate_hz": rate,
        "initial_params": _params(),
        "dt": 0.004,
        "duration_s": 0.2,
        "maxiter": 0,
        "population_size": 1,
        "polish": False,
        "random_seed": 17,
        "fmin_hz": 0.5,
        "fmax_hz": 45.0,
    }

    default = fit_wilson_cowan_psd(**common)
    explicit = fit_wilson_cowan_psd(
        **common,
        psd_weight=1.0,
        stft_weight=0.0,
        temporal_weight=0.0,
    )

    assert default.objective == explicit.objective
    assert default.parameters == explicit.parameters
    assert default.loss_components == explicit.loss_components


def test_log_stft_loss_is_finite_and_sensitive_to_local_spectral_order() -> None:
    rate = 200.0
    half_time = np.arange(200, dtype=np.float64) / rate
    low_then_high = np.concatenate(
        (
            np.sin(2.0 * np.pi * 5.0 * half_time),
            np.sin(2.0 * np.pi * 20.0 * half_time),
        )
    )
    high_then_low = np.concatenate(
        (
            np.sin(2.0 * np.pi * 20.0 * half_time),
            np.sin(2.0 * np.pi * 5.0 * half_time),
        )
    )
    observed = _normalized_log_magnitude_stft(
        low_then_high,
        sampling_rate_hz=rate,
        window_seconds=0.5,
        overlap_fraction=0.5,
        fmin_hz=1.0,
        fmax_hz=40.0,
    )
    reordered = _normalized_log_magnitude_stft(
        high_then_low,
        sampling_rate_hz=rate,
        window_seconds=0.5,
        overlap_fraction=0.5,
        fmin_hz=1.0,
        fmax_hz=40.0,
    )

    identical_loss = _interpolated_stft_mse(observed, observed)
    reordered_loss = _interpolated_stft_mse(observed, reordered)
    assert identical_loss == 0.0
    assert np.isfinite(reordered_loss)
    assert reordered_loss > 0.0


def test_enabled_stft_weight_runs_through_hybrid_fit() -> None:
    rate = 250.0
    time = np.arange(250, dtype=np.float64) / rate
    observed = np.sin(2.0 * np.pi * 10.0 * time)

    result = fit_wilson_cowan_psd(
        observed_signal=observed,
        sampling_rate_hz=rate,
        initial_params=_params(),
        dt=0.004,
        duration_s=0.2,
        maxiter=0,
        population_size=1,
        polish=False,
        random_seed=4,
        fmin_hz=0.5,
        fmax_hz=45.0,
        psd_weight=1.0,
        stft_weight=0.25,
        temporal_weight=0.0,
        stft_window_seconds=0.1,
        stft_overlap_fraction=0.5,
    )

    assert np.isfinite(result.objective)
    assert np.isfinite(result.loss_components["psd"])
    assert np.isfinite(result.loss_components["stft"])
    assert result.loss_components["weighted_total"] == result.objective
