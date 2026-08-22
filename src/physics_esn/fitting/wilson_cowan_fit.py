from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path

import numpy as np
from scipy import optimize, signal

from physics_esn.analysis.spectrum import compute_psd
from physics_esn.models.wilson_cowan import (
    WILSON_COWAN_PARAMETER_BOUNDS,
    WilsonCowanParameters,
    continuous_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    simulate_wilson_cowan,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WilsonCowanFitResult:
    parameters: WilsonCowanParameters
    objective: float
    evaluations: int
    optimizer_success: bool
    optimizer_message: str
    loss_components: dict[str, float] = field(default_factory=dict)


def _log_spectral_shape(power: np.ndarray) -> np.ndarray:
    log_power = np.log10(np.maximum(np.asarray(power, dtype=np.float64), 1.0e-15))
    return log_power - log_power.mean()


def _validate_hybrid_loss_options(
    psd_weight: float,
    stft_weight: float,
    temporal_weight: float,
    stft_window_seconds: float,
    stft_overlap_fraction: float,
) -> None:
    weights = np.asarray(
        [psd_weight, stft_weight, temporal_weight],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Hybrid loss weights must be finite and non-negative.")
    if not np.any(weights > 0.0):
        raise ValueError("At least one hybrid loss weight must be positive.")
    if not np.isfinite(stft_window_seconds) or stft_window_seconds <= 0.0:
        raise ValueError("stft_window_seconds must be finite and positive.")
    if not np.isfinite(stft_overlap_fraction) or not 0.0 <= stft_overlap_fraction < 1.0:
        raise ValueError("stft_overlap_fraction must be finite and in [0, 1).")


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean()
    scale = float(centered.std())
    return centered if scale == 0.0 else centered / scale


def _normalized_log_magnitude_stft(
    values: np.ndarray,
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_fraction: float,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("STFT input must be a finite one-dimensional signal.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive.")

    window_samples = int(round(window_seconds * sampling_rate_hz))
    if window_samples < 2:
        raise ValueError("The STFT window must span at least two samples.")
    if window_samples > array.size:
        raise ValueError("The STFT window cannot exceed the equal-duration signal segment.")
    overlap_samples = int(np.floor(overlap_fraction * window_samples))
    frequencies, times, coefficients = signal.stft(
        array,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=window_samples,
        noverlap=overlap_samples,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    frequency_mask = (frequencies >= fmin_hz) & (frequencies <= fmax_hz)
    if not np.any(frequency_mask):
        raise ValueError("No STFT bins fall within the requested frequency range.")
    log_magnitude = np.log10(
        np.maximum(np.abs(coefficients[frequency_mask]), 1.0e-15)
    )
    return frequencies[frequency_mask], times, _standardize(log_magnitude)


def _interpolated_stft_mse(
    observed: tuple[np.ndarray, np.ndarray, np.ndarray],
    simulated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    obs_frequencies, obs_times, obs_values = observed
    sim_frequencies, sim_times, sim_values = simulated
    minimum_frequency = max(float(obs_frequencies[0]), float(sim_frequencies[0]))
    maximum_frequency = min(float(obs_frequencies[-1]), float(sim_frequencies[-1]))
    frequency_mask = (obs_frequencies >= minimum_frequency) & (
        obs_frequencies <= maximum_frequency
    )
    minimum_time = max(float(obs_times[0]), float(sim_times[0]))
    maximum_time = min(float(obs_times[-1]), float(sim_times[-1]))
    time_mask = (obs_times >= minimum_time) & (obs_times <= maximum_time)
    if not np.any(frequency_mask) or not np.any(time_mask):
        raise ValueError("Observed and simulated STFT grids do not overlap.")

    target_frequencies = obs_frequencies[frequency_mask]
    target_times = obs_times[time_mask]
    observed_grid = obs_values[frequency_mask][:, time_mask]
    simulated_on_frequencies = np.empty(
        (target_frequencies.size, sim_times.size),
        dtype=np.float64,
    )
    for time_index in range(sim_times.size):
        simulated_on_frequencies[:, time_index] = np.interp(
            target_frequencies,
            sim_frequencies,
            sim_values[:, time_index],
        )
    simulated_grid = np.empty_like(observed_grid)
    for frequency_index in range(target_frequencies.size):
        simulated_grid[frequency_index] = np.interp(
            target_times,
            sim_times,
            simulated_on_frequencies[frequency_index],
        )
    return float(np.mean((simulated_grid - observed_grid) ** 2))


def _normalized_aligned_temporal_mse(
    observed_values: np.ndarray,
    observed_rate_hz: float,
    simulated_values: np.ndarray,
    simulated_dt: float,
) -> float:
    observed = np.asarray(observed_values, dtype=np.float64)
    simulated = np.asarray(simulated_values, dtype=np.float64)
    observed_times = np.arange(observed.size, dtype=np.float64) / observed_rate_hz
    simulated_times = np.arange(simulated.size, dtype=np.float64) * simulated_dt
    common_end = min(float(observed_times[-1]), float(simulated_times[-1]))
    observed_mask = observed_times <= common_end + 1.0e-12
    observed_aligned = _standardize(observed[observed_mask])
    simulated_aligned = np.interp(
        observed_times[observed_mask],
        simulated_times,
        _standardize(simulated),
    )
    return float(np.mean((simulated_aligned - observed_aligned) ** 2))


def _equal_duration_observed_segment(
    observed_signal: np.ndarray,
    sampling_rate_hz: float,
    duration_s: float,
    dt: float,
) -> np.ndarray:
    effective_duration_s = int(round(duration_s / dt)) * dt
    sample_count = int(round(effective_duration_s * sampling_rate_hz)) + 1
    if observed_signal.size < sample_count:
        raise ValueError(
            "The observed training signal is shorter than the configured hybrid-loss duration."
        )
    return np.asarray(observed_signal[:sample_count], dtype=np.float64)


def _vector_to_params(vector: np.ndarray, template: WilsonCowanParameters) -> WilsonCowanParameters:
    return WilsonCowanParameters(
        tau_e=float(vector[0]),
        tau_i=float(vector[1]),
        w_ee=float(vector[2]),
        w_ei=float(vector[3]),
        w_ie=float(vector[4]),
        w_ii=float(vector[5]),
        p=float(vector[6]),
        q=float(vector[7]),
        sigmoid_gain=template.sigmoid_gain,
        sigmoid_theta=template.sigmoid_theta,
    )


def fit_wilson_cowan_psd(
    observed_signal: np.ndarray,
    sampling_rate_hz: float,
    initial_params: WilsonCowanParameters,
    dt: float,
    duration_s: float,
    maxiter: int = 20,
    population_size: int = 8,
    polish: bool = False,
    random_seed: int = 0,
    fmin_hz: float = 0.5,
    fmax_hz: float = 45.0,
    psd_weight: float = 1.0,
    stft_weight: float = 0.0,
    temporal_weight: float = 0.0,
    stft_window_seconds: float = 1.0,
    stft_overlap_fraction: float = 0.5,
) -> WilsonCowanFitResult:
    if maxiter < 0 or population_size < 1:
        raise ValueError("maxiter must be non-negative and population_size must be positive.")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("dt and duration_s must be finite and positive.")
    _validate_hybrid_loss_options(
        psd_weight,
        stft_weight,
        temporal_weight,
        stft_window_seconds,
        stft_overlap_fraction,
    )
    observed = np.asarray(observed_signal, dtype=np.float64)
    obs_freqs: np.ndarray | None = None
    obs_log_power: np.ndarray | None = None
    if psd_weight > 0.0:
        obs_freqs, obs_power = compute_psd(
            observed,
            sampling_rate_hz,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )
        obs_log_power = _log_spectral_shape(obs_power)

    observed_segment: np.ndarray | None = None
    observed_stft: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if stft_weight > 0.0 or temporal_weight > 0.0:
        observed_segment = _equal_duration_observed_segment(
            observed,
            sampling_rate_hz,
            duration_s,
            dt,
        )
    if stft_weight > 0.0:
        assert observed_segment is not None
        observed_stft = _normalized_log_magnitude_stft(
            observed_segment,
            sampling_rate_hz,
            stft_window_seconds,
            stft_overlap_fraction,
            fmin_hz,
            fmax_hz,
        )
    initial = np.array(
        [
            initial_params.tau_e,
            initial_params.tau_i,
            initial_params.w_ee,
            initial_params.w_ei,
            initial_params.w_ie,
            initial_params.w_ii,
            initial_params.p,
            initial_params.q,
        ],
        dtype=np.float64,
    )
    bounds = [WILSON_COWAN_PARAMETER_BOUNDS[name] for name in (
        "tau_e", "tau_i", "w_ee", "w_ei", "w_ie", "w_ii", "p", "q"
    )]
    bounds[0] = (max(dt, bounds[0][0]), bounds[0][1])
    bounds[1] = (max(dt, bounds[1][0]), bounds[1][1])
    if bounds[0][0] > bounds[0][1] or bounds[1][0] > bounds[1][1]:
        raise ValueError("dt exceeds the supported Wilson-Cowan time-constant bounds.")

    def evaluate_objective(vector: np.ndarray) -> tuple[float, dict[str, float]]:
        params = _vector_to_params(vector, initial_params)
        try:
            equilibrium = find_equilibrium(params)
            lambdas = continuous_eigenvalues(jacobian_at_equilibrium(equilibrium, params))
        except (RuntimeError, ValueError, FloatingPointError):
            penalty = 1.0e12
            return penalty, {"penalty": penalty, "weighted_total": penalty}
        max_real_part = float(np.max(np.real(lambdas)))
        if max_real_part >= 0.0:
            penalty = 1.0e9 + max_real_part**2
            return penalty, {"penalty": penalty, "weighted_total": penalty}

        _, states = simulate_wilson_cowan(params, dt=dt, duration_s=duration_s)
        simulated = states[:, 0]
        if not np.all(np.isfinite(simulated)):
            penalty = 1.0e12
            return penalty, {"penalty": penalty, "weighted_total": penalty}

        psd_loss = 0.0
        if psd_weight > 0.0:
            assert obs_freqs is not None and obs_log_power is not None
            sim_rate = 1.0 / dt
            sim_freqs, sim_power = compute_psd(
                simulated,
                sim_rate,
                fmin_hz=float(obs_freqs.min()),
                fmax_hz=float(obs_freqs.max()),
            )
            aligned = np.interp(obs_freqs, sim_freqs, _log_spectral_shape(sim_power))
            psd_loss = float(np.mean((aligned - obs_log_power) ** 2))

        stft_loss = 0.0
        if stft_weight > 0.0:
            assert observed_stft is not None
            simulated_stft = _normalized_log_magnitude_stft(
                simulated,
                1.0 / dt,
                stft_window_seconds,
                stft_overlap_fraction,
                fmin_hz,
                fmax_hz,
            )
            stft_loss = _interpolated_stft_mse(observed_stft, simulated_stft)

        temporal_loss = 0.0
        if temporal_weight > 0.0:
            assert observed_segment is not None
            temporal_loss = _normalized_aligned_temporal_mse(
                observed_segment,
                sampling_rate_hz,
                simulated,
                dt,
            )

        # Keep the legacy default path numerically identical rather than routing
        # it through a generalized weighted reduction.
        if psd_weight == 1.0 and stft_weight == 0.0 and temporal_weight == 0.0:
            total = psd_loss
        else:
            total = (
                psd_weight * psd_loss
                + stft_weight * stft_loss
                + temporal_weight * temporal_loss
            )
        return total, {
            "psd": psd_loss,
            "stft": stft_loss,
            "temporal": temporal_loss,
            "weighted_total": float(total),
        }

    def objective(vector: np.ndarray) -> float:
        return evaluate_objective(vector)[0]

    result = optimize.differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=population_size,
        polish=polish,
        x0=initial,
        seed=random_seed,
        updating="deferred",
        workers=1,
    )
    if not np.isfinite(result.fun):
        raise RuntimeError("Wilson-Cowan fitting did not produce a finite objective.")
    _, loss_components = evaluate_objective(result.x)
    LOGGER.info("Wilson-Cowan fit completed after %d evaluations (loss %.6g).", result.nfev, result.fun)
    return WilsonCowanFitResult(
        parameters=_vector_to_params(result.x, initial_params),
        objective=float(result.fun),
        evaluations=int(result.nfev),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        loss_components=loss_components,
    )


def save_subject_fit(
    path: str | Path,
    subject_id: str,
    fit: WilsonCowanFitResult,
    equilibrium: np.ndarray,
    jacobian: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
) -> None:
    payload = {
        "subject_id": subject_id,
        "parameters": asdict(fit.parameters),
        "fit": {
            "objective": fit.objective,
            "evaluations": fit.evaluations,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "loss_components": fit.loss_components,
        },
        "equilibrium": {"E": float(equilibrium[0]), "I": float(equilibrium[1])},
        "jacobian": np.asarray(jacobian, dtype=np.float64).tolist(),
        "continuous_eigenvalues": [[float(value.real), float(value.imag)] for value in np.asarray(lambdas)],
        "discrete_eigenvalues": [[float(value.real), float(value.imag)] for value in np.asarray(mus)],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
