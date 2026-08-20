from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np
from scipy import optimize

from physics_esn.analysis.spectrum import compute_psd
from physics_esn.models.wilson_cowan import (
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


def _log_spectral_shape(power: np.ndarray) -> np.ndarray:
    log_power = np.log10(np.maximum(np.asarray(power, dtype=np.float64), 1.0e-15))
    return log_power - log_power.mean()


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
) -> WilsonCowanFitResult:
    if maxiter < 0 or population_size < 1:
        raise ValueError("maxiter must be non-negative and population_size must be positive.")
    obs_freqs, obs_power = compute_psd(
        observed_signal,
        sampling_rate_hz,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )
    obs_log_power = _log_spectral_shape(obs_power)
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
    bounds = [
        (max(dt, 0.002), 0.1),
        (max(dt, 0.002), 0.1),
        (0.0, 20.0),
        (0.0, 20.0),
        (0.0, 20.0),
        (0.0, 20.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
    ]

    def objective(vector: np.ndarray) -> float:
        params = _vector_to_params(vector, initial_params)
        try:
            equilibrium = find_equilibrium(params)
            lambdas = continuous_eigenvalues(jacobian_at_equilibrium(equilibrium, params))
        except (RuntimeError, ValueError, FloatingPointError):
            return 1.0e12
        max_real_part = float(np.max(np.real(lambdas)))
        if max_real_part >= 0.0:
            return 1.0e9 + max_real_part**2

        _, states = simulate_wilson_cowan(params, dt=dt, duration_s=duration_s)
        simulated = states[:, 0]
        if not np.all(np.isfinite(simulated)):
            return 1.0e12
        sim_rate = 1.0 / dt
        sim_freqs, sim_power = compute_psd(
            simulated,
            sim_rate,
            fmin_hz=float(obs_freqs.min()),
            fmax_hz=float(obs_freqs.max()),
        )
        aligned = np.interp(obs_freqs, sim_freqs, _log_spectral_shape(sim_power))
        return float(np.mean((aligned - obs_log_power) ** 2))

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
    LOGGER.info("Wilson-Cowan fit completed after %d evaluations (loss %.6g).", result.nfev, result.fun)
    return WilsonCowanFitResult(
        parameters=_vector_to_params(result.x, initial_params),
        objective=float(result.fun),
        evaluations=int(result.nfev),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
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
        },
        "equilibrium": {"E": float(equilibrium[0]), "I": float(equilibrium[1])},
        "jacobian": np.asarray(jacobian, dtype=np.float64).tolist(),
        "continuous_eigenvalues": [[float(value.real), float(value.imag)] for value in np.asarray(lambdas)],
        "discrete_eigenvalues": [[float(value.real), float(value.imag)] for value in np.asarray(mus)],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
