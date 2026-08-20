from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy import optimize
from scipy.special import expit


@dataclass(frozen=True)
class WilsonCowanParameters:
    tau_e: float
    tau_i: float
    w_ee: float
    w_ei: float
    w_ie: float
    w_ii: float
    p: float
    q: float
    sigmoid_gain: float
    sigmoid_theta: float

    def __post_init__(self) -> None:
        values = (
            self.tau_e,
            self.tau_i,
            self.w_ee,
            self.w_ei,
            self.w_ie,
            self.w_ii,
            self.p,
            self.q,
            self.sigmoid_gain,
            self.sigmoid_theta,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Wilson-Cowan parameters must be finite.")
        if self.tau_e <= 0.0 or self.tau_i <= 0.0:
            raise ValueError("Wilson-Cowan time constants must be positive.")
        if self.sigmoid_gain <= 0.0:
            raise ValueError("Sigmoid gain must be positive.")


def sigmoid(x: np.ndarray | float, gain: float, theta: float) -> np.ndarray | float:
    return expit(gain * (np.asarray(x) - theta))


def sigmoid_derivative(x: np.ndarray | float, gain: float, theta: float) -> np.ndarray | float:
    s = sigmoid(x, gain, theta)
    return gain * s * (1.0 - s)


def wilson_cowan_rhs(state: np.ndarray, params: WilsonCowanParameters) -> np.ndarray:
    e, i = np.asarray(state, dtype=np.float64)
    input_e = params.w_ee * e - params.w_ei * i + params.p
    input_i = params.w_ie * e - params.w_ii * i + params.q
    de_dt = (-e + sigmoid(input_e, params.sigmoid_gain, params.sigmoid_theta)) / params.tau_e
    di_dt = (-i + sigmoid(input_i, params.sigmoid_gain, params.sigmoid_theta)) / params.tau_i
    return np.array([de_dt, di_dt], dtype=np.float64)


def _rhs_components(e: float, i: float, params: WilsonCowanParameters) -> tuple[float, float]:
    input_e = params.w_ee * e - params.w_ei * i + params.p
    input_i = params.w_ie * e - params.w_ii * i + params.q
    de_dt = (-e + float(sigmoid(input_e, params.sigmoid_gain, params.sigmoid_theta))) / params.tau_e
    di_dt = (-i + float(sigmoid(input_i, params.sigmoid_gain, params.sigmoid_theta))) / params.tau_i
    return de_dt, di_dt


def simulate_wilson_cowan(
    params: WilsonCowanParameters,
    dt: float,
    duration_s: float,
    initial_state: tuple[float, float] = (0.1, 0.1),
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0 or duration_s <= 0.0:
        raise ValueError("dt and duration_s must be positive.")
    if len(initial_state) != 2 or not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain two finite values.")

    step_count = int(round(duration_s / dt))
    if step_count < 1:
        raise ValueError("duration_s must span at least one integration step.")
    time = np.arange(step_count + 1, dtype=np.float64) * dt
    states = np.empty((time.size, 2), dtype=np.float64)
    states[0] = np.asarray(initial_state, dtype=np.float64)
    for index in range(1, time.size):
        e, i = states[index - 1]
        k1_e, k1_i = _rhs_components(e, i, params)
        k2_e, k2_i = _rhs_components(e + 0.5 * dt * k1_e, i + 0.5 * dt * k1_i, params)
        k3_e, k3_i = _rhs_components(e + 0.5 * dt * k2_e, i + 0.5 * dt * k2_i, params)
        k4_e, k4_i = _rhs_components(e + dt * k3_e, i + dt * k3_i, params)
        next_e = e + (dt / 6.0) * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e)
        next_i = i + (dt / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
        states[index] = np.clip((next_e, next_i), 0.0, 1.0)
    return time, states


def find_equilibrium(
    params: WilsonCowanParameters,
    initial_guess: tuple[float, float] = (0.2, 0.1),
) -> np.ndarray:
    result = optimize.least_squares(
        lambda state: wilson_cowan_rhs(state, params),
        np.asarray(initial_guess, dtype=np.float64),
        bounds=(np.zeros(2), np.ones(2)),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    residual_norm = np.linalg.norm(wilson_cowan_rhs(result.x, params))
    if not result.success or residual_norm > 1.0e-7:
        raise RuntimeError(f"Failed to find Wilson-Cowan equilibrium: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def jacobian_at_equilibrium(equilibrium: np.ndarray, params: WilsonCowanParameters) -> np.ndarray:
    e, i = np.asarray(equilibrium, dtype=np.float64)
    input_e = params.w_ee * e - params.w_ei * i + params.p
    input_i = params.w_ie * e - params.w_ii * i + params.q
    s_e_prime = float(sigmoid_derivative(input_e, params.sigmoid_gain, params.sigmoid_theta))
    s_i_prime = float(sigmoid_derivative(input_i, params.sigmoid_gain, params.sigmoid_theta))
    return np.array(
        [
            [(-1.0 + s_e_prime * params.w_ee) / params.tau_e, (-s_e_prime * params.w_ei) / params.tau_e],
            [(s_i_prime * params.w_ie) / params.tau_i, (-1.0 - s_i_prime * params.w_ii) / params.tau_i],
        ],
        dtype=np.float64,
    )


def continuous_eigenvalues(jacobian: np.ndarray) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("The Wilson-Cowan Jacobian must be a finite 2x2 matrix.")
    return np.linalg.eigvals(matrix)


def discrete_reservoir_eigenvalues(lambdas: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return np.exp(np.asarray(lambdas, dtype=np.complex128) * dt)
