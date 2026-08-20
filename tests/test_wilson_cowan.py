from __future__ import annotations

import numpy as np

from physics_esn.models.wilson_cowan import (
    WilsonCowanParameters,
    continuous_eigenvalues,
    discrete_reservoir_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    simulate_wilson_cowan,
    wilson_cowan_rhs,
)


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


def test_equilibrium_is_numerical_fixed_point() -> None:
    equilibrium = find_equilibrium(_params())
    residual = wilson_cowan_rhs(equilibrium, _params())
    assert np.all((0.0 <= equilibrium) & (equilibrium <= 1.0))
    assert np.linalg.norm(residual) < 1.0e-8


def test_jacobian_matches_finite_difference() -> None:
    params = _params()
    equilibrium = find_equilibrium(params)
    analytic = jacobian_at_equilibrium(equilibrium, params)
    epsilon = 1.0e-6
    finite = np.empty((2, 2), dtype=np.float64)
    for column in range(2):
        delta = np.zeros(2, dtype=np.float64)
        delta[column] = epsilon
        finite[:, column] = (
            wilson_cowan_rhs(equilibrium + delta, params) - wilson_cowan_rhs(equilibrium - delta, params)
        ) / (2.0 * epsilon)
    assert np.allclose(analytic, finite, atol=1.0e-4)


def test_eigenvalues_match_numpy_calculation() -> None:
    params = _params()
    equilibrium = find_equilibrium(params)
    jacobian = jacobian_at_equilibrium(equilibrium, params)
    eigenvalues = continuous_eigenvalues(jacobian)
    assert eigenvalues.shape == (2,)
    assert np.allclose(np.sort_complex(eigenvalues), np.sort_complex(np.linalg.eigvals(jacobian)))


def test_discrete_mapping_matches_complex_exponential() -> None:
    lambdas = np.array([-1.0 + 2.0j, -0.5 - 0.25j])
    dt = 0.1
    mapped = discrete_reservoir_eigenvalues(lambdas, dt)
    assert np.allclose(mapped, np.exp(lambdas * dt))


def test_simulation_returns_bounded_population_activity() -> None:
    time, states = simulate_wilson_cowan(_params(), dt=0.001, duration_s=0.1)
    assert time.shape == (101,)
    assert states.shape == (101, 2)
    assert np.all((0.0 <= states) & (states <= 1.0))
