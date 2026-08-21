from __future__ import annotations

import numpy as np
import pytest

from physics_esn.models.physics_reservoir import (
    build_physics_informed_reservoir,
    generate_continuous_reservoir_modes,
)
from physics_esn.models.wilson_cowan import discrete_reservoir_eigenvalues


CENTERS = np.array([-2.0 + 8.0j, -2.0 - 8.0j])


def _sample_modes(seed: int, size: int = 12) -> np.ndarray:
    return generate_continuous_reservoir_modes(
        CENTERS,
        reservoir_mode="gaussian_eigenvalue_cloud",
        reservoir_size=size,
        eigenvalue_sigma_real=0.4,
        eigenvalue_sigma_imag=0.7,
        seed=seed,
    )


def test_eigenvalue_cloud_has_requested_size_and_stable_discrete_modes() -> None:
    modes = _sample_modes(seed=13, size=20)
    discrete_modes = discrete_reservoir_eigenvalues(modes, dt=0.01)

    assert modes.shape == (20,)
    assert np.all(modes.real < 0.0)
    assert np.all(np.abs(discrete_modes) < 1.0)
    assert np.allclose(discrete_modes, np.exp(modes * 0.01))


def test_eigenvalue_cloud_is_reproducible_for_identical_seeds() -> None:
    assert np.array_equal(_sample_modes(seed=21), _sample_modes(seed=21))


def test_eigenvalue_cloud_changes_for_different_seeds() -> None:
    assert not np.array_equal(_sample_modes(seed=21), _sample_modes(seed=22))


def test_eigenvalue_cloud_constructs_exact_conjugate_pairs() -> None:
    modes = _sample_modes(seed=7)
    assert np.array_equal(modes[1::2], modes[::2].conjugate())


def test_zero_spread_repeats_deterministic_wilson_cowan_modes() -> None:
    modes = generate_continuous_reservoir_modes(
        CENTERS,
        reservoir_mode="gaussian_eigenvalue_cloud",
        reservoir_size=8,
        eigenvalue_sigma_real=0.0,
        eigenvalue_sigma_imag=0.0,
        seed=99,
    )
    assert np.array_equal(modes, np.tile(CENTERS, 4))


def test_deterministic_mode_preserves_original_two_mode_behavior() -> None:
    reversed_centers = CENTERS[::-1]
    modes = generate_continuous_reservoir_modes(
        reversed_centers,
        reservoir_mode="deterministic",
        reservoir_size=100,
        eigenvalue_sigma_real=1.0,
        eigenvalue_sigma_imag=1.0,
        seed=3,
    )
    assert np.array_equal(modes, reversed_centers)


def test_zero_spread_repeats_real_wilson_cowan_modes_for_odd_size() -> None:
    real_centers = np.array([-1.0, -3.0])
    modes = generate_continuous_reservoir_modes(
        real_centers,
        reservoir_mode="gaussian_eigenvalue_cloud",
        reservoir_size=5,
        eigenvalue_sigma_real=0.0,
        eigenvalue_sigma_imag=0.0,
    )
    assert np.array_equal(modes, np.array([-1.0, -3.0, -1.0, -3.0, -1.0]))


def test_oscillatory_cloud_requires_even_reservoir_size() -> None:
    with pytest.raises(ValueError, match="even"):
        _sample_modes(seed=1, size=11)


def test_physics_reservoir_runs_one_step_prediction() -> None:
    eigenvalues = np.array([0.92 + 0.1j, 0.92 - 0.1j])
    reservoir = build_physics_informed_reservoir(eigenvalues, input_scale=0.2)
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 500))

    reservoir.fit_one_step(signal[:400], ridge=1.0e-4, washout_samples=25)
    predictions = reservoir.predict_one_step(signal[400:], warmup_values=signal[350:400])

    assert predictions.shape == (99,)
    assert np.all(np.isfinite(predictions))


def test_training_keeps_recurrent_dynamics_fixed() -> None:
    modes = discrete_reservoir_eigenvalues(_sample_modes(seed=5), dt=0.01)
    reservoir = build_physics_informed_reservoir(modes, input_scale=0.2, seed=17)
    eigenvalues_before = reservoir.reservoir.eigenvalues.copy()
    input_weights_before = reservoir.reservoir.input_weights.copy()

    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 500))
    reservoir.fit_one_step(signal, ridge=1.0e-4, washout_samples=25)

    assert np.array_equal(reservoir.reservoir.eigenvalues, eigenvalues_before)
    assert np.array_equal(reservoir.reservoir.input_weights, input_weights_before)
