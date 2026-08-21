from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_esn.models.complex_esn import ComplexESN
from physics_esn.models.esn import LinearReadout, fit_ridge_readout


DETERMINISTIC_RESERVOIR_MODE = "deterministic"
GAUSSIAN_EIGENVALUE_CLOUD_MODE = "gaussian_eigenvalue_cloud"


def _validated_wilson_cowan_centers(eigenvalue_centers: np.ndarray) -> tuple[np.ndarray, bool]:
    centers = np.asarray(eigenvalue_centers, dtype=np.complex128)
    if centers.shape != (2,) or not np.all(np.isfinite(centers)):
        raise ValueError("Wilson-Cowan eigenvalue centers must be a finite array of length two.")
    if np.any(centers.real >= 0.0):
        raise ValueError("Wilson-Cowan eigenvalue centers must have strictly negative real parts.")

    centers = centers.copy()
    real_centers = np.isclose(centers.imag, 0.0, rtol=0.0, atol=1.0e-12)
    if np.all(real_centers):
        return centers, False
    if np.any(real_centers) or not np.allclose(
        centers[0],
        centers[1].conjugate(),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Non-real Wilson-Cowan eigenvalue centers must form a conjugate pair.")
    return centers, True


def _sample_stable_real_part(
    rng: np.random.Generator,
    center: float,
    sigma: float,
) -> float:
    if sigma == 0.0:
        return center

    # This is a Gaussian conditioned on stability, rather than a clipped Gaussian.
    for _ in range(10_000):
        sampled = float(rng.normal(center, sigma))
        if np.isfinite(sampled) and sampled < 0.0:
            return sampled
    raise RuntimeError("Unable to sample a finite stable real eigenvalue component.")


def generate_continuous_reservoir_modes(
    eigenvalue_centers: np.ndarray,
    reservoir_mode: str = DETERMINISTIC_RESERVOIR_MODE,
    reservoir_size: int = 2,
    eigenvalue_sigma_real: float = 0.0,
    eigenvalue_sigma_imag: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Construct stable continuous-time modes centered on a Wilson-Cowan spectrum."""
    centers, oscillatory_centers = _validated_wilson_cowan_centers(eigenvalue_centers)
    if reservoir_mode == DETERMINISTIC_RESERVOIR_MODE:
        return centers
    if reservoir_mode != GAUSSIAN_EIGENVALUE_CLOUD_MODE:
        raise ValueError(
            "reservoir_mode must be 'deterministic' or 'gaussian_eigenvalue_cloud'."
        )
    if isinstance(reservoir_size, bool) or not isinstance(reservoir_size, (int, np.integer)):
        raise ValueError("reservoir_size must be a positive integer.")
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be a positive integer.")
    if not np.isfinite(eigenvalue_sigma_real) or eigenvalue_sigma_real < 0.0:
        raise ValueError("eigenvalue_sigma_real must be finite and non-negative.")
    if not np.isfinite(eigenvalue_sigma_imag) or eigenvalue_sigma_imag < 0.0:
        raise ValueError("eigenvalue_sigma_imag must be finite and non-negative.")

    modes = np.empty(reservoir_size, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    needs_conjugate_pairs = oscillatory_centers or eigenvalue_sigma_imag > 0.0
    if needs_conjugate_pairs and reservoir_size % 2:
        raise ValueError("reservoir_size must be even when constructing conjugate mode pairs.")

    if needs_conjugate_pairs:
        pair_centers = centers[[int(np.argmax(centers.imag))]] if oscillatory_centers else centers
        for pair_index in range(reservoir_size // 2):
            center = pair_centers[pair_index % pair_centers.size]
            real_part = _sample_stable_real_part(rng, float(center.real), eigenvalue_sigma_real)
            imaginary_part = float(rng.normal(center.imag, eigenvalue_sigma_imag))
            representative = real_part + 1j * imaginary_part
            modes[2 * pair_index] = representative
            modes[2 * pair_index + 1] = representative.conjugate()
    else:
        for index in range(reservoir_size):
            center = centers[index % centers.size]
            modes[index] = _sample_stable_real_part(
                rng,
                float(center.real),
                eigenvalue_sigma_real,
            )

    if not np.all(np.isfinite(modes)) or np.any(modes.real >= 0.0):
        raise RuntimeError("Generated reservoir modes must be finite and strictly stable.")
    return modes


@dataclass
class PhysicsInformedReservoir:
    reservoir: ComplexESN
    readout: LinearReadout | None = None

    @staticmethod
    def _real_features(states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.complex128)
        return np.concatenate((states.real, states.imag), axis=1)

    def fit_one_step(
        self,
        signal_values: np.ndarray,
        ridge: float,
        washout_samples: int = 0,
    ) -> LinearReadout:
        values = np.asarray(signal_values, dtype=np.float64)
        if values.ndim != 1 or values.size <= washout_samples + 1 or washout_samples < 0:
            raise ValueError("Training signal is too short for the requested washout.")
        self.reservoir.reset()
        states = self.reservoir.run(values[:-1])
        features = self._real_features(states)[washout_samples:]
        targets = values[1 + washout_samples :]
        self.readout = fit_ridge_readout(features, targets, ridge=ridge)
        return self.readout

    def predict_one_step(
        self,
        signal_values: np.ndarray,
        warmup_values: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("Readout has not been fitted.")
        values = np.asarray(signal_values, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Prediction signal must contain at least two samples.")
        self.reservoir.reset()
        if warmup_values is not None:
            warmup = np.asarray(warmup_values, dtype=np.float64)
            if warmup.ndim != 1:
                raise ValueError("warmup_values must be one-dimensional.")
            self.reservoir.run(warmup)
        states = self.reservoir.run(values[:-1])
        return self.readout.predict(self._real_features(states))


def build_physics_informed_reservoir(
    eigenvalues: np.ndarray,
    input_scale: float,
    seed: int = 0,
) -> PhysicsInformedReservoir:
    return PhysicsInformedReservoir(
        ComplexESN.from_eigenvalues(eigenvalues, input_scale=input_scale, seed=seed)
    )
