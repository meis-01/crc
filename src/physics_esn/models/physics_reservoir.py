from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_esn.models.complex_esn import ComplexESN
from physics_esn.models.esn import LinearReadout, fit_ridge_readout


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
