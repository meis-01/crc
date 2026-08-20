from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComplexESN:
    eigenvalues: np.ndarray
    input_weights: np.ndarray
    state: np.ndarray

    @classmethod
    def from_eigenvalues(cls, eigenvalues: np.ndarray, input_scale: float, seed: int = 0) -> "ComplexESN":
        rng = np.random.default_rng(seed)
        eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        if eigenvalues.ndim != 1 or eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("eigenvalues must be a non-empty finite one-dimensional array.")
        if np.any(np.abs(eigenvalues) >= 1.0):
            raise ValueError("Discrete reservoir eigenvalues must lie inside the unit circle.")
        if not np.isfinite(input_scale) or input_scale <= 0.0:
            raise ValueError("input_scale must be finite and positive.")
        input_weights = input_scale * (
            rng.standard_normal(eigenvalues.size) + 1j * rng.standard_normal(eigenvalues.size)
        )
        state = np.zeros(eigenvalues.size, dtype=np.complex128)
        return cls(eigenvalues=eigenvalues, input_weights=input_weights, state=state)

    def reset(self) -> None:
        self.state[...] = 0.0

    def step(self, value: float) -> np.ndarray:
        self.state = self.eigenvalues * self.state + self.input_weights * value
        if not np.all(np.isfinite(self.state)):
            raise FloatingPointError("Complex reservoir state became non-finite.")
        return self.state.copy()

    def run(self, signal_values: np.ndarray) -> np.ndarray:
        outputs = np.empty((len(signal_values), self.state.size), dtype=np.complex128)
        for index, value in enumerate(np.asarray(signal_values, dtype=np.float64)):
            outputs[index] = self.step(float(value))
        return outputs
