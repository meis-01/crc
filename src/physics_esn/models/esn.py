from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearReadout:
    weights: np.ndarray
    bias: float

    def predict(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64) @ self.weights + self.bias


def fit_ridge_readout(states: np.ndarray, targets: np.ndarray, ridge: float) -> LinearReadout:
    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("states and targets must be aligned 2D and 1D arrays.")
    if x.shape[0] == 0 or ridge < 0.0:
        raise ValueError("Training data must be non-empty and ridge must be non-negative.")

    design = np.column_stack((np.ones(x.shape[0]), x))
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return LinearReadout(weights=coefficients[1:], bias=float(coefficients[0]))
