from __future__ import annotations

import numpy as np

from physics_esn.models.physics_reservoir import build_physics_informed_reservoir


def test_physics_reservoir_runs_one_step_prediction() -> None:
    eigenvalues = np.array([0.92 + 0.1j, 0.92 - 0.1j])
    reservoir = build_physics_informed_reservoir(eigenvalues, input_scale=0.2)
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 500))

    reservoir.fit_one_step(signal[:400], ridge=1.0e-4, washout_samples=25)
    predictions = reservoir.predict_one_step(signal[400:], warmup_values=signal[350:400])

    assert predictions.shape == (99,)
    assert np.all(np.isfinite(predictions))
