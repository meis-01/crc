from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from physics_esn.analysis.spectrum import compute_psd
from physics_esn.config import load_config
from physics_esn.data.loader import (
    discover_edf_recordings,
    inspect_raw_recording,
    load_single_channel,
    select_subject_recording,
)
from physics_esn.data.preprocessing import preprocess_oz_signal
from physics_esn.fitting.wilson_cowan_fit import fit_wilson_cowan_psd, save_subject_fit
from physics_esn.models.physics_reservoir import build_physics_informed_reservoir
from physics_esn.models.wilson_cowan import (
    WilsonCowanParameters,
    continuous_eigenvalues,
    discrete_reservoir_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    simulate_wilson_cowan,
)


LOGGER = logging.getLogger(__name__)


def _params_from_config(config: dict[str, Any]) -> WilsonCowanParameters:
    wc = config["wilson_cowan"]
    return WilsonCowanParameters(
        tau_e=wc["tau_e"],
        tau_i=wc["tau_i"],
        w_ee=wc["w_ee"],
        w_ei=wc["w_ei"],
        w_ie=wc["w_ie"],
        w_ii=wc["w_ii"],
        p=wc["p"],
        q=wc["q"],
        sigmoid_gain=wc["sigmoid_gain"],
        sigmoid_theta=wc["sigmoid_theta"],
    )


def _prediction_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    persistence_predictions: np.ndarray,
) -> dict[str, float]:
    residuals = np.asarray(predictions) - np.asarray(targets)
    persistence_residuals = np.asarray(persistence_predictions) - np.asarray(targets)
    if np.std(targets) == 0.0 or np.std(predictions) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(targets, predictions)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "correlation": correlation,
        "persistence_rmse": float(np.sqrt(np.mean(persistence_residuals**2))),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sub001_pipeline(config_path: str | Path, output_dir: str | Path = "artifacts") -> dict[str, Any]:
    config = load_config(config_path)
    LOGGER.info("Discovering local EDF recordings under %s.", config["data_root"])
    recordings = discover_edf_recordings(config["data_root"])
    recording = select_subject_recording(
        recordings,
        subject_id=config["subject_id"],
        session_id=config.get("session"),
        task=config.get("task"),
    )
    LOGGER.info("Inspecting raw EDF header before loading or resampling: %s", recording.edf_path)
    raw_summary = inspect_raw_recording(recording.edf_path)
    raw_oz, raw_rate = load_single_channel(recording.edf_path, config["target_channel"])
    signal, rate = preprocess_oz_signal(raw_oz, raw_rate, target_rate_hz=config.get("resample_hz"))
    LOGGER.info("Loaded %s and preprocessed Oz from %.1f Hz to %.1f Hz.", recording.subject_id, raw_rate, rate)
    freqs, power = compute_psd(
        signal,
        rate,
        fmin_hz=config["psd"]["fmin_hz"],
        fmax_hz=config["psd"]["fmax_hz"],
    )

    train_fraction = float(config["prediction"]["train_fraction"])
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("prediction.train_fraction must be between zero and one.")
    split_index = int(signal.size * train_fraction)
    train_signal = signal[:split_index]
    test_signal = signal[split_index:]
    if train_signal.size < 2 or test_signal.size < 2:
        raise ValueError("The chronological train/test split produced an empty partition.")

    initial_params = _params_from_config(config)
    dt = float(config["fit"]["dt"])
    LOGGER.info("Fitting Wilson-Cowan spectral dynamics on the sub-001 training partition only.")
    fit_result = fit_wilson_cowan_psd(
        observed_signal=train_signal,
        sampling_rate_hz=rate,
        initial_params=initial_params,
        dt=dt,
        duration_s=float(config["fit"]["duration_s"]),
        maxiter=int(config["fit"]["maxiter"]),
        population_size=int(config["fit"]["population_size"]),
        polish=bool(config["fit"]["polish"]),
        random_seed=int(config["fit"]["random_seed"]),
        fmin_hz=float(config["psd"]["fmin_hz"]),
        fmax_hz=float(config["psd"]["fmax_hz"]),
    )
    fitted = fit_result.parameters
    equilibrium = find_equilibrium(fitted)
    jacobian = jacobian_at_equilibrium(equilibrium, fitted)
    lambdas = continuous_eigenvalues(jacobian)
    mus = discrete_reservoir_eigenvalues(lambdas, dt=dt)
    reservoir = build_physics_informed_reservoir(
        mus,
        input_scale=float(config["reservoir"]["input_scale"]),
    )
    washout_samples = int(config["prediction"]["washout_samples"])
    warmup_samples = min(int(config["prediction"]["warmup_samples"]), train_signal.size)
    reservoir.fit_one_step(
        train_signal,
        ridge=float(config["reservoir"]["readout_ridge"]),
        washout_samples=washout_samples,
    )
    warmup_values = train_signal[-warmup_samples:] if warmup_samples else None
    predictions = reservoir.predict_one_step(test_signal, warmup_values=warmup_values)
    targets = test_signal[1:]
    metrics = _prediction_metrics(targets, predictions, test_signal[:-1])

    simulation_time, simulated_states = simulate_wilson_cowan(
        fitted,
        dt=dt,
        duration_s=float(config["fit"]["duration_s"]),
    )
    subject_dir = Path(output_dir) / recording.subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    _write_json(subject_dir / "raw_inspection.json", raw_summary)
    np.savez_compressed(subject_dir / "psd.npz", frequencies_hz=freqs, power=power)
    np.savez_compressed(subject_dir / "simulation.npz", time_s=simulation_time, states=simulated_states)
    np.savez_compressed(
        subject_dir / "prediction.npz",
        targets=targets,
        predictions=predictions,
        sampling_rate_hz=rate,
        split_index=split_index,
    )
    save_subject_fit(
        subject_dir / "wilson_cowan_fit.json",
        recording.subject_id,
        fit_result,
        equilibrium,
        jacobian,
        lambdas,
        mus,
    )
    summary = {
        "subject_id": recording.subject_id,
        "recording_path": str(recording.edf_path),
        "raw_summary": raw_summary,
        "processed_sampling_rate_hz": rate,
        "psd_points": len(freqs),
        "train_samples": int(train_signal.size),
        "test_samples": int(test_signal.size),
        "fit_objective": fit_result.objective,
        "fit_evaluations": fit_result.evaluations,
        "equilibrium": equilibrium.tolist(),
        "jacobian": jacobian.tolist(),
        "continuous_eigenvalues": [[float(value.real), float(value.imag)] for value in lambdas],
        "discrete_eigenvalues": [[float(value.real), float(value.imag)] for value in mus],
        "reservoir_size": int(reservoir.reservoir.state.size),
        "simulation_samples": int(simulated_states.shape[0]),
        "prediction_metrics": metrics,
        "artifact_directory": str(subject_dir),
    }
    _write_json(subject_dir / "summary.json", summary)
    LOGGER.info("Completed sub-001 pipeline; artifacts saved to %s.", subject_dir)
    return summary
