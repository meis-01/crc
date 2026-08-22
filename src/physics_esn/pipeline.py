from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from physics_esn.analysis.spectrum import compute_psd
from physics_esn.config import load_config
from physics_esn.data.loader import (
    discover_edf_recordings,
    inspect_raw_recording,
    load_single_channel,
    select_subject_recording,
)
from physics_esn.data.preprocessing import preprocess_chronological_split
from physics_esn.fitting.wilson_cowan_fit import (
    WilsonCowanFitResult,
    fit_wilson_cowan_psd,
    save_subject_fit,
)
from physics_esn.models.physics_reservoir import (
    DETERMINISTIC_POLES_MODE,
    normalize_reservoir_mode,
)
from physics_esn.models.reservoir_factory import ReservoirBuildResult, build_reservoir
from physics_esn.models.wilson_cowan import (
    WilsonCowanParameters,
    continuous_eigenvalues,
    discrete_reservoir_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    simulate_wilson_cowan,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedExperiment:
    """Training-only WC fit and chronological data reused across reservoir modes."""

    config: dict[str, Any]
    subject_id: str
    recording_path: str
    raw_summary: dict[str, Any]
    train_signal: np.ndarray
    test_signal: np.ndarray
    sampling_rate_hz: float
    raw_split_index: int
    preprocessing: dict[str, Any]
    frequencies_hz: np.ndarray
    power: np.ndarray
    fit_result: WilsonCowanFitResult
    equilibrium: np.ndarray
    jacobian: np.ndarray
    continuous_eigenvalues: np.ndarray
    discrete_eigenvalues: np.ndarray
    simulation_time_s: np.ndarray
    simulated_states: np.ndarray
    preparation_runtime_s: float = 0.0


def _params_from_config(config: dict[str, Any]) -> WilsonCowanParameters:
    wc = config["wilson_cowan"]
    return WilsonCowanParameters(
        tau_e=float(wc["tau_e"]),
        tau_i=float(wc["tau_i"]),
        w_ee=float(wc["w_ee"]),
        w_ei=float(wc["w_ei"]),
        w_ie=float(wc["w_ie"]),
        w_ii=float(wc["w_ii"]),
        p=float(wc["p"]),
        q=float(wc["q"]),
        sigmoid_gain=float(wc["sigmoid_gain"]),
        sigmoid_theta=float(wc["sigmoid_theta"]),
    )


def _pearson_correlation(targets: np.ndarray, predictions: np.ndarray) -> float:
    if np.std(targets) == 0.0 or np.std(predictions) == 0.0:
        return 0.0
    return float(np.corrcoef(targets, predictions)[0, 1])


def _basic_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets_array = np.asarray(targets, dtype=np.float64)
    predictions_array = np.asarray(predictions, dtype=np.float64)
    if (
        targets_array.ndim != 1
        or predictions_array.ndim != 1
        or targets_array.shape != predictions_array.shape
        or targets_array.size == 0
        or not np.all(np.isfinite(targets_array))
        or not np.all(np.isfinite(predictions_array))
    ):
        raise ValueError("Metric targets and predictions must be aligned finite vectors.")
    residuals = predictions_array - targets_array
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "pearson_correlation": _pearson_correlation(targets_array, predictions_array),
    }


def _prediction_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    persistence_predictions: np.ndarray,
) -> dict[str, float]:
    """Return current metrics while retaining legacy persistence fields."""

    prediction = _basic_metrics(targets, predictions)
    persistence = _basic_metrics(targets, persistence_predictions)
    return {
        **prediction,
        # The old summary used ``correlation``. Keep it as an alias.
        "correlation": prediction["pearson_correlation"],
        "persistence_rmse": persistence["rmse"],
        "persistence_mae": persistence["mae"],
        "persistence_pearson_correlation": persistence["pearson_correlation"],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value)
        return {
            "shape": list(matrix.shape),
            "data": matrix.data.tolist(),
            "indices": matrix.indices.tolist(),
            "indptr": matrix.indptr.tolist(),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )


def _complex_pairs(values: np.ndarray | None) -> list[list[float]]:
    if values is None:
        return []
    return [[float(value.real), float(value.imag)] for value in np.asarray(values)]


def _hybrid_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    fit_loss = dict(config.get("fit", {}).get("loss", {}))
    # Accept the prompt's ``wc_fit.loss`` spelling without breaking the existing
    # repository's top-level ``fit`` section.
    wc_fit_loss = dict(config.get("wc_fit", {}).get("loss", {}))
    return {**fit_loss, **wc_fit_loss}


def prepare_sub001_experiment(config: dict[str, Any]) -> PreparedExperiment:
    """Load, split, preprocess, and fit WC once using training data only."""

    started = perf_counter()
    LOGGER.info("Discovering local EDF recordings under %s.", config["data_root"])
    recordings = discover_edf_recordings(config["data_root"])
    recording = select_subject_recording(
        recordings,
        subject_id=str(config["subject_id"]),
        session_id=config.get("session"),
        task=config.get("task"),
    )
    LOGGER.info("Inspecting raw EDF header before preprocessing: %s", recording.edf_path)
    raw_summary = inspect_raw_recording(recording.edf_path)
    raw_signal, raw_rate = load_single_channel(recording.edf_path, str(config["target_channel"]))
    split = preprocess_chronological_split(
        raw_signal,
        raw_rate,
        float(config["prediction"]["train_fraction"]),
        target_rate_hz=config.get("resample_hz"),
    )
    train_signal = np.asarray(split.train, dtype=np.float64)
    test_signal = np.asarray(split.test, dtype=np.float64)
    rate = float(split.sampling_rate_hz)
    LOGGER.info(
        "Split raw %s at sample %d before fitting preprocessing; processed rate %.3f Hz.",
        config["target_channel"],
        split.raw_split_index,
        rate,
    )
    frequencies_hz, power = compute_psd(
        train_signal,
        rate,
        fmin_hz=float(config["psd"]["fmin_hz"]),
        fmax_hz=float(config["psd"]["fmax_hz"]),
    )

    fit_config = config["fit"]
    loss_config = _hybrid_loss_config(config)
    fit_dt = float(fit_config["dt"])
    LOGGER.info("Fitting Wilson-Cowan dynamics on the training partition only.")
    fit_result = fit_wilson_cowan_psd(
        observed_signal=train_signal,
        sampling_rate_hz=rate,
        initial_params=_params_from_config(config),
        dt=fit_dt,
        duration_s=float(fit_config["duration_s"]),
        maxiter=int(fit_config["maxiter"]),
        population_size=int(fit_config["population_size"]),
        polish=bool(fit_config["polish"]),
        random_seed=int(fit_config["random_seed"]),
        fmin_hz=float(config["psd"]["fmin_hz"]),
        fmax_hz=float(config["psd"]["fmax_hz"]),
        psd_weight=float(loss_config.get("psd_weight", 1.0)),
        stft_weight=float(loss_config.get("stft_weight", 0.0)),
        temporal_weight=float(loss_config.get("temporal_weight", 0.0)),
        stft_window_seconds=float(loss_config.get("stft_window_seconds", 1.0)),
        stft_overlap_fraction=float(loss_config.get("stft_overlap_fraction", 0.5)),
    )
    fitted = fit_result.parameters
    equilibrium = find_equilibrium(fitted)
    jacobian = jacobian_at_equilibrium(equilibrium, fitted)
    lambdas = continuous_eigenvalues(jacobian)
    sample_interval_s = 1.0 / rate
    mus = discrete_reservoir_eigenvalues(lambdas, dt=sample_interval_s)
    simulation_time_s, simulated_states = simulate_wilson_cowan(
        fitted,
        dt=fit_dt,
        duration_s=float(fit_config["duration_s"]),
    )
    preprocessing = {
        "split_before_fitted_preprocessing": True,
        "trend_intercept": split.preprocessor.trend_intercept_,
        "trend_slope_per_raw_sample": split.preprocessor.trend_slope_,
        "normalization_mean": split.preprocessor.normalization_mean_,
        "normalization_std": split.preprocessor.normalization_std_,
        "resample_partitions_independently": True,
        "resample_edge_guard_samples": split.preprocessor.resample_guard_samples,
    }
    return PreparedExperiment(
        config=deepcopy(config),
        subject_id=recording.subject_id,
        recording_path=str(recording.edf_path),
        raw_summary=raw_summary,
        train_signal=train_signal,
        test_signal=test_signal,
        sampling_rate_hz=rate,
        raw_split_index=int(split.raw_split_index),
        preprocessing=preprocessing,
        frequencies_hz=frequencies_hz,
        power=power,
        fit_result=fit_result,
        equilibrium=equilibrium,
        jacobian=jacobian,
        continuous_eigenvalues=lambdas,
        discrete_eigenvalues=mus,
        simulation_time_s=simulation_time_s,
        simulated_states=simulated_states,
        preparation_runtime_s=float(perf_counter() - started),
    )


def _update_hash(hasher: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())


def _fixed_dynamics_fingerprint(reservoir: Any) -> str:
    """Hash parameters that must remain fixed while excluding mutable state."""

    hasher = hashlib.sha256(type(reservoir).__qualname__.encode("utf-8"))
    for name in ("eigenvalues", "input_weights", "input_weights_e", "input_weights_i"):
        if hasattr(reservoir, name):
            _update_hash(hasher, name, np.asarray(getattr(reservoir, name)))
    parameters = getattr(reservoir, "parameters", None)
    if parameters is not None and hasattr(parameters, "parameter_matrix"):
        _update_hash(hasher, "parameter_matrix", parameters.parameter_matrix())
        _update_hash(hasher, "sigmoid_gain", np.asarray(parameters.sigmoid_gain))
        _update_hash(hasher, "sigmoid_theta", np.asarray(parameters.sigmoid_theta))
    graph = getattr(reservoir, "coupling_graph", None)
    if graph is not None:
        matrix = sparse.csr_matrix(graph)
        _update_hash(hasher, "coupling_data", matrix.data)
        _update_hash(hasher, "coupling_indices", matrix.indices)
        _update_hash(hasher, "coupling_indptr", matrix.indptr)
    for name in ("sample_dt", "rk4_substeps", "coupling_strength"):
        if hasattr(reservoir, name):
            hasher.update(f"{name}={getattr(reservoir, name)!r}".encode("ascii"))
    return hasher.hexdigest()


def _effective_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    effective = deepcopy(config)
    effective.setdefault("reservoir", {})["reservoir_mode"] = mode
    return effective


def _save_mode_artifacts(
    mode_dir: Path,
    prepared: PreparedExperiment,
    built: ReservoirBuildResult,
    summary: dict[str, Any],
    effective_config: dict[str, Any],
    targets: np.ndarray,
    predictions: np.ndarray,
    persistence_predictions: np.ndarray,
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    _write_json(mode_dir / "raw_inspection.json", prepared.raw_summary)
    _write_json(mode_dir / "effective_config.json", effective_config)
    np.savez_compressed(
        mode_dir / "psd.npz",
        frequencies_hz=prepared.frequencies_hz,
        power=prepared.power,
    )
    np.savez_compressed(
        mode_dir / "simulation.npz",
        time_s=prepared.simulation_time_s,
        states=prepared.simulated_states,
    )
    np.savez_compressed(
        mode_dir / "prediction.npz",
        targets=targets,
        predictions=predictions,
        persistence_predictions=persistence_predictions,
        sampling_rate_hz=prepared.sampling_rate_hz,
        processed_split_index=prepared.train_signal.size,
        raw_split_index=prepared.raw_split_index,
    )
    dynamics = built.model.reservoir
    dynamics_payload: dict[str, np.ndarray | float | int] = {
        "reservoir_mode": np.asarray(built.mode),
    }
    if built.continuous_eigenvalues is not None:
        dynamics_payload["continuous_eigenvalues"] = built.continuous_eigenvalues
    if built.discrete_eigenvalues is not None:
        dynamics_payload["discrete_eigenvalues"] = built.discrete_eigenvalues
    if hasattr(dynamics, "input_weights"):
        dynamics_payload["input_weights"] = np.asarray(dynamics.input_weights)
    parameters = getattr(dynamics, "parameters", None)
    if parameters is not None and hasattr(parameters, "parameter_matrix"):
        dynamics_payload["parameter_names"] = np.asarray(
            ("tau_e", "tau_i", "w_ee", "w_ei", "w_ie", "w_ii", "p", "q")
        )
        dynamics_payload["parameter_matrix"] = parameters.parameter_matrix()
        dynamics_payload["sigmoid_gain"] = np.asarray(parameters.sigmoid_gain)
        dynamics_payload["sigmoid_theta"] = np.asarray(parameters.sigmoid_theta)
        dynamics_payload["input_weights_e"] = np.asarray(dynamics.input_weights_e)
        dynamics_payload["input_weights_i"] = np.asarray(dynamics.input_weights_i)
        dynamics_payload["sample_interval_s"] = float(dynamics.sample_dt)
        dynamics_payload["rk4_substeps"] = int(dynamics.rk4_substeps)
        dynamics_payload["initial_state"] = np.asarray(dynamics.initial_state)
        dynamics_payload["state_bounds"] = np.asarray(dynamics.state_bounds)
        dynamics_payload["coupling_strength"] = float(dynamics.coupling_strength)
        graph = dynamics.coupling_graph
        if graph is None:
            dynamics_payload["coupling_graph_data"] = np.empty(0, dtype=np.float64)
            dynamics_payload["coupling_graph_indices"] = np.empty(0, dtype=np.int64)
            dynamics_payload["coupling_graph_indptr"] = np.zeros(
                dynamics.num_blocks + 1,
                dtype=np.int64,
            )
        else:
            matrix = sparse.csr_matrix(graph)
            dynamics_payload["coupling_graph_data"] = matrix.data
            dynamics_payload["coupling_graph_indices"] = matrix.indices
            dynamics_payload["coupling_graph_indptr"] = matrix.indptr
        dynamics_payload["coupling_graph_shape"] = np.asarray(
            (dynamics.num_blocks, dynamics.num_blocks),
            dtype=np.int64,
        )
    np.savez_compressed(mode_dir / "reservoir_dynamics.npz", **dynamics_payload)
    save_subject_fit(
        mode_dir / "wilson_cowan_fit.json",
        prepared.subject_id,
        prepared.fit_result,
        prepared.equilibrium,
        prepared.jacobian,
        prepared.continuous_eigenvalues,
        prepared.discrete_eigenvalues,
    )
    _write_json(mode_dir / "summary.json", summary)


def run_prepared_reservoir_experiment(
    prepared: PreparedExperiment,
    output_dir: str | Path,
    reservoir_mode: str | None = None,
) -> dict[str, Any]:
    """Train one readout and evaluate one fixed reservoir architecture."""

    requested_mode = str(
        reservoir_mode
        if reservoir_mode is not None
        else prepared.config.get("reservoir", {}).get(
            "reservoir_mode", DETERMINISTIC_POLES_MODE
        )
    )
    mode = normalize_reservoir_mode(requested_mode)
    effective_config = _effective_config(prepared.config, mode)
    mode_started = perf_counter()
    construction_started = perf_counter()
    built: ReservoirBuildResult = build_reservoir(
        effective_config,
        prepared.fit_result.parameters,
        prepared.continuous_eigenvalues,
        sample_interval_s=1.0 / prepared.sampling_rate_hz,
        reservoir_mode=mode,
    )
    construction_runtime = perf_counter() - construction_started

    dynamics = built.model.reservoir
    fingerprint_before = _fixed_dynamics_fingerprint(dynamics)
    prediction_config = prepared.config["prediction"]
    fit_started = perf_counter()
    built.fit_one_step(
        prepared.train_signal,
        ridge=float(prepared.config["reservoir"]["readout_ridge"]),
        washout_samples=int(prediction_config["washout_samples"]),
    )
    readout_runtime = perf_counter() - fit_started
    fingerprint_after_fit = _fixed_dynamics_fingerprint(dynamics)
    if fingerprint_before != fingerprint_after_fit:
        raise RuntimeError("Reservoir dynamics changed while fitting the readout.")

    warmup_samples = min(
        int(prediction_config["warmup_samples"]),
        prepared.train_signal.size,
    )
    warmup_values = (
        prepared.train_signal[-warmup_samples:] if warmup_samples > 0 else None
    )
    prediction_started = perf_counter()
    predictions = built.predict_one_step(
        prepared.test_signal,
        warmup_values=warmup_values,
    )
    prediction_runtime = perf_counter() - prediction_started
    fingerprint_after_prediction = _fixed_dynamics_fingerprint(dynamics)
    if fingerprint_before != fingerprint_after_prediction:
        raise RuntimeError("Reservoir dynamics changed while predicting.")

    targets = prepared.test_signal[1:]
    persistence_predictions = prepared.test_signal[:-1]
    metrics = _prediction_metrics(targets, predictions, persistence_predictions)
    persistence_metrics = _basic_metrics(targets, persistence_predictions)
    mode_runtime = perf_counter() - mode_started
    mode_dir = Path(output_dir) / prepared.subject_id / mode
    reservoir_config = prepared.config.get("reservoir", {})
    requested_reservoir_size = (
        int(prepared.config.get("nonlinear_reservoir", {}).get("num_blocks", built.reservoir_units))
        if built.num_wc_blocks > 0
        else int(reservoir_config.get("reservoir_size", 2))
    )
    fit_integration_dt_s = float(
        prepared.config.get("fit", {}).get("dt", 1.0 / prepared.sampling_rate_hz)
    )
    summary: dict[str, Any] = {
        "subject_id": prepared.subject_id,
        "recording_path": prepared.recording_path,
        "raw_summary": prepared.raw_summary,
        "processed_sampling_rate_hz": prepared.sampling_rate_hz,
        "sample_interval_s": 1.0 / prepared.sampling_rate_hz,
        "fit_integration_dt_s": fit_integration_dt_s,
        "fit_dt_matches_sample_interval": bool(
            np.isclose(fit_integration_dt_s, 1.0 / prepared.sampling_rate_hz)
        ),
        "raw_split_index": prepared.raw_split_index,
        "processed_split_index": int(prepared.train_signal.size),
        "preprocessing": prepared.preprocessing,
        "psd_points": int(prepared.frequencies_hz.size),
        "train_samples": int(prepared.train_signal.size),
        "test_samples": int(prepared.test_signal.size),
        "fit_objective": prepared.fit_result.objective,
        "fit_loss_components": prepared.fit_result.loss_components,
        "fit_evaluations": prepared.fit_result.evaluations,
        "fitted_wc_parameters": asdict(prepared.fit_result.parameters),
        "equilibrium": prepared.equilibrium.tolist(),
        "jacobian": prepared.jacobian.tolist(),
        "continuous_eigenvalues": _complex_pairs(prepared.continuous_eigenvalues),
        "discrete_eigenvalues": _complex_pairs(prepared.discrete_eigenvalues),
        "requested_reservoir_mode": requested_mode,
        "reservoir_mode": built.mode,
        "reservoir_architecture": built.mode,
        "requested_reservoir_size": requested_reservoir_size,
        "reservoir_size": built.reservoir_units,
        "reservoir_state_dimension": built.reservoir_state_dimension,
        "num_wc_blocks": built.num_wc_blocks,
        "reservoir_random_seed": built.random_seed,
        "reservoir_continuous_eigenvalues": _complex_pairs(built.continuous_eigenvalues),
        "reservoir_discrete_eigenvalues": _complex_pairs(built.discrete_eigenvalues),
        "reservoir_max_eigenvalue_modulus": (
            float(np.max(np.abs(built.discrete_eigenvalues)))
            if built.discrete_eigenvalues is not None
            else None
        ),
        "stability_or_boundedness": built.diagnostics,
        "recurrent_dynamics_unchanged": True,
        "fixed_dynamics_fingerprint": fingerprint_before,
        "fixed_dynamics_artifact": str(mode_dir / "reservoir_dynamics.npz"),
        "simulation_samples": int(prepared.simulated_states.shape[0]),
        "prediction_metrics": metrics,
        "persistence_baseline": persistence_metrics,
        "runtime_seconds": {
            "shared_preparation": prepared.preparation_runtime_s,
            "reservoir_construction": construction_runtime,
            "readout_fit": readout_runtime,
            "prediction": prediction_runtime,
            "mode_total": mode_runtime,
        },
        "model_configuration": effective_config,
        "artifact_directory": str(mode_dir),
    }
    _save_mode_artifacts(
        mode_dir,
        prepared,
        built,
        summary,
        effective_config,
        targets,
        predictions,
        persistence_predictions,
    )
    LOGGER.info("Completed %s; artifacts saved to %s.", mode, mode_dir)
    return summary


def run_sub001_pipeline(
    config_path: str | Path,
    output_dir: str | Path = "artifacts",
    reservoir_mode: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    prepared = prepare_sub001_experiment(config)
    return run_prepared_reservoir_experiment(
        prepared,
        output_dir=output_dir,
        reservoir_mode=reservoir_mode,
    )


def _compact_ablation_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["prediction_metrics"]
    return {
        "reservoir_mode": summary["reservoir_mode"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "pearson_correlation": metrics["pearson_correlation"],
        "persistence_rmse": metrics["persistence_rmse"],
        "reservoir_state_dimension": summary["reservoir_state_dimension"],
        "num_wc_blocks": summary["num_wc_blocks"],
        "random_seed": summary["reservoir_random_seed"],
        "runtime_seconds": summary["runtime_seconds"]["mode_total"],
        "artifact_directory": summary["artifact_directory"],
    }


def _ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Reservoir ablation — {report['subject_id']}",
        "",
        "All modes use one chronological split, one training-only WC fit, and the same "
        "teacher-forced one-step evaluation protocol.",
        "",
        "| Mode | RMSE | MAE | Pearson r | Persistence RMSE | State dim. | WC blocks | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["modes"]:
        lines.append(
            "| {reservoir_mode} | {rmse:.6f} | {mae:.6f} | "
            "{pearson_correlation:.6f} | {persistence_rmse:.6f} | "
            "{reservoir_state_dimension} | {num_wc_blocks} | {runtime_seconds:.3f} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            f"Shared WC fit objective: `{report['shared_fit_objective']:.8g}`.",
            "",
            "The metrics are descriptive results for this configured run, not uncertainty-adjusted "
            "evidence that one architecture generalizes better.",
            "",
        )
    )
    return "\n".join(lines)


def run_sub001_ablation(
    config_path: str | Path,
    output_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Run configured modes against one shared preprocessing result and WC fit."""

    config = load_config(config_path)
    configured_modes = config.get("ablation", {}).get(
        "modes",
        [
            "deterministic_poles",
            "distributed_poles",
            "independent_nonlinear_wc",
            "coupled_nonlinear_wc",
        ],
    )
    if not isinstance(configured_modes, list) or not configured_modes:
        raise ValueError("ablation.modes must be a non-empty list.")
    modes = [normalize_reservoir_mode(str(value)) for value in configured_modes]
    if len(set(modes)) != len(modes):
        raise ValueError("ablation.modes must identify distinct canonical architectures.")

    prepared = prepare_sub001_experiment(config)
    summaries = [
        run_prepared_reservoir_experiment(prepared, output_dir, reservoir_mode=mode)
        for mode in modes
    ]
    report = {
        "subject_id": prepared.subject_id,
        "shared_fit_objective": prepared.fit_result.objective,
        "shared_fitted_wc_parameters": asdict(prepared.fit_result.parameters),
        "shared_preparation_runtime_s": prepared.preparation_runtime_s,
        "modes": [_compact_ablation_row(summary) for summary in summaries],
    }
    subject_dir = Path(output_dir) / prepared.subject_id
    report_path = subject_dir / "ablation_summary.json"
    markdown_path = subject_dir / "ablation_report.md"
    report["artifact_path"] = str(report_path)
    report["markdown_report_path"] = str(markdown_path)
    _write_json(report_path, report)
    markdown_path.write_text(_ablation_markdown(report), encoding="utf-8")
    return report
