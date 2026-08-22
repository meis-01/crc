from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from physics_esn.data.loader import SubjectRecording
from physics_esn.fitting.wilson_cowan_fit import WilsonCowanFitResult
from physics_esn.models.reservoir_factory import build_reservoir
from physics_esn.models.wilson_cowan import (
    WilsonCowanParameters,
    continuous_eigenvalues,
    discrete_reservoir_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    simulate_wilson_cowan,
)
from physics_esn import pipeline as pipeline_module
from physics_esn.pipeline import (
    PreparedExperiment,
    prepare_sub001_experiment,
    run_prepared_reservoir_experiment,
    run_sub001_ablation,
)


MODES = (
    "deterministic_poles",
    "distributed_poles",
    "independent_nonlinear_wc",
    "coupled_nonlinear_wc",
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


def _config() -> dict[str, object]:
    return {
        "reservoir": {
            "reservoir_mode": "deterministic_poles",
            "reservoir_size": 6,
            "eigenvalue_sigma_real": 0.2,
            "eigenvalue_sigma_imag": 0.3,
            "random_seed": 11,
            "input_scale": 0.15,
            "readout_ridge": 1.0e-4,
        },
        "nonlinear_reservoir": {
            "num_blocks": 4,
            "parameter_jitter": 0.02,
            "input_scale": 0.08,
            "random_seed": 11,
            "rk4_substeps": 2,
            "initial_state": [0.1, 0.1],
            "preflight_duration_s": 0.04,
            "max_resample_attempts": 20,
            "state_bounds": [-0.05, 1.05],
        },
        "coupling": {
            "degree": 2,
            "strength": 0.04,
            "random_seed": 23,
        },
        "prediction": {
            "washout_samples": 10,
            "warmup_samples": 20,
        },
    }


def _prepared() -> PreparedExperiment:
    params = _params()
    equilibrium = find_equilibrium(params)
    jacobian = jacobian_at_equilibrium(equilibrium, params)
    lambdas = continuous_eigenvalues(jacobian)
    rate = 250.0
    simulation_time, simulated_states = simulate_wilson_cowan(
        params,
        dt=0.001,
        duration_s=0.02,
    )
    train_time = np.arange(320) / rate
    test_time = np.arange(120) / rate + train_time[-1] + 1.0 / rate
    train_signal = np.sin(2.0 * np.pi * 9.0 * train_time) + 0.1 * np.sin(
        2.0 * np.pi * 3.0 * train_time
    )
    test_signal = np.sin(2.0 * np.pi * 9.0 * test_time) + 0.1 * np.sin(
        2.0 * np.pi * 3.0 * test_time
    )
    fit_result = WilsonCowanFitResult(
        parameters=params,
        objective=0.25,
        evaluations=12,
        optimizer_success=True,
        optimizer_message="synthetic",
        loss_components={"psd": 0.25, "stft": 0.0, "temporal": 0.0},
    )
    return PreparedExperiment(
        config=_config(),
        subject_id="synthetic-001",
        recording_path="synthetic.edf",
        raw_summary={"source": "synthetic"},
        train_signal=train_signal,
        test_signal=test_signal,
        sampling_rate_hz=rate,
        raw_split_index=train_signal.size,
        preprocessing={"split_before_fitted_preprocessing": True},
        frequencies_hz=np.array([3.0, 9.0]),
        power=np.array([0.1, 1.0]),
        fit_result=fit_result,
        equilibrium=equilibrium,
        jacobian=jacobian,
        continuous_eigenvalues=lambdas,
        discrete_eigenvalues=discrete_reservoir_eigenvalues(lambdas, 1.0 / rate),
        simulation_time_s=simulation_time,
        simulated_states=simulated_states,
    )


def test_all_four_modes_run_end_to_end_and_write_isolated_artifacts(tmp_path: Path) -> None:
    expected_dimensions = {
        "deterministic_poles": 4,
        "distributed_poles": 12,
        "independent_nonlinear_wc": 8,
        "coupled_nonlinear_wc": 8,
    }
    summaries = {
        mode: run_prepared_reservoir_experiment(_prepared(), tmp_path, mode)
        for mode in MODES
    }

    for mode, summary in summaries.items():
        mode_dir = tmp_path / "synthetic-001" / mode
        assert summary["reservoir_mode"] == mode
        assert summary["model_configuration"]["reservoir"]["reservoir_mode"] == mode
        assert summary["reservoir_state_dimension"] == expected_dimensions[mode]
        assert summary["recurrent_dynamics_unchanged"] is True
        assert np.isfinite(summary["prediction_metrics"]["rmse"])
        assert np.isfinite(summary["prediction_metrics"]["mae"])
        assert np.isfinite(summary["prediction_metrics"]["pearson_correlation"])
        assert np.isfinite(summary["prediction_metrics"]["persistence_rmse"])
        assert (mode_dir / "summary.json").is_file()
        assert (mode_dir / "effective_config.json").is_file()
        assert (mode_dir / "prediction.npz").is_file()
        assert (mode_dir / "reservoir_dynamics.npz").is_file()
        saved = json.loads((mode_dir / "summary.json").read_text(encoding="utf-8"))
        assert saved["reservoir_architecture"] == mode
        with np.load(mode_dir / "prediction.npz") as prediction:
            assert prediction["targets"].shape == (119,)
            assert prediction["predictions"].shape == (119,)

    assert summaries["deterministic_poles"]["num_wc_blocks"] == 0
    assert summaries["distributed_poles"]["num_wc_blocks"] == 0
    assert summaries["independent_nonlinear_wc"]["num_wc_blocks"] == 4
    assert summaries["coupled_nonlinear_wc"]["num_wc_blocks"] == 4
    assert len({summary["artifact_directory"] for summary in summaries.values()}) == 4


def test_held_out_signal_does_not_change_reservoir_construction_metadata(tmp_path: Path) -> None:
    prepared = _prepared()
    changed_test = replace(prepared, test_signal=prepared.test_signal + 10_000.0)

    baseline = run_prepared_reservoir_experiment(
        prepared,
        tmp_path / "baseline",
        "independent_nonlinear_wc",
    )
    changed = run_prepared_reservoir_experiment(
        changed_test,
        tmp_path / "changed",
        "independent_nonlinear_wc",
    )

    assert baseline["fixed_dynamics_fingerprint"] == changed["fixed_dynamics_fingerprint"]
    assert baseline["fitted_wc_parameters"] == changed["fitted_wc_parameters"]
    assert baseline["fit_objective"] == changed["fit_objective"]


def test_pipeline_passes_only_training_partition_to_wc_fit(monkeypatch) -> None:
    positions = np.arange(400, dtype=np.float64)
    baseline_signal = 2.0 + 0.01 * positions + np.sin(positions / 7.0)
    changed_signal = baseline_signal.copy()
    changed_signal[300:] += 10_000.0
    active_signal = [baseline_signal]
    fitted_inputs: list[np.ndarray] = []
    params = _params()
    recording = SubjectRecording(
        subject_id="sub-001",
        session_id="ses-01",
        task="rest",
        edf_path=Path("synthetic.edf"),
    )

    monkeypatch.setattr(
        pipeline_module,
        "discover_edf_recordings",
        lambda _root: {"sub-001": [recording]},
    )
    monkeypatch.setattr(
        pipeline_module,
        "select_subject_recording",
        lambda *_args, **_kwargs: recording,
    )
    monkeypatch.setattr(
        pipeline_module,
        "inspect_raw_recording",
        lambda _path: {"source": "synthetic"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_single_channel",
        lambda *_args, **_kwargs: (active_signal[0].copy(), 100.0),
    )

    def fake_fit(**kwargs) -> WilsonCowanFitResult:
        fitted_inputs.append(np.array(kwargs["observed_signal"], copy=True))
        return WilsonCowanFitResult(
            parameters=params,
            objective=0.1,
            evaluations=1,
            optimizer_success=True,
            optimizer_message="synthetic",
            loss_components={"psd": 0.1},
        )

    monkeypatch.setattr(pipeline_module, "fit_wilson_cowan_psd", fake_fit)
    config = {
        "data_root": "unused",
        "target_channel": "Oz",
        "subject_id": "sub-001",
        "session": None,
        "task": "rest",
        "resample_hz": None,
        "prediction": {"train_fraction": 0.75},
        "psd": {"fmin_hz": 0.5, "fmax_hz": 45.0},
        "wilson_cowan": {
            name: getattr(params, name)
            for name in (
                "tau_e",
                "tau_i",
                "w_ee",
                "w_ei",
                "w_ie",
                "w_ii",
                "p",
                "q",
                "sigmoid_gain",
                "sigmoid_theta",
            )
        },
        "fit": {
            "dt": 0.001,
            "duration_s": 0.02,
            "maxiter": 0,
            "population_size": 1,
            "polish": False,
            "random_seed": 0,
        },
    }

    first = prepare_sub001_experiment(config)
    active_signal[0] = changed_signal
    second = prepare_sub001_experiment(config)

    assert len(fitted_inputs) == 2
    assert np.array_equal(fitted_inputs[0], fitted_inputs[1])
    assert np.array_equal(first.train_signal, second.train_signal)
    assert not np.array_equal(first.test_signal, second.test_signal)


def test_ablation_runner_reuses_prepared_fit_and_writes_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    config["ablation"] = {"modes": list(MODES)}
    prepared = _prepared()
    preparation_calls = 0

    def fake_prepare(_config: dict[str, object]) -> PreparedExperiment:
        nonlocal preparation_calls
        preparation_calls += 1
        return prepared

    monkeypatch.setattr(pipeline_module, "load_config", lambda _path: config)
    monkeypatch.setattr(pipeline_module, "prepare_sub001_experiment", fake_prepare)

    report = run_sub001_ablation("unused.yaml", output_dir=tmp_path)

    assert preparation_calls == 1
    assert [row["reservoir_mode"] for row in report["modes"]] == list(MODES)
    assert (tmp_path / "synthetic-001" / "ablation_summary.json").is_file()
    assert (tmp_path / "synthetic-001" / "ablation_report.md").is_file()


def test_disabled_coupling_mode_matches_factory_independent_mode() -> None:
    prepared = _prepared()
    config = _config()
    disabled_config = deepcopy(config)
    disabled_config["coupling"]["enabled"] = False
    independent = build_reservoir(
        config,
        prepared.fit_result.parameters,
        prepared.continuous_eigenvalues,
        sample_interval_s=1.0 / prepared.sampling_rate_hz,
        reservoir_mode="independent_nonlinear_wc",
    )
    disabled_coupled = build_reservoir(
        disabled_config,
        prepared.fit_result.parameters,
        prepared.continuous_eigenvalues,
        sample_interval_s=1.0 / prepared.sampling_rate_hz,
        reservoir_mode="coupled_nonlinear_wc",
    )
    signal_values = prepared.train_signal[:50]

    assert disabled_coupled.diagnostics["coupling_enabled"] is False
    assert disabled_coupled.diagnostics["coupling_strength"] == 0.0
    assert np.array_equal(
        independent.model.reservoir.run(signal_values),
        disabled_coupled.model.reservoir.run(signal_values),
    )
