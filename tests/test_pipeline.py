from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import h5py
import numpy as np

from data.constants import CANONICAL_LEADS
from data.dataset import invert_record, load_reconstruction_sample
from data.inspection import inspect_manifest
from data.io import read_manifest
from data.preparation import prepare_dataset
from data.validation import validate_prepared_dataset


def test_inspection_verifies_actual_source_order(synthetic_manifest: tuple[Path, Path]) -> None:
    manifest, raw = synthetic_manifest
    summary, issues = inspect_manifest(read_manifest(manifest), raw)
    assert summary["patients"] == 12
    assert summary["manifest_records"] == 12
    assert summary["valid_records"] == 12
    assert summary["sampling_rate_counts_hz"] == {"300": 6, "500": 6}
    assert len(summary["source_lead_order_counts"]) == 2
    assert summary["nan_values"] == 0
    assert summary["infinite_values"] == 0
    assert all(issue["issue"] == "duration_outside_published_range" for issue in issues)


def test_prepare_validate_and_target_exclusion(synthetic_manifest: tuple[Path, Path], tmp_path: Path) -> None:
    manifest, raw = synthetic_manifest
    output = tmp_path / "prepared.h5"
    summary = prepare_dataset(
        read_manifest(manifest),
        output,
        raw_root=raw,
        target_lead="V2",
        target_rate=400,
        normalization="training_set",
        split_mode="auto",
        split_seed=17,
    )
    assert summary["split_mode"] == "official"
    assert summary["split_counts"] == {
        "train": {"patients": 6, "records": 6},
        "validation": {"patients": 3, "records": 3},
        "test": {"patients": 3, "records": 3},
    }
    assert summary["native_sampling_rates_hz"] == [300.0, 500.0]

    validation = validate_prepared_dataset(output, target_lead="V2", expected_rate=400)
    assert validation["passed"], validation
    with h5py.File(output, "r") as handle:
        assert handle["ecg"].dtype == np.dtype("float32")
        assert handle["ecg"].shape == (12 * 400, 12)
        assert list(handle["metadata/length"][...]) == [400] * 12
        sample = load_reconstruction_sample(handle, 0, "V2")
        assert sample["inputs"].shape == (400, 11)
        assert sample["target"].shape == (400,)
        assert "V2" not in sample["input_lead_names"]
        assert sample["target_lead"] == "V2"
        restored = invert_record(handle, 0)
        assert restored.shape == (400, 12)
        assert np.isfinite(restored).all()


def test_mixed_rates_require_explicit_resampling(
    synthetic_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    manifest, raw = synthetic_manifest
    try:
        prepare_dataset(read_manifest(manifest), tmp_path / "bad.h5", raw_root=raw)
    except ValueError as exc:
        assert "Mixed native sampling rates" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected mixed-rate preparation to require --target-rate")


def test_diagnostic_plot_cli(synthetic_manifest: tuple[Path, Path], tmp_path: Path) -> None:
    manifest, raw = synthetic_manifest
    dataset = tmp_path / "prepared.h5"
    figures = tmp_path / "figures"
    prepare_dataset(
        read_manifest(manifest),
        dataset,
        raw_root=raw,
        target_rate=400,
        target_lead="V2",
        split_mode="official",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_diagnostics.py"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            str(dataset),
            "--target-lead",
            "V2",
            "--output-dir",
            str(figures),
            "--per-split",
            "1",
        ],
        check=True,
        env=environment,
    )
    outputs = sorted(figures.glob("*.png"))
    assert len(outputs) == 3
    assert all(path.stat().st_size > 0 for path in outputs)
