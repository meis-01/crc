from __future__ import annotations

from pathlib import Path

from physics_esn.data.loader import discover_edf_recordings, load_single_channel


def test_discover_edf_recordings_for_all_subjects(local_dataset_root: Path) -> None:
    recordings = discover_edf_recordings(local_dataset_root)
    assert list(recordings) == [f"sub-{index:03d}" for index in range(1, 11)]
    assert all(recordings[subject_id] for subject_id in recordings)
    assert recordings["sub-001"][0].edf_path.name.endswith("_eeg.edf")


def test_oz_extraction(local_dataset_root: Path) -> None:
    recordings = discover_edf_recordings(local_dataset_root)
    edf_path = recordings["sub-001"][0].edf_path
    signal, sampling_rate_hz = load_single_channel(edf_path, "Oz")
    assert signal.ndim == 1
    assert signal.size > 1000
    assert sampling_rate_hz > 0.0
