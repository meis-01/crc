from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import mne
import numpy as np


SUBJECT_DIRECTORY_PATTERN = re.compile(r"sub-\d{3}")


@dataclass(frozen=True)
class SubjectRecording:
    subject_id: str
    session_id: str
    task: str
    edf_path: Path


def discover_subject_dirs(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and SUBJECT_DIRECTORY_PATTERN.fullmatch(path.name)
    )


def discover_edf_recordings(data_root: str | Path) -> dict[str, list[SubjectRecording]]:
    recordings: dict[str, list[SubjectRecording]] = {}
    for subject_dir in discover_subject_dirs(data_root):
        subject_recordings: list[SubjectRecording] = []
        for edf_path in sorted(subject_dir.glob("ses-*/eeg/*_eeg.edf")):
            session_id = edf_path.parents[1].name
            filename = edf_path.stem
            task = "unknown"
            for token in filename.split("_"):
                if token.startswith("task-"):
                    task = token.removeprefix("task-")
                    break
            subject_recordings.append(
                SubjectRecording(
                    subject_id=subject_dir.name,
                    session_id=session_id,
                    task=task,
                    edf_path=edf_path,
                )
            )
        recordings[subject_dir.name] = subject_recordings
    return recordings


def select_subject_recording(
    recordings: dict[str, list[SubjectRecording]],
    subject_id: str,
    session_id: str | None = None,
    task: str | None = None,
) -> SubjectRecording:
    subject_recordings = recordings.get(subject_id, [])
    if not subject_recordings:
        raise KeyError(f"No recordings found for subject {subject_id}.")
    filtered = [
        recording
        for recording in subject_recordings
        if (session_id is None or recording.session_id == session_id)
        and (task is None or recording.task == task)
    ]
    if not filtered:
        raise KeyError(f"No recording matched subject={subject_id}, session={session_id}, task={task}.")
    return filtered[0]


def inspect_raw_recording(edf_path: str | Path) -> dict[str, Any]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    try:
        return {
            "path": str(edf_path),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "channel_names": list(raw.ch_names),
            "duration_s": float(raw.n_times / raw.info["sfreq"]),
        }
    finally:
        raw.close()


def load_single_channel(edf_path: str | Path, channel_name: str) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    try:
        if channel_name not in raw.ch_names:
            raise KeyError(f"Channel {channel_name} not found in {edf_path}.")
        picked = raw.copy().pick([channel_name])
        signal = picked.get_data()[0].astype(np.float64, copy=True)
        return signal, float(picked.info["sfreq"])
    finally:
        raw.close()
