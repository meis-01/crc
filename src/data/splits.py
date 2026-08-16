from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterable

from .io import ManifestRow

SPLIT_NAMES = ("train", "validation", "test")


def normalize_split_name(value: str) -> str:
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    normalized = aliases.get(value.strip().lower(), value.strip().lower())
    if normalized not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {value!r}; expected train, validation, or test")
    return normalized


def assign_patient_splits(
    patient_ids: Iterable[str],
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    ids = sorted(set(str(value) for value in patient_ids))
    if not ids:
        raise ValueError("No patient identifiers available for splitting")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError(f"Invalid split ratios: {ratios}")
    total = sum(ratios)
    normalized = tuple(value / total for value in ratios)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * normalized[0])
    n_validation = int(n * normalized[1])
    if n >= 3:
        n_train = max(1, n_train)
        n_validation = max(1, n_validation)
        if n_train + n_validation >= n:
            n_train = max(1, n - n_validation - 1)
    boundaries = (n_train, n_train + n_validation)
    return {
        patient_id: (
            "train" if index < boundaries[0] else "validation" if index < boundaries[1] else "test"
        )
        for index, patient_id in enumerate(ids)
    }


def resolve_splits(
    rows: list[ManifestRow],
    mode: str = "auto",
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[dict[str, str], str]:
    requested = mode.lower()
    have_official = all(row.official_split.strip() for row in rows)
    official_names = (
        {normalize_split_name(row.official_split) for row in rows} if have_official else set()
    )
    if requested == "auto":
        if have_official and official_names != set(SPLIT_NAMES):
            raise ValueError(
                "The manifest contains a partial official split "
                f"({sorted(official_names)}). Include the non-overlapping CODE-II-test records for an "
                "official train/validation/test design, or explicitly request --split-mode random."
            )
        requested = "official" if have_official else "random"
    if requested == "official":
        if not have_official:
            raise ValueError("Official split requested, but at least one manifest row lacks official_split")
        if official_names != set(SPLIT_NAMES):
            raise ValueError(
                f"Official split requires train, validation, and test; found {sorted(official_names)}"
            )
        assignments: dict[str, str] = {}
        for row in rows:
            split = normalize_split_name(row.official_split)
            previous = assignments.setdefault(row.patient_id, split)
            if previous != split:
                raise ValueError(
                    f"Official split leaks patient {row.patient_id}: {previous} and {split}"
                )
        return assignments, "official"
    if requested != "random":
        raise ValueError("split mode must be auto, official, or random")
    return assign_patient_splits((row.patient_id for row in rows), seed, ratios), "random"


def split_assignment_digest(assignments: dict[str, str]) -> str:
    canonical = "\n".join(f"{patient}\t{assignments[patient]}" for patient in sorted(assignments))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_counts(rows: Iterable[ManifestRow], assignments: dict[str, str]) -> dict[str, dict[str, int]]:
    patients: dict[str, set[str]] = defaultdict(set)
    records: dict[str, int] = defaultdict(int)
    for row in rows:
        split = assignments[row.patient_id]
        patients[split].add(row.patient_id)
        records[split] += 1
    return {
        split: {"patients": len(patients[split]), "records": records[split]}
        for split in SPLIT_NAMES
    }
