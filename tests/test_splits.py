from data.splits import assign_patient_splits


def test_patient_splits_are_reproducible_and_complete() -> None:
    patients = [f"patient-{index}" for index in range(100)]
    first = assign_patient_splits(patients, seed=37)
    second = assign_patient_splits(reversed(patients), seed=37)
    assert first == second
    assert set(first) == set(patients)
    assert set(first.values()) == {"train", "validation", "test"}


def test_different_seed_changes_assignments() -> None:
    patients = [f"patient-{index}" for index in range(100)]
    assert assign_patient_splits(patients, seed=1) != assign_patient_splits(patients, seed=2)

