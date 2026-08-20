from __future__ import annotations

from pathlib import Path

import pytest

from physics_esn.config import load_config


@pytest.fixture
def local_dataset_root() -> Path:
    config = load_config(Path("config") / "defaults.yaml")
    return Path(config["data_root"])
