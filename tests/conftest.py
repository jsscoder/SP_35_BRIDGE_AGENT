from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def kb():
    from sp35 import load_kb
    return load_kb()


@pytest.fixture(scope="session")
def golden():
    return yaml.safe_load((ROOT / "kb/sp35_2024/golden_examples.yaml").read_text(encoding="utf-8"))
