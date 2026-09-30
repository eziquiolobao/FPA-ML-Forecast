import json
from pathlib import Path

import pandas as pd
import pytest

from fpa.generate import generate_raw
from fpa.ingest import load_raw, validate
from fpa.pipeline import ROOT, load_settings

MARTS = ROOT / "data" / "marts"


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def committed_as_of() -> pd.Period:
    """The close period of the committed marts (what the deployed app shows)."""
    meta = json.loads((MARTS / "run_metadata.json").read_text())
    return pd.Period(meta["as_of"], "M")


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory, settings, committed_as_of) -> Path:
    out = tmp_path_factory.mktemp("raw")
    generate_raw(settings, committed_as_of, out)
    return out


@pytest.fixture(scope="session")
def raw(raw_dir):
    return validate(load_raw(raw_dir))
