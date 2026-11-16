import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Base de datos SQLite temporal con los datos de demostración."""
    monkeypatch.setenv("ENERGY_PREDICT_DB", str(tmp_path / "test.db"))
    from core import database
    database.init_db()
    return database


@pytest.fixture()
def df(db):
    from core import data_processing
    return data_processing.cargar_consumo()
