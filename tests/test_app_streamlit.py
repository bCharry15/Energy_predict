"""Prueba de integración con Streamlit REAL (streamlit.testing.v1.AppTest).

Ejecuta app.py y navega por las 11 pantallas comprobando que ninguno lanza excepciones.
Se omite automáticamente si Streamlit o Plotly no están instalados.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

if "fake" in type(sys.modules.get("streamlit", 0)).__name__.lower():
    pytest.skip("Se están usando los dobles de prueba", allow_module_level=True)
pytest.importorskip("plotly")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGINAS = ["Inicio", "1. Mis electrodomésticos", "2. Mi consumo mensual", "3. Mi factura", "4. Consejos de ahorro",
           "Análisis · Resumen general", "Análisis · Estadística", "Análisis · Regresión lineal",
           "Análisis · Probabilidad", "Análisis · Monte Carlo", "Configuración"]


@pytest.fixture()
def app(db):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


def test_inicio_sin_errores(app):
    assert not app.exception
    assert len(app.error) == 0


@pytest.mark.parametrize("pagina", PAGINAS)
def test_navegacion(app, pagina):
    app.sidebar.radio[0].set_value(pagina).run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.error) == 0, [e.value for e in app.error]


@pytest.mark.parametrize("pagina", PAGINAS)
def test_sin_datos(app, db, pagina):
    db.borrar_todo()
    app.sidebar.radio[0].set_value(pagina).run()
    assert not app.exception, [e.value for e in app.exception]
