"""Prueba de humo de TODAS las vistas ejecutando app.py de principio a fin.

- Si Streamlit y Plotly están instalados, este archivo se omite: use test_app_streamlit.py,
  que ejecuta la aplicación real con streamlit.testing.
- Si no lo están, se usan dobles mínimos (tests/_stubs) para ejecutar todo el código de la
  interfaz: detecta errores de lógica, de datos y de uso de widgets, pero NO valida el render.
"""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REAL = importlib.util.find_spec("streamlit") is not None and "fake" not in type(sys.modules.get("streamlit", 0)).__name__.lower()
if REAL and importlib.util.find_spec("plotly") is not None:
    pytest.skip("Streamlit real instalado: se usa test_app_streamlit.py", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent / "_stubs"))
import fake_streamlit  # noqa: E402

PAGINAS = ["Inicio", "1. Mis electrodomésticos", "2. Mi consumo mensual", "3. Mi factura", "4. Consejos de ahorro",
           "Análisis · Resumen general", "Análisis · Estadística", "Análisis · Regresión lineal",
           "Análisis · Probabilidad", "Análisis · Monte Carlo", "Configuración"]


@pytest.fixture()
def st(db):
    fake = fake_streamlit.install()
    for mod in [m for m in sys.modules if m == "ui" or m.startswith("ui.")]:
        del sys.modules[mod]
    return fake


def run(st, pagina, **overrides):
    st.reset_run()
    st.session_state["nav"] = pagina
    st.overrides = overrides
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except fake_streamlit.RerunException:
        return "rerun"
    return "ok"


def sin_fallos(st):
    assert st.messages("exception") == [], st.messages("error")


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_con_datos_demo(st, pagina):
    assert run(st, pagina) == "ok"
    sin_fallos(st)
    assert st.messages("error") == []


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 7, 9, 12, 14])
@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_con_pocos_datos(st, db, pagina, n):
    """Con 0..14 registros ninguna vista debe fallar: avisa o degrada el modelo."""
    with db.connect() as c:
        c.execute("DELETE FROM consumo WHERE id > ?", (n,))
        if n == 0:
            c.execute("DELETE FROM electrodomesticos")
    assert run(st, pagina) == "ok"
    sin_fallos(st)


def test_graficos_presentes(st):
    esperados = {"Análisis · Resumen general": 3, "1. Mis electrodomésticos": 2, "Análisis · Estadística": 2, "Análisis · Regresión lineal": 5,
                 "Análisis · Probabilidad": 3, "Análisis · Monte Carlo": 3, "3. Mi factura": 2}
    for pagina, minimo in esperados.items():
        run(st, pagina)
        assert st.charts >= minimo, (pagina, st.charts)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_todos_los_botones_pulsados(st, pagina):
    """Pulsar todos los botones a la vez (con confirmaciones) no debe producir excepciones."""
    st.press = lambda label: True
    st.check_all = True
    run(st, pagina)
    sin_fallos(st)


def test_flujo_crud_consumo(st, db):
    st.press = lambda label: label == "Guardar registro"
    assert run(st, "2. Mi consumo mensual", add_anio=2026, add_mes="Nov", add_kwh=207.5, add_fac=190000.0,
               add_n=13, add_h=6.4) == "rerun"
    df = db.listar_consumo()
    fila = df[df["periodo"] == "2026-11"].iloc[0]
    assert fila["consumo_kwh"] == 207.5 and fila["es_simulado"] == 0
    assert st.session_state["_flash"][0] == "success"

    # periodo duplicado -> error de validación, sin insertar
    assert run(st, "2. Mi consumo mensual", add_anio=2026, add_mes="Nov", add_kwh=100.0, add_fac=1.0, add_n=1, add_h=1.0) == "ok"
    assert any("Ya existe" in m for m in st.messages("error"))
    # consumo 0 -> error de validación
    run(st, "2. Mi consumo mensual", add_anio=2027, add_mes="Ene", add_kwh=0.0)
    assert any("mayor que 0" in m for m in st.messages("error"))
    assert len(db.listar_consumo()) == 37

    rid = int(fila["id"])
    st.press = lambda label: label == "Guardar cambios"
    assert run(st, "2. Mi consumo mensual", edit_consumo_sel=rid, **{f"edit{rid}_kwh": 250.0}) == "rerun"
    assert db.listar_consumo().set_index("id").loc[rid, "consumo_kwh"] == 250.0

    st.press = lambda label: label == "Eliminar registro"
    assert run(st, "2. Mi consumo mensual", edit_consumo_sel=rid) == "ok"       # sin confirmar: no borra
    assert len(db.listar_consumo()) == 37
    st.check_all = True
    assert run(st, "2. Mi consumo mensual", edit_consumo_sel=rid) == "rerun"
    assert len(db.listar_consumo()) == 36


def test_flujo_crud_electrodomesticos(st, db):
    from core import business
    otro = len(business.CATALOGO)            # índice de «Otro (escribir el nombre)»
    st.press = lambda label: label == "Guardar aparato"
    assert run(st, "1. Mis electrodomésticos", eq_tipo=business.OTRO, **{f"eqadd{otro}_nom": "Aire acondicionado",
               f"eqadd{otro}_pot": 1200.0, f"eqadd{otro}_hor": 3.0, f"eqadd{otro}_dias": 20, f"eqadd{otro}_cant": 1}) == "rerun"
    eq = db.listar_electrodomesticos()
    assert "Aire acondicionado" in set(eq["nombre"])
    assert "72.0 kWh" in st.session_state["_flash"][1]          # 1200 W × 3 h × 20 días / 1000
    assert run(st, "1. Mis electrodomésticos", eq_tipo=business.OTRO) == "ok"   # nombre vacío
    assert any("obligatorio" in m for m in st.messages("error"))
    # un aparato del catálogo se guarda con sus valores habituales sin escribir nada
    assert run(st, "1. Mis electrodomésticos", eq_tipo="Cafetera") == "rerun"
    caf = db.listar_electrodomesticos().query("nombre == 'Cafetera'").iloc[0]
    assert (caf["potencia_w"], caf["horas_dia"], caf["es_simulado"]) == (800, 0.25, 0)
    rid = int(eq[eq["nombre"] == "Aire acondicionado"]["id"].iloc[0])
    st.press = lambda label: label == "Borrar aparato"
    st.check_all = True
    assert run(st, "1. Mis electrodomésticos", edit_eq_sel=rid) == "rerun"
    assert "Aire acondicionado" not in set(db.listar_electrodomesticos()["nombre"])


def test_registro_sugerido_desde_los_aparatos(st, db):
    """Sin escribir nada, «Guardar registro» usa el consumo calculado con los electrodomésticos."""
    from core import business
    sug = business.sugerencia_registro(db.listar_electrodomesticos())
    st.press = lambda label: label == "Guardar registro"
    assert run(st, "2. Mi consumo mensual") == "rerun"
    fila = db.listar_consumo().query("periodo == '2026-11'").iloc[0]
    assert fila["consumo_kwh"] == sug["consumo_kwh"] and fila["num_electrodomesticos"] == 13
    assert fila["valor_factura"] == round(business.factura_s(sug["consumo_kwh"], db.get_settings()) / 100) * 100


def test_inicio_botones_navegan(st):
    st.press = lambda label: label == "Ir a Mi factura"
    run(st, "Inicio")
    assert st.session_state["nav"] == "3. Mi factura"
    st.press = lambda label: False
    for letra in (False, True):
        st.session_state["letra_grande"] = letra
        assert run(st, "Inicio") == "ok"
        assert any("ep-big" in m for m in st.messages("markdown"))


def test_factura_por_estrato(st):
    for e in (1, 2, 3, 4, 5, 6):
        assert run(st, "3. Mi factura", fac_estrato=e) == "ok"
        sin_fallos(st)
        assert st.messages("error") == []


def test_importar_csv(st, db):
    class Archivo:
        def getvalue(self):
            return (b"periodo,consumo_kwh,valor_factura,num_electrodomesticos,horas_uso\n"
                    b"2027-01,180.5,160000,12,5.5\n2027-02,-4,1,1,1\n")
    st.press = lambda label: label == "Importar filas válidas"
    assert run(st, "2. Mi consumo mensual", csv_up=Archivo()) == "rerun"
    df = db.listar_consumo()
    assert "2027-01" in set(df["periodo"]) and "2027-02" not in set(df["periodo"])


def test_configuracion_guarda_y_restaura(st, db):
    st.press = lambda label: label == "Guardar configuración"
    assert run(st, "Configuración") == "rerun"
    assert db.get_settings()["estrato"] == 3
    st.press = lambda label: label == "Eliminar todos los datos"
    st.check_all = True
    assert run(st, "Configuración") == "rerun"
    assert db.listar_consumo().empty
    st.press = lambda label: label == "Restaurar datos de demostración"
    assert run(st, "Configuración") == "rerun"
    assert len(db.listar_consumo()) == 36 and db.listar_consumo()["es_simulado"].all()


def test_regresion_con_distintas_variables(st):
    for feats in (["horas_uso"], ["tendencia"], ["horas_uso", "num_electrodomesticos", "tendencia"], []):
        assert run(st, "Análisis · Regresión lineal", reg_feats=feats) == "ok"
        sin_fallos(st)
        assert st.messages("error") == []


def test_montecarlo_modelos(st):
    for modelo in ("regresion", "distribucion", "bootstrap"):
        for n in (10000, 100000):
            assert run(st, "Análisis · Monte Carlo", mc_modelo=modelo, mc_n=n, mc_ruido="normal") == "ok"
            sin_fallos(st)
