"""ENERGY PREDICT — Sistema Inteligente de Predicción del Consumo Eléctrico Residencial.

Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from core import config, data_processing, database
from ui import theme
from ui.views import (configuracion, consumo, dashboard, electrodomesticos, estadistica, factura, inicio,
                      montecarlo, probabilidad, recomendaciones, regresion)

st.set_page_config(page_title="ENERGY PREDICT", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Primero lo que usa cualquier persona (pasos 1 a 4); después el análisis estadístico.
VISTAS = {
    "Inicio": inicio,
    "1. Mis electrodomésticos": electrodomesticos,
    "2. Mi consumo mensual": consumo,
    "3. Mi factura": factura,
    "4. Consejos de ahorro": recomendaciones,
    "Análisis · Resumen general": dashboard,
    "Análisis · Estadística": estadistica,
    "Análisis · Regresión lineal": regresion,
    "Análisis · Probabilidad": probabilidad,
    "Análisis · Monte Carlo": montecarlo,
    "Configuración": configuracion,
}


def main() -> None:
    theme.inject_css(bool(st.session_state.get("letra_grande", False)))
    try:
        database.init_db()
        df = data_processing.cargar_consumo()
        equipos = database.listar_electrodomesticos()
        settings = database.get_settings()
    except Exception as exc:  # error de base de datos: se informa sin romper la app
        st.error(f"No se pudo abrir la base de datos ({config.db_path()}): {exc}")
        st.stop()
        return

    with st.sidebar:
        theme.brand()
        pagina = st.radio("Navegación", list(VISTAS), key="nav", label_visibility="collapsed")
        simulados = int(df["es_simulado"].sum()) if len(df) else 0
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.checkbox("Letra más grande", key="letra_grande", help="Aumenta el tamaño de todos los textos.")
        if simulados:
            st.markdown(theme.badge(f"Datos de ejemplo: {simulados} de {len(df)} meses", "amber", "info"), unsafe_allow_html=True)
            st.caption("Los meses de ejemplo son simulados (una casa de Popayán), no mediciones reales. "
                       "Puede cambiarlos por los suyos.")
        elif len(df):
            st.markdown(theme.badge(f"Datos del usuario: {len(df)} meses", "green", "check"), unsafe_allow_html=True)
        else:
            st.markdown(theme.badge("Sin registros de consumo", "muted", "info"), unsafe_allow_html=True)
        st.caption(f"{config.CIUDAD} · Proyecto académico de Probabilidad y Estadística")

    theme.show_flash()
    try:
        VISTAS[pagina].render(df, equipos, settings)
    except Exception as exc:  # cualquier fallo de un módulo se muestra de forma controlada
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        st.error(f"Ocurrió un error al mostrar «{pagina}»: {exc}")
        with st.expander("Detalle técnico"):
            st.exception(exc)


main()
