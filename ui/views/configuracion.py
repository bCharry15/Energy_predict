"""Configuración: parámetros del análisis y administración de datos."""
from __future__ import annotations

import streamlit as st

from core import config, database
from ui import theme


def render(df, equipos, s) -> None:
    theme.header("Configuración", "Tarifa, límites del análisis y administración de los datos", "sliders")
    with st.form("form_settings"):
        st.markdown("**Datos de la vivienda y límites**")
        estrato = st.selectbox("Estrato de la vivienda (aparece en el recibo)", [1, 2, 3, 4, 5, 6],
                               index=int(s.get("estrato", 4)) - 1)
        c1, c2 = st.columns(2)
        with c1:
            tarifa = st.number_input("Tarifa por kWh ($)", min_value=1.0, value=float(s["tarifa_kwh"]), step=10.0, format="%.1f")
            limite = st.number_input("Límite de consumo mensual (kWh)", min_value=1.0, value=float(s["limite_kwh"]), step=5.0, format="%.0f")
        with c2:
            cargo = st.number_input("Cargos fijos mensuales ($)", min_value=0.0, value=float(s["cargo_fijo"]), step=1000.0, format="%.0f")
            presupuesto = st.number_input("Presupuesto mensual de factura ($)", min_value=0.0, value=float(s["presupuesto"]), step=5000.0, format="%.0f")
        if st.form_submit_button("Guardar configuración", type="primary"):
            for k, v in (("tarifa_kwh", tarifa), ("limite_kwh", limite), ("cargo_fijo", cargo), ("presupuesto", presupuesto),
                         ("estrato", estrato)):
                database.set_setting(k, v)
            theme.flash("success", "Configuración guardada.")
            st.rerun()

    theme.note(f"<b>Reglas de {config.CIUDAD}.</b> Operador: {config.OPERADOR}. Tarifa de referencia: "
               f"{config.FUENTE_TARIFA}; cambia cada mes. Consumo de subsistencia: {config.SUBSISTENCIA_KWH:.0f} kWh al "
               "mes (municipios a más de 1.000 m de altura). Subsidio sobre ese consumo: hasta 60 % en estrato 1, "
               "50 % en estrato 2 y 15 % en estrato 3. El estrato 4 paga tarifa plena; los estratos 5 y 6 pagan "
               "una contribución del 20 %.")

    theme.section("Datos")
    sim_c = int(df["es_simulado"].sum()) if len(df) else 0
    sim_e = int(equipos["es_simulado"].sum()) if len(equipos) else 0
    theme.note(f"Base de datos SQLite: <b>{len(df)}</b> registros de consumo ({sim_c} simulados) y <b>{len(equipos)}</b> "
               f"electrodomésticos ({sim_e} simulados). Los datos de demostración son <b>sintéticos</b>: se generan con "
               "un modelo estadístico conocido y no corresponden a mediciones reales de ninguna vivienda.", "amber")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.form("form_reset_demo"):
            st.markdown("**Restaurar demostración**")
            st.caption("Borra todo y vuelve a cargar los datos simulados y la configuración inicial.")
            ok = st.checkbox("Confirmo", key="cfg_ok_demo")
            if st.form_submit_button("Restaurar datos de demostración"):
                if ok:
                    database.restablecer_demo()
                    theme.flash("success", "Datos de demostración restaurados.")
                    st.rerun()
                else:
                    st.error("Marque «Confirmo» para continuar.")
    with c2:
        with st.form("form_del_sim"):
            st.markdown("**Quitar datos simulados**")
            st.caption("Elimina solo los registros de demostración y conserva los que usted ingresó.")
            ok = st.checkbox("Confirmo", key="cfg_ok_sim")
            if st.form_submit_button("Eliminar datos simulados"):
                if ok:
                    database.borrar_simulados()
                    theme.flash("success", "Datos simulados eliminados.")
                    st.rerun()
                else:
                    st.error("Marque «Confirmo» para continuar.")
    with c3:
        with st.form("form_del_all"):
            st.markdown("**Vaciar la base de datos**")
            st.caption("Elimina todos los consumos y electrodomésticos para empezar desde cero.")
            ok = st.checkbox("Confirmo", key="cfg_ok_all")
            if st.form_submit_button("Eliminar todos los datos"):
                if ok:
                    database.borrar_todo()
                    theme.flash("success", "Se eliminaron todos los datos.")
                    st.rerun()
                else:
                    st.error("Marque «Confirmo» para continuar.")
