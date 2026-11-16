"""H. Predicción de factura."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import business, config, database, regression
from core.config import money
from ui import charts, theme


def render(df, equipos, s) -> None:
    theme.header("Mi factura", "Valor estimado del recibo de la luz del próximo mes", "receipt")
    if not theme.requiere(df, 3, "estimar la factura"):
        return
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    with c0:
        estrato = st.selectbox("Estrato de la vivienda", [1, 2, 3, 4, 5, 6], index=int(s.get("estrato", 4)) - 1,
                               key="fac_estrato")
    with c1:
        tarifa = st.number_input("Tarifa por kWh ($)", min_value=1.0, value=float(s["tarifa_kwh"]), step=10.0,
                                 format="%.1f", key="fac_tar")
    with c2:
        cargo = st.number_input("Cargos fijos mensuales ($)", min_value=0.0, value=float(s["cargo_fijo"]), step=1000.0,
                                format="%.0f", key="fac_cargo")
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Guardar como valores predeterminados", key="fac_save"):
            database.set_setting("tarifa_kwh", tarifa)
            database.set_setting("cargo_fijo", cargo)
            database.set_setting("estrato", estrato)
            theme.flash("success", "Tarifa, estrato y cargo fijo guardados.")
            st.rerun()
    implicita = float(df["tarifa_implicita"].tail(3).mean())
    st.caption(f"Tarifa de referencia en {config.CIUDAD}: {config.FUENTE_TARIFA}. Cambia cada mes: use la de su recibo. "
               f"En sus últimos 3 recibos pagó en promedio {money(implicita)} por kWh (ya con subsidio o contribución).")

    pron = regression.pronostico(df)
    et = config.periodo_label(pron["periodo"])
    kwh = {"Bajo (P10)": regression.cuantil_predictivo(pron, 0.10), "Medio (esperado)": pron["yhat"],
           "Alto (P90)": regression.cuantil_predictivo(pron, 0.90)}
    fact = {k: business.factura(v, tarifa, cargo, estrato) for k, v in kwh.items()}
    ultima = float(df["valor_factura"].iloc[-1])
    prom12 = float(df["valor_factura"].tail(12).mean())

    theme.section(f"Escenarios para {et}")
    c = st.columns(4)
    tonos = ["green", "blue", "amber"]
    iconos = ["down", "receipt", "trend"]
    for i, (nombre, valor) in enumerate(fact.items()):
        with c[i]:
            theme.card(f"Escenario {nombre.split(' ')[0].lower()}", money(valor), "", f"{kwh[nombre]:.1f} kWh · {nombre.split(' ', 1)[1]}",
                       iconos[i], tonos[i])
    with c[3]:
        dif = fact["Medio (esperado)"] - ultima
        theme.card("Frente a la última factura", ("+" if dif >= 0 else "−") + money(abs(dif)), "",
                   f"{dif / ultima * 100:+.1f}% vs. {df['etiqueta'].iloc[-1]}" if ultima > 0 else "Sin factura previa",
                   "percent", "red" if dif > 0 else "green")

    medio_kwh = kwh["Medio (esperado)"]
    ajuste = business.subsidio_aplicado(medio_kwh, tarifa, estrato)
    if ajuste > 0:
        detalle = (f"− subsidio de estrato {estrato} ({config.SUBSIDIO[estrato] * 100:.0f} % sobre los primeros "
                   f"{min(medio_kwh, config.SUBSISTENCIA_KWH):.0f} kWh): <b>−{money(ajuste)}</b>")
    elif ajuste < 0:
        detalle = (f"+ contribución de estrato {estrato} ({config.CONTRIBUCION[estrato] * 100:.0f} % sobre todo el "
                   f"consumo): <b>+{money(-ajuste)}</b>")
    else:
        detalle = "estrato 4: sin subsidio ni contribución"
    theme.note(f"<b>Cómo sale el valor esperado:</b> {medio_kwh:.1f} kWh × {money(tarifa)} = "
               f"<b>{money(medio_kwh * tarifa)}</b>; {detalle}; + cargos fijos {money(cargo)} → "
               f"<b>{money(fact['Medio (esperado)'])}</b>.")

    izq, der = st.columns(2)
    with izq:
        theme.plot(charts.escenarios(list(fact), list(fact.values()), s["presupuesto"]), key="fac_esc", ancho=0.5)
    with der:
        theme.plot(charts.facturas_historicas(df, fact["Medio (esperado)"], et + " (est.)"), key="fac_hist", ancho=0.5)

    theme.section("Comparación con facturas anteriores")
    medio = fact["Medio (esperado)"]
    comp = pd.DataFrame({
        "Referencia": [f"Última factura ({df['etiqueta'].iloc[-1]})", "Promedio últimos 12 meses",
                       "Factura máxima histórica", "Factura mínima histórica"],
        "Valor ($)": [ultima, prom12, float(df["valor_factura"].max()), float(df["valor_factura"].min())]})
    comp["Diferencia de la estimación ($)"] = (medio - comp["Valor ($)"]).round(0)
    comp["Diferencia (%)"] = ((medio - comp["Valor ($)"]) / comp["Valor ($)"].replace(0, np.nan) * 100).round(1)
    comp["Valor ($)"] = comp["Valor ($)"].round(0)
    theme.table(comp)

    # Validación retrospectiva de la fórmula con las tarifas implícitas del mes anterior.
    if len(df) >= 4:
        est = df["consumo_kwh"].iloc[1:].to_numpy() * df["tarifa_implicita"].iloc[:-1].to_numpy() + cargo
        real = df["valor_factura"].iloc[1:].to_numpy()
        ok = real > 0
        if ok.any():
            mape = float(np.mean(np.abs(est[ok] - real[ok]) / real[ok]) * 100)
            st.caption(f"Validación retrospectiva: aplicando «consumo × tarifa del mes anterior» a los meses pasados, "
                       f"el error medio frente a las facturas registradas fue de {mape:.1f}%.")

    theme.note("<b>Aviso:</b> esta estimación aplica la tarifa, el subsidio o la contribución del estrato y los cargos "
               "fijos indicados. El recibo real puede ser distinto por el alumbrado público, el aseo, intereses, "
               "ajustes de tarifa del mes y otros cobros del operador. Los escenarios bajo y alto son "
               f"los percentiles 10 y 90 de la distribución predictiva del consumo ({pron['dof']} g.l.); no son cotas "
               "garantizadas.", "amber")
