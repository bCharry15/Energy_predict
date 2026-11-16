"""I. Recomendaciones inteligentes."""
from __future__ import annotations

import streamlit as st

from core import probability, recommendations, regression
from core.config import money
from ui import theme


def render(df, equipos, s) -> None:
    theme.header("Consejos de ahorro", "Recomendaciones calculadas a partir de sus datos de consumo y sus aparatos", "check")
    if df.empty and equipos.empty:
        st.info("Registre consumos o electrodomésticos para generar recomendaciones.")
        return
    pron, p_lim = None, None
    if len(df) >= 3:
        pron = regression.pronostico(df)
        p_lim = probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], s["limite_kwh"])
    recs = recommendations.generar(df, equipos, s, pron, p_lim)
    kwh, dinero = recommendations.ahorro_potencial(recs)
    n_alta = sum(r["nivel"] == "alta" for r in recs)

    c = st.columns(4)
    with c[0]:
        theme.card("Recomendaciones", str(len(recs)), "", "Generadas con los datos actuales", "check", "blue")
    with c[1]:
        theme.card("Prioridad alta", str(n_alta), "", "Requieren atención", "alert", "red" if n_alta else "green")
    with c[2]:
        theme.card("Ahorro potencial", f"{kwh:.1f}", "kWh/mes", "Si se aplican las oportunidades de ahorro", "down", "green")
    with c[3]:
        theme.card("Ahorro en dinero", money(dinero), "/mes", f"{money(dinero * 12)} al año", "wallet", "green")

    categorias = sorted({r["categoria"] for r in recs})
    filtro = st.multiselect("Filtrar por categoría", categorias, default=categorias, key="rec_cat")
    visibles = [r for r in recs if r["categoria"] in filtro]
    if not visibles:
        st.info("No hay recomendaciones para el filtro seleccionado.")
    for r in visibles:
        theme.recommendation(r)
    theme.note("Las recomendaciones se calculan con reglas sobre sus datos (desviaciones respecto al histórico, "
               "tendencia, probabilidad de superar el límite y participación de cada equipo). El «ahorro potencial» "
               "supone reducir un 20% el uso de los equipos de mayor consumo y es una estimación teórica; las "
               "oportunidades no son necesariamente acumulables entre sí.")
