"""A. Dashboard principal."""
from __future__ import annotations

import streamlit as st

from core import analysis, business, config, recommendations, regression
from core.config import money
from ui import charts, theme


def render(df, equipos, s) -> None:
    theme.header("Resumen general", "Visión general del consumo eléctrico de la vivienda", "grid")
    if not theme.requiere(df, 3, "calcular los indicadores"):
        return
    r = analysis.resumen_general(df, s)
    pron = r["pronostico"]
    et_pron = config.periodo_label(pron["periodo"])
    wf = regression.walk_forward(df, pron["features"])
    m_wf = regression.metricas_walk_forward(wf)
    var = r["variacion_pct"]
    n_sim, n = int(df["es_simulado"].sum()), len(df)
    meses_txt = (f"{n} meses registrados" if n_sim == 0 else f"{n} meses simulados" if n_sim == n
                 else f"{n} meses ({n_sim} simulados)")
    sim_wf = wf["periodo"].map(dict(zip(df["periodo"], df["es_simulado"]))).to_numpy()
    ori_actual = theme.origen(df["es_simulado"].iloc[-1:])
    ori_var = theme.origen(df["es_simulado"].iloc[-2:])
    ori_dato = theme.origen(df["es_simulado"])
    ori_calc = theme.origen(df["es_simulado"], derivado=True)
    sim_eq = list(equipos["es_simulado"]) if "es_simulado" in equipos else []
    ori_recs = theme.origen(list(df["es_simulado"]) + sim_eq, derivado=True, unidad="registros")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.card("Consumo actual", f"{r['actual']:.1f}", "kWh", f"{df['etiqueta'].iloc[-1]}", "bolt", "blue",
                   pill=f"{var:+.1f}%", pill_tono="red" if var > 0 else "green", origen=ori_actual)
    with c2:
        theme.card("Promedio mensual", f"{r['promedio']:.1f}", "kWh", meses_txt, "activity", "violet", origen=ori_dato)
    with c3:
        theme.card("Consumo proyectado", f"{pron['yhat']:.1f}", "kWh",
                   f"{et_pron} · IP 90%: {pron['lo']:.0f}–{pron['hi']:.0f}", "trend", "green", origen=ori_calc)
    with c4:
        dif = r["factura_estimada"] - float(df["valor_factura"].iloc[-1])
        theme.card("Factura estimada", money(r["factura_estimada"]), "", f"{et_pron} · tarifa {money(s['tarifa_kwh'])}/kWh",
                   "receipt", "blue", pill=("+" if dif >= 0 else "−") + money(abs(dif)),
                   pill_tono="red" if dif > 0 else "green", origen=ori_calc)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.card("Variación mensual", f"{var:+.1f}", "%",
                   f"{r['anterior']:.1f} → {r['actual']:.1f} kWh", "percent", "red" if var > 0 else "green", origen=ori_var)
    with c2:
        p = r["p_limite"]
        theme.card("Prob. de superar el límite", theme.pct(p), "", f"Límite: {s['limite_kwh']:.0f} kWh · {et_pron}",
                   "alert", riesgo=theme.nivel_riesgo(p), origen=ori_calc)
    with c3:
        p = r["p_presupuesto"]
        theme.card("Prob. de superar presupuesto", theme.pct(p), "", f"Presupuesto: {money(s['presupuesto'])}",
                   "wallet", riesgo=theme.nivel_riesgo(p), origen=ori_calc)
    with c4:
        if m_wf:
            theme.card("Error típico del modelo", f"{m_wf['mae']:.1f}", "kWh",
                       f"MAE fuera de muestra · {len(wf)} meses", "target", "violet", origen=ori_calc)
        else:
            theme.card("Error típico del modelo", "—", "", "Faltan datos para validar", "target", "muted")

    izq, der = st.columns([3, 2])
    with izq:
        theme.plot(charts.historico(df, pron, s["limite_kwh"], et_pron), key="dash_hist", ancho=0.6)
    with der:
        if len(wf) >= 2:
            theme.plot(charts.real_vs_predicho(wf, sim_wf), key="dash_wf", ancho=0.4)
        else:
            st.info("La comparación real vs. predicho aparecerá cuando haya más registros (mínimo 8).")

    modelo = "regresión lineal con " + ", ".join(config.FEATURE_LABELS[f].lower() for f in pron["features"]) \
        if pron["features"] else "la media histórica (aún no hay datos suficientes para la regresión)"
    aviso = "" if r["residuos_normales"] in (True, None) else \
        " <b>Atención:</b> los residuos no superan la prueba de normalidad; interprete las probabilidades con cautela."
    theme.note(f"La proyección usa <b>{modelo}</b>, suponiendo para {et_pron} las horas de uso promedio de los "
               "últimos 3 meses. Las probabilidades provienen de la distribución predictiva t de Student del modelo. "
               "La curva «predicho» se obtiene entrenando solo con meses anteriores a cada mes evaluado." + aviso)

    izq, der = st.columns([2, 3])
    tabla = business.tabla_electrodomesticos(equipos)
    with izq:
        if tabla.empty:
            st.info("Registre electrodomésticos para ver su participación en el consumo.")
        else:
            theme.plot(charts.barras_equipos(tabla, 7), key="dash_eq", ancho=0.4)
    with der:
        theme.section("Recomendaciones destacadas")
        theme.marca_origen(ori_recs)
        recs = recommendations.generar(df, equipos, s, pron, r["p_limite"])
        for rec in recs[:3]:
            theme.recommendation(rec)
        if not recs:
            st.info("No hay recomendaciones con los datos actuales.")
