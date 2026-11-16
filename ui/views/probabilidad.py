"""F. Análisis de probabilidad."""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import stats

from core import business, config, probability, regression
from core.config import money
from ui import charts, theme


def _tres_probabilidades(x, mejor, pron, limite: float, normales, unidad: str, ori: str = "") -> None:
    emp = probability.prob_empirica(x, limite)
    par = probability.prob_parametrica(mejor["dist"], limite)
    pred = probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], limite)
    c = st.columns(3)
    with c[0]:
        theme.card("Empírica (frecuencia)", theme.pct(emp["p"]), "",
                   f"{emp['k']} de {emp['n']} meses · IC 95%: {theme.pct(emp['lo'])}–{theme.pct(emp['hi'])}", "bars", "blue", origen=ori)
    with c[1]:
        theme.card(f"Distribución {mejor['nombre']}", theme.pct(par), "", "P(X > límite) = 1 − F(límite), un mes cualquiera", "activity", "violet", origen=ori)
    with c[2]:
        theme.card("Predictiva próximo mes", theme.pct(pred), "",
                   f"t({pron['dof']}) centrada en {pron['yhat']:.1f} kWh" + ("" if normales in (True, None) else " · residuos no normales"),
                   "trend", riesgo=theme.nivel_riesgo(pred), origen=ori)


def render(df, equipos, s) -> None:
    theme.header("Análisis de probabilidad", "Probabilidades de superar límites de consumo y de costo", "percent")
    if not theme.requiere(df, config.MIN_REGISTROS_ANALISIS, "ajustar modelos probabilísticos"):
        return
    x = df["consumo_kwh"].to_numpy(float)
    try:
        ajustes = probability.ajustar_distribuciones(x)
    except ValueError as exc:
        st.error(str(exc))
        return
    mejor = ajustes[0]
    pron = regression.pronostico(df)
    normales = regression.residuos_normales(pron["modelo"])
    w, p_sw = probability.shapiro(x)
    ori = theme.origen(df["es_simulado"], derivado=True)

    theme.section("1 · Modelo probabilístico del consumo mensual")
    izq, der = st.columns([3, 2])
    with izq:
        theme.plot(charts.histograma(x, max(6, int(np.ceil(1 + np.log2(len(x))))), "Histórico y distribuciones ajustadas (MLE)",
                                     "Consumo (kWh)", dists=ajustes, simulado=df["es_simulado"]), key="prob_fit", ancho=0.6)
    with der:
        normal_txt = ("no se rechaza la normalidad" if p_sw > 0.05 else
                      "<b>se rechaza la normalidad</b>; no debe usarse la distribución normal")
        aviso = "" if mejor["ks_p"] > 0.05 else " <b>Ninguna distribución ajusta bien: prefiera la probabilidad empírica.</b>"
        theme.note(f"Se elige la distribución <b>{mejor['nombre']}</b> ({probability.descripcion_parametros(mejor)}) "
                   f"por tener el menor AIC. Shapiro-Wilk sobre el consumo: W = {w:.3f}, p = {theme.fmt_p(p_sw)} → "
                   f"{normal_txt}.{aviso} Con n = {len(x)}, las diferencias de AIC menores que 2 no son concluyentes.")
    # Tabla de ajustes a todo el ancho: seis columnas no caben legibles en la columna angosta.
    t = probability.tabla_ajustes(ajustes)
    t["Log-verosimilitud"] = t["Log-verosimilitud"].round(2)
    t["AIC"] = t["AIC"].round(2)
    t["KS (D)"] = t["KS (D)"].round(3)
    t["Valor p KS"] = [theme.fmt_p(v) for v in t["Valor p KS"]]
    theme.tabla(t)
    theme.marca_origen(ori)

    theme.section("2 · Probabilidad de superar un límite de consumo")
    lo, hi = float(np.floor(x.min() * 0.8)), float(np.ceil(x.max() * 1.25))
    limite = st.slider("Límite de consumo (kWh)", lo, hi, float(np.clip(s["limite_kwh"], lo, hi)), 1.0, key="prob_lim")
    _tres_probabilidades(x, mejor, pron, limite, normales, "kWh", ori)
    izq, der = st.columns(2)
    xmin, xmax = min(lo, pron["yhat"] - 4 * pron["se"]), max(hi, pron["yhat"] + 4 * pron["se"])
    with izq:
        theme.plot(charts.cola_superior(mejor["dist"].pdf, xmin, xmax, limite,
                                        probability.prob_parametrica(mejor["dist"], limite),
                                        f"Distribución {mejor['nombre']} del consumo mensual", "Consumo (kWh)",
                                        mejor["nombre"], simulado=df["es_simulado"]), key="prob_cola1", ancho=0.5)
    with der:
        tdist = stats.t(pron["dof"], loc=pron["yhat"], scale=pron["se"])
        theme.plot(charts.cola_superior(tdist.pdf, xmin, xmax, limite,
                                        probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], limite),
                                        f"Distribución predictiva de {config.periodo_label(pron['periodo'])}",
                                        "Consumo (kWh)", "t predictiva", simulado=df["es_simulado"]), key="prob_cola2", ancho=0.5)
    theme.note("<b>Cómo leerlo:</b> la probabilidad <b>empírica</b> y la <b>paramétrica</b> describen un mes cualquiera "
               "del histórico (distribución marginal). La <b>predictiva</b> se refiere específicamente al próximo mes "
               "y usa la regresión con las horas de uso recientes, por eso puede diferir. El área sombreada es la "
               "probabilidad de superar el límite.")

    theme.section("3 · Probabilidad de superar un costo")
    c1, c2, c3 = st.columns(3)
    with c1:
        presupuesto = st.number_input("Límite de factura ($)", min_value=0.0, value=float(s["presupuesto"]),
                                      step=5000.0, format="%.0f", key="prob_pres")
    with c2:
        tarifa = st.number_input("Tarifa ($/kWh)", min_value=1.0, value=float(s["tarifa_kwh"]), step=10.0,
                                 format="%.1f", key="prob_tar")
    with c3:
        cargo = st.number_input("Cargo fijo ($)", min_value=0.0, value=float(s["cargo_fijo"]), step=1000.0,
                                format="%.0f", key="prob_cargo")
    estrato = int(s.get("estrato", 4))
    kwh_eq = business.kwh_para_presupuesto(presupuesto, tarifa, cargo, estrato)
    theme.note(f"Con tarifa determinista y estrato {estrato}, pagar más de <b>{money(presupuesto)}</b> equivale a "
               f"consumir más de <b>{kwh_eq:.1f} kWh</b>. Como la factura crece con el consumo, "
               "P(Factura > B) = P(X > consumo equivalente). La incertidumbre de la tarifa se trata en el módulo "
               "Monte Carlo.")
    _tres_probabilidades(x, mejor, pron, kwh_eq, normales, "$", ori)

    theme.section("4 · Probabilidad de aumentar respecto al mes anterior")
    emp = probability.prob_aumento_empirica(x)
    p_pred = probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], float(x[-1]))
    c = st.columns(3)
    with c[0]:
        theme.card("Frecuencia histórica de aumentos", theme.pct(emp["p"]), "",
                   f"{emp['k']} de {emp['n']} cambios mensuales · IC 95%: {theme.pct(emp['lo'])}–{theme.pct(emp['hi'])}", "bars", origen=ori)
    with c[1]:
        theme.card("P(próximo mes > mes actual)", theme.pct(p_pred), "",
                   f"Predicción {pron['yhat']:.1f} vs. actual {x[-1]:.1f} kWh", "trend",
                   riesgo=theme.nivel_riesgo(p_pred, medio=.5), origen=ori)
    with c[2]:
        theme.card("Consumo del mes actual", f"{x[-1]:.1f}", "kWh", df["etiqueta"].iloc[-1], "bolt", "violet",
                   origen=theme.origen(df["es_simulado"].iloc[-1:]))
    theme.note("La frecuencia histórica trata cada cambio mensual como un ensayo de Bernoulli (modelo binomial). "
               "La segunda probabilidad condiciona al nivel actual: tras un mes inusualmente alto, lo esperable es "
               "que el siguiente sea menor (regresión a la media).")

    with st.expander("Supuestos y limitaciones"):
        st.markdown(
            "- Los ajustes paramétricos suponen meses **independientes e idénticamente distribuidos**; si el consumo "
            "tiene tendencia o estacionalidad, la probabilidad marginal mezcla regímenes distintos.\n"
            "- El valor p de Kolmogorov-Smirnov es optimista porque los parámetros se estiman con los mismos datos.\n"
            "- La probabilidad predictiva solo es válida si los residuos de la regresión son aproximadamente normales "
            "y los valores supuestos de los predictores para el próximo mes son razonables.\n"
            f"- Con n = {len(x)} meses, las probabilidades de eventos raros (colas) son muy inciertas: observe la "
            "amplitud del intervalo de Wilson de la probabilidad empírica.")
