"""E. Regresión lineal."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core import config, regression
from ui import charts, theme


def render(df, equipos, s) -> None:
    theme.header("Regresión lineal", "Modelo predictivo del consumo eléctrico mensual (mínimos cuadrados)", "trend")
    if not theme.requiere(df, config.MIN_REGISTROS_REGRESION, "entrenar y evaluar la regresión"):
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        feats = st.multiselect("Variables predictoras", list(config.FEATURE_LABELS), default=config.DEFAULT_FEATURES,
                               format_func=lambda f: config.FEATURE_LABELS[f], key="reg_feats")
    with c2:
        frac = st.slider("Proporción de datos para prueba", 0.15, 0.40, 0.25, 0.05, key="reg_frac",
                         help="Los meses más recientes se reservan para evaluar; no participan en el entrenamiento.")
    if not feats:
        st.warning("Seleccione al menos una variable predictora.")
        return
    try:
        res = regression.ajustar(df, feats)
        h = regression.evaluar_holdout(df, feats, frac)
    except ValueError as exc:
        st.error(str(exc))
        return
    wf = regression.walk_forward(df, feats)
    m_wf = regression.metricas_walk_forward(wf)
    X = df[feats].to_numpy(float)
    y = df["consumo_kwh"].to_numpy(float)
    ori = theme.origen(df["es_simulado"], derivado=True)
    card = partial(theme.card, origen=ori)

    theme.section("Evaluación fuera de muestra")
    c = st.columns(4)
    te, tr, base = h["test"], h["train"], h["baseline"]
    with c[0]:
        card("R² en prueba", f"{te['r2']:.3f}", "", f"Entrenamiento: {tr['r2']:.3f}", "target",
                   "green" if te["r2"] > 0.5 else "amber" if te["r2"] > 0 else "red")
    with c[1]:
        card("MAE en prueba", f"{te['mae']:.2f}", "kWh", f"Referencia (media): {base['mae']:.2f} kWh", "activity", "blue")
    with c[2]:
        card("RMSE en prueba", f"{te['rmse']:.2f}", "kWh", f"Referencia (media): {base['rmse']:.2f} kWh", "bars", "violet")
    with c[3]:
        if m_wf:
            card("MAE walk-forward", f"{m_wf['mae']:.2f}", "kWh", f"RMSE {m_wf['rmse']:.2f} · {len(wf)} predicciones", "shuffle", "green")
        else:
            card("MAE walk-forward", "—", "", "Datos insuficientes", "shuffle", "muted")
    mejora = (1 - te["rmse"] / base["rmse"]) * 100 if base["rmse"] > 0 else float("nan")
    veredicto = (f"El modelo reduce el RMSE un <b>{mejora:.0f}%</b> frente a predecir siempre la media."
                 if mejora > 0 else "<b>El modelo no mejora a la media histórica en el conjunto de prueba</b>: "
                 "con estas variables no hay evidencia de capacidad predictiva.")
    theme.note(f"<b>Metodología:</b> partición cronológica — entrenamiento con los primeros {h['n_train']} meses y "
               f"prueba con los últimos {h['n_test']} (nunca vistos por el modelo). Además, validación walk-forward: "
               f"cada mes se predice con un modelo entrenado solo con los meses anteriores. {veredicto} "
               f"Con solo {h['n_test']} meses de prueba, estas métricas tienen alta incertidumbre.")

    theme.section("Modelo ajustado (todo el histórico)")
    theme.equation(res.ecuacion())
    theme.marca_origen(ori)
    # La tabla ocupa todo el ancho para que sus cinco columnas se lean sin desplazarse, aun con letra grande.
    coef = res.tabla_coeficientes()
    coef["Coeficiente"] = coef["Coeficiente"].round(4)
    coef["Error estándar"] = coef["Error estándar"].round(4)
    coef["t"] = coef["t"].round(3)
    coef["Valor p"] = [theme.fmt_p(p) for p in coef["Valor p"]]
    theme.tabla(coef)
    st.markdown(f"- **Intercepto (b₀):** {res.coef[0]:.3f} kWh\n"
                + "\n".join(f"- **Pendiente de {config.FEATURE_LABELS[f].lower()}:** {b:.3f} kWh por unidad"
                            for f, b in zip(feats, res.coef[1:]))
                + f"\n- **R² (ajuste):** {res.r2:.3f} · **R² ajustado:** {res.r2_adj:.3f}"
                + f"\n- **Error estándar residual (s):** {res.sigma:.2f} kWh con {res.dof} g.l.")
    st.caption("Cada pendiente es el cambio esperado del consumo por unidad del predictor, manteniendo constantes "
               "los demás. Es una asociación estadística; no demuestra causalidad.")

    izq, der = st.columns(2)
    with izq:
        fx = st.selectbox("Variable para el diagrama de dispersión", feats,
                          format_func=lambda f: config.FEATURE_LABELS[f], key="reg_fx")
        lr = stats.linregress(df[fx].to_numpy(float), y)
        theme.plot(charts.dispersion(df[fx], y, df["etiqueta"], config.FEATURE_LABELS[fx], lr.slope, lr.intercept,
                                     lr.rvalue, simulado=df["es_simulado"]), key="reg_disp", ancho=0.5)
        st.caption(f"Regresión simple consumo ~ {fx}: pendiente = {lr.slope:.3f}, intercepto = {lr.intercept:.2f}, "
                   f"r = {lr.rvalue:.3f}, R² = {lr.rvalue ** 2:.3f}, p = {theme.fmt_p(lr.pvalue)}.")
    with der:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        theme.plot(charts.real_vs_pred_scatter(y[h["idx_test"]], h["pred_test"],
                                                simulado=df["es_simulado"].to_numpy()[h["idx_test"]]), key="reg_rvp", ancho=0.5)
    theme.plot(charts.ajuste_holdout(df, h), key="reg_hold")

    theme.section("Análisis de residuos y supuestos")
    izq, der = st.columns(2)
    with izq:
        theme.plot(charts.residuos_vs_ajustados(res.fitted, res.residuals, simulado=df["es_simulado"]), key="reg_res", ancho=0.5)
    with der:
        theme.plot(charts.qq_plot(res.residuals, simulado=df["es_simulado"]), key="reg_qq", ancho=0.5)
    diag = pd.DataFrame(regression.diagnosticos(res, X))
    diag["Estadístico"] = diag["Estadístico"].round(3)
    diag["Valor p"] = [theme.fmt_p(p) for p in diag["Valor p"]]
    diag["Resultado"] = diag.pop("cumple").map({True: "Cumple", False: "Revisar"})
    theme.tabla(diag[["Supuesto", "Prueba", "Estadístico", "Valor p", "Resultado", "Interpretación"]])
    if len(feats) > 1:
        v = regression.vif(X)
        st.caption("VIF por variable: " + " · ".join(f"{config.FEATURE_LABELS[f]} = {vi:.2f}" for f, vi in zip(feats, v)))
    theme.marca_origen(ori)

    theme.section("Predicción del próximo mes")
    base_x = regression.valores_proximo_mes(df, feats)
    cols = st.columns(len(feats) + 1)
    x_next = {}
    for i, f in enumerate(feats):
        with cols[i]:
            if f == "horas_uso":
                x_next[f] = st.number_input(config.FEATURE_LABELS[f], 0.0, 24.0, round(base_x[f], 1), 0.1, key="reg_x_h")
            elif f == "num_electrodomesticos":
                x_next[f] = st.number_input(config.FEATURE_LABELS[f], 0, 500, int(base_x[f]), 1, key="reg_x_n")
            else:
                x_next[f] = base_x[f]
                st.number_input(config.FEATURE_LABELS[f], value=int(base_x[f]), disabled=True, key="reg_x_t",
                                help="El próximo mes es el n + 1 de la serie.")
    with cols[-1]:
        conf = st.selectbox("Nivel de confianza", [0.80, 0.90, 0.95], index=1,
                            format_func=lambda c: f"{int(c * 100)}%", key="reg_conf")
    pron = regression.pronostico(df, feats, x_next, conf)
    c = st.columns(3)
    with c[0]:
        card("Consumo predicho", f"{pron['yhat']:.1f}", "kWh", config.periodo_label(pron["periodo"]), "trend", "green")
    with c[1]:
        card(f"Intervalo de predicción {int(conf * 100)}%", f"{pron['lo']:.1f} – {pron['hi']:.1f}", "kWh",
                   f"t de Student con {pron['dof']} g.l.", "target", "blue")
    with c[2]:
        card("Error estándar de predicción", f"{pron['se']:.2f}", "kWh", "Incluye ruido e incertidumbre de coeficientes", "activity", "violet")
    fuera = [config.FEATURE_LABELS[f] for f in feats if f != "tendencia"
             and not (df[f].min() <= x_next[f] <= df[f].max())]
    if fuera:
        st.warning("Extrapolación: " + ", ".join(fuera) + " está fuera del rango observado en el histórico. "
                   "La relación lineal no está validada en esa zona.")

    with st.expander("Justificación del modelo y limitaciones"):
        st.markdown(
            "- **Por qué regresión lineal:** el consumo es aproximadamente proporcional al tiempo de uso y al número de "
            "equipos (energía = potencia × tiempo), por lo que una relación lineal es físicamente razonable.\n"
            "- **Regresión ≠ probabilidad:** la regresión estima el valor *esperado* del consumo dadas las variables; "
            "el módulo de probabilidad cuantifica la *incertidumbre* alrededor de ese valor.\n"
            f"- **Tamaño de muestra:** n = {res.n}. Los valores p y los intervalos suponen residuos independientes, "
            "con varianza constante y aproximadamente normales (ver tabla de supuestos).\n"
            "- **«Tendencia» y «cantidad de electrodomésticos»** suelen estar correlacionadas (ambas crecen con el "
            "tiempo); revise el VIF antes de usarlas juntas.\n"
            "- El modelo no incluye temperatura, número de habitantes ni cambios de equipos; la predicción supone "
            "que el próximo mes se parece a los meses observados.\n"
            "- El modelo final se reajusta con todos los datos para pronosticar; su calidad se informa con las "
            "métricas fuera de muestra, no con el R² de ajuste.")
