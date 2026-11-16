"""G. Simulación de Monte Carlo."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

from core import business, config, montecarlo, probability, regression
from core.config import money
from ui import charts, theme

MODELOS = {"regresion": "Regresión + incertidumbre (recomendado)",
           "distribucion": "Distribución ajustada al histórico",
           "bootstrap": "Bootstrap del histórico (no paramétrico)"}


def render(df, equipos, s) -> None:
    theme.header("Simulación de Monte Carlo", "Escenarios posibles de consumo y factura para el próximo mes", "shuffle")
    if not theme.requiere(df, config.MIN_REGISTROS_ANALISIS, "parametrizar la simulación"):
        return
    x = df["consumo_kwh"].to_numpy(float)
    feats = regression.features_utilizables(df)
    opciones = list(MODELOS) if feats else ["distribucion", "bootstrap"]

    sim_hist = df["es_simulado"]
    n_sim, n = int(sim_hist.sum()), len(df)
    ori = theme.origen(sim_hist, derivado=True)
    card = partial(theme.card, origen=ori)
    entrada = (f"{n} meses ingresados por usted" if n_sim == 0 else
               f"{n} meses simulados de ejemplo, no mediciones reales" if n_sim == n else
               f"{n} meses del histórico: {n_sim} simulados de ejemplo y {n - n_sim} ingresados por usted")
    theme.note(f"<b>Datos de entrada:</b> {entrada}. <b>Resultados:</b> escenarios del próximo mes que la "
               "simulación genera a partir de esos datos; no son meses observados.")

    theme.section("Parámetros de la simulación")
    c1, c2, c3 = st.columns(3)
    with c1:
        modelo = st.selectbox("Modelo generador", opciones, format_func=lambda m: MODELOS[m], key="mc_modelo")
        n_sims = st.select_slider("Número de escenarios a generar", [10000, 20000, 50000, 100000], value=10000,
                                  key="mc_n")
    with c2:
        limite = st.number_input("Límite de consumo (kWh)", min_value=0.0, value=float(s["limite_kwh"]), step=5.0, key="mc_lim")
        presupuesto = st.number_input("Presupuesto de factura ($)", min_value=0.0, value=float(s["presupuesto"]),
                                      step=5000.0, format="%.0f", key="mc_pres")
    with c3:
        tarifa = st.number_input("Tarifa esperada ($/kWh)", min_value=1.0, value=float(s["tarifa_kwh"]), step=10.0, key="mc_tar")
        cv_tar = st.slider("Incertidumbre de la tarifa (± % , 1σ)", 0.0, 15.0, 3.0, 0.5, key="mc_cv") / 100
    semilla = int(st.number_input("Semilla aleatoria (reproducibilidad)", min_value=0, max_value=999999, value=2026, step=1, key="mc_seed"))
    rng = np.random.default_rng(semilla)
    supuestos = []

    if modelo == "regresion":
        res = regression.ajustar(df, feats)
        base = regression.valores_proximo_mes(df, feats)
        sd_h = float(df["horas_uso"].std(ddof=1))
        c1, c2, c3 = st.columns(3)
        centro, sds = dict(base), {}
        with c1:
            if "horas_uso" in feats:
                centro["horas_uso"] = st.number_input("Horas de uso esperadas (media)", 0.0, 24.0,
                                                      round(base["horas_uso"], 1), 0.1, key="mc_h")
                sds["horas_uso"] = st.number_input("Desviación estándar de las horas", 0.0, 12.0, round(sd_h, 2), 0.05,
                                                   key="mc_sdh", help="Por defecto, la desviación estándar histórica.")
        with c2:
            if "num_electrodomesticos" in feats:
                centro["num_electrodomesticos"] = float(st.number_input(
                    "Electrodomésticos el próximo mes", 0, 500, int(base["num_electrodomesticos"]), 1, key="mc_eq"))
            ruido = st.selectbox("Ruido residual", ["bootstrap", "normal"], key="mc_ruido",
                                 format_func=lambda r: "Bootstrap de residuos" if r == "bootstrap" else "Normal(0, s²)")
        with c3:
            inc = st.checkbox("Incluir incertidumbre de los coeficientes", value=True, key="mc_inc")
        if ruido == "normal" and regression.residuos_normales(res) is False:
            st.warning("Los residuos no superan la prueba de normalidad; use el bootstrap de residuos.")
        sims = montecarlo.simular_regresion(res, centro, sds, n_sims, rng, ruido, inc)
        supuestos = [
            f"Consumo = {res.ecuacion().split('= ', 1)[1]} + ε (coeficientes estimados con {res.n} meses).",
            (f"Horas de uso ~ Normal({centro.get('horas_uso', 0):.1f}, {sds.get('horas_uso', 0):.2f}²) truncada a [0, 24]."
             if "horas_uso" in feats else "Sin variable de horas de uso."),
            ("Coeficientes ~ Normal multivariada(b̂, Cov(b̂))." if inc else "Coeficientes fijos en su estimación puntual."),
            ("ε remuestreado de los residuos observados (sin suponer normalidad)." if ruido == "bootstrap"
             else f"ε ~ Normal(0, {res.sigma:.2f}²)."),
        ]
    elif modelo == "distribucion":
        try:
            mejor = probability.ajustar_distribuciones(x)[0]
        except ValueError as exc:
            st.error(str(exc))
            return
        sims = montecarlo.simular_distribucion(mejor["dist"], n_sims, rng)
        supuestos = [f"Consumo ~ {mejor['nombre']}({probability.descripcion_parametros(mejor)}), ajustada por máxima "
                     f"verosimilitud (menor AIC; KS p = {theme.fmt_p(mejor['ks_p'])}).",
                     "Meses independientes e idénticamente distribuidos; no usa información del mes actual."]
    else:
        sims = montecarlo.simular_bootstrap(x, n_sims, rng)
        supuestos = [f"Remuestreo con reemplazo de los {len(x)} consumos del histórico"
                     + (" (simulados de ejemplo)." if n_sim == n else
                        f" ({n_sim} simulados de ejemplo)." if n_sim else "."),
                     "No puede generar valores fuera del rango histórico."]
    supuestos.append(f"Tarifa ~ Normal({tarifa:.0f}, ({cv_tar * 100:.1f}% · tarifa)²), independiente del consumo; "
                     f"cargo fijo {money(s['cargo_fijo'])}." if cv_tar > 0 else
                     f"Tarifa fija de {money(tarifa)}/kWh; cargo fijo {money(s['cargo_fijo'])}.")
    supuestos.append(f"Factura calculada con las reglas de Popayán para estrato {int(s.get('estrato', 4))} "
                     f"(subsidio o contribución; consumo de subsistencia de {config.SUBSISTENCIA_KWH:.0f} kWh).")

    tar_sims = montecarlo.simular_tarifa(tarifa, cv_tar, n_sims, rng)
    estrato = int(s.get("estrato", 4))
    fact = business.factura(sims, tar_sims, s["cargo_fijo"], estrato)
    r, rf = montecarlo.resumen(sims), montecarlo.resumen(fact)
    pl, pp = montecarlo.prob_superar(sims, limite), montecarlo.prob_superar(fact, presupuesto)
    lo90, hi90 = montecarlo.intervalo(sims, 0.90)

    theme.section(f"Resultados · {n_sims:,} escenarios generados".replace(",", "."))
    c = st.columns(4)
    with c[0]:
        card("Consumo esperado", f"{r['media']:.1f}", "kWh", f"σ = {r['desviacion']:.1f} · EE = {r['ee_media']:.2f}", "bolt", "blue")
    with c[1]:
        card("Intervalo central 90%", f"{lo90:.0f} – {hi90:.0f}", "kWh", "Percentiles 5 y 95", "target", "violet")
    with c[2]:
        card("P(consumo > límite)", theme.pct(pl["p"]), "", f"Límite {limite:.0f} kWh · EE ± {pl['ee'] * 100:.2f} pp", "alert",
             riesgo=theme.nivel_riesgo(pl["p"]))
    with c[3]:
        card("P(factura > presupuesto)", theme.pct(pp["p"]), "", f"{money(presupuesto)} · EE ± {pp['ee'] * 100:.2f} pp", "wallet",
             riesgo=theme.nivel_riesgo(pp["p"]))

    izq, der = st.columns(2)
    pc, pf = r["percentiles"], rf["percentiles"]
    with izq:
        theme.plot(charts.mc_histograma(sims, limite, pc[5], pc[50], pc[95], "Escenarios generados: consumo",
                                        "Consumo (kWh)", simulado=sim_hist), key="mc_h1", ancho=0.5)
    with der:
        theme.plot(charts.mc_histograma(fact, presupuesto, pf[5], pf[50], pf[95], "Escenarios generados: factura",
                                        "Factura ($)", simulado=sim_hist), key="mc_h2", ancho=0.5)

    izq, der = st.columns([2, 3])
    with izq:
        theme.section("Percentiles de los escenarios")
        theme.tabla(pd.DataFrame({"Percentil": [f"P{p}" for p in montecarlo.PERCENTILES],
                                  "Consumo (kWh)": [round(pc[p], 1) for p in montecarlo.PERCENTILES],
                                  "Factura ($)": [round(pf[p], 0) for p in montecarlo.PERCENTILES]}))
        theme.marca_origen(ori)
        st.caption(f"Factura esperada: {money(rf['media'])} (σ = {money(rf['desviacion'])}).")
    with der:
        idx, medias = montecarlo.media_acumulada(sims)
        theme.plot(charts.convergencia(idx, medias, simulado=sim_hist), key="mc_conv", ancho=0.6)

    theme.note("<b>Supuestos explícitos de esta simulación</b><br>" + "<br>".join("• " + t for t in supuestos)
               + "<br>EE = error estándar de Monte Carlo, √(p(1−p)/N): mide el error numérico de la simulación, "
               "<b>no</b> la validez del modelo. Los resultados son tan buenos como los supuestos.")
