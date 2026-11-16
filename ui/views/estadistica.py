"""D. Análisis estadístico descriptivo."""
from __future__ import annotations

from functools import partial

import streamlit as st

from core import config, descriptive, probability
from ui import charts, theme

VARIABLES = {"consumo_kwh": ("Consumo mensual", "kWh"), "valor_factura": ("Valor de la factura", "$"),
             "horas_uso": ("Horas de uso diario", "h")}


def _f(v: float, dec: int = 2) -> str:
    if v != v:
        return "—"
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render(df, equipos, s) -> None:
    n_sim, n = int(df["es_simulado"].sum()), len(df)
    subtitulo = ("Estadística descriptiva de las variables registradas" if n_sim == 0 else
                 "Estadística descriptiva de los datos simulados de ejemplo (no son mediciones reales)" if n_sim == n else
                 f"Estadística descriptiva del histórico: {n_sim} de {n} meses son simulados de ejemplo")
    theme.header("Análisis estadístico", subtitulo, "bars")
    if not theme.requiere(df, config.MIN_REGISTROS_ANALISIS, "el análisis descriptivo"):
        return
    ori = theme.origen(df["es_simulado"])
    card = partial(theme.card, origen=ori)
    c1, c2 = st.columns([2, 1])
    with c1:
        var = st.selectbox("Variable aleatoria a analizar", list(VARIABLES), format_func=lambda v: VARIABLES[v][0],
                           key="est_var")
    nombre, unidad = VARIABLES[var]
    x = df[var].to_numpy(float)
    r = descriptive.resumen(x)
    dec = 0 if var == "valor_factura" else 2
    with c2:
        k = st.slider("Número de clases del histograma", 3, 15, descriptive.n_clases_sturges(len(x)), key="est_k",
                      help="Por defecto se usa la regla de Sturges: k = 1 + log2(n).")

    theme.section("Tendencia central")
    tabla = descriptive.tabla_frecuencias(x, k)
    moda = r["moda"]
    c = st.columns(4)
    with c[0]:
        card("Media", _f(r["media"], dec), unidad, f"Error estándar: {_f(r['error_estandar'], dec)}", "activity")
    with c[1]:
        card("Mediana", _f(r["mediana"], dec), unidad, "Percentil 50", "target", "violet")
    with c[2]:
        if moda:
            texto = " / ".join(_f(m, 0) for m in moda[:3])
            card("Moda", texto, unidad, "Valores redondeados a la unidad" + (" (multimodal)" if len(moda) > 1 else ""), "bars", "green")
        else:
            card("Moda", "No aplica", "", "Ningún valor se repite; use la clase modal", "bars", "muted")
    with c[3]:
        card("Clase modal", descriptive.clase_modal(tabla), "", "Intervalo con mayor frecuencia", "grid", "amber")

    theme.section("Dispersión y posición")
    c = st.columns(4)
    with c[0]:
        card("Varianza muestral", _f(r["varianza"], dec), f"{unidad}²", "Divisor n − 1", "shuffle")
    with c[1]:
        card("Desviación estándar", _f(r["desviacion"], dec), unidad, "Raíz de la varianza muestral", "activity", "violet")
    with c[2]:
        cv = r["cv"]
        nivel = "baja" if cv < 10 else "moderada" if cv < 25 else "alta"
        card("Coef. de variación", _f(cv, 1), "%", f"Variabilidad relativa {nivel}", "percent",
                   "green" if cv < 10 else "amber" if cv < 25 else "red")
    with c[3]:
        card("Mínimo – Máximo", f"{_f(r['minimo'], dec)} – {_f(r['maximo'], dec)}", "", f"Rango: {_f(r['rango'], dec)} {unidad}", "trend", "blue")

    izq, der = st.columns([3, 2])
    with izq:
        theme.plot(charts.histograma(x, k, f"Histograma de {nombre.lower()}", f"{nombre} ({unidad})",
                                     media=r["media"], mediana=r["mediana"], simulado=df["es_simulado"]),
                   key="est_hist", ancho=0.6)
    with der:
        theme.plot(charts.caja(x, f"{nombre} ({unidad})", simulado=df["es_simulado"]), key="est_box", ancho=0.4)
        w, p = probability.shapiro(x)
        asim = r["asimetria"]
        forma = "aproximadamente simétrica" if abs(asim) < 0.5 else ("asimétrica a la derecha" if asim > 0 else "asimétrica a la izquierda")
        normal = "no se rechaza la normalidad" if p > 0.05 else "se rechaza la normalidad"
        theme.note(f"<b>Forma:</b> asimetría = {asim:.2f} ({forma}); curtosis en exceso = {r['curtosis']:.2f}. "
                   f"<b>Cuartiles:</b> Q1 = {_f(r['q1'], dec)}, Q3 = {_f(r['q3'], dec)}, IQR = {_f(r['iqr'], dec)}. "
                   f"<b>Shapiro-Wilk:</b> W = {w:.3f}, p = {theme.fmt_p(p)} → {normal} (α = 0,05).")

    theme.section("Distribución de frecuencias")
    vista = tabla.copy()
    for col in ("Marca de clase", "Frec. relativa (%)", "Frec. rel. acumulada (%)"):
        vista[col] = vista[col].round(2)
    theme.tabla(vista)
    theme.marca_origen(ori)

    with st.expander("Fórmulas y limitaciones"):
        st.markdown(
            "- **Media:** x̄ = Σxᵢ / n  ·  **Varianza muestral:** s² = Σ(xᵢ − x̄)² / (n − 1)  ·  "
            "**Desviación estándar:** s = √s²  ·  **CV:** s / x̄ × 100.\n"
            "- Se usa el divisor **n − 1** porque los meses registrados son una muestra del comportamiento de la vivienda.\n"
            "- La **moda** de una variable continua solo tiene sentido si hay valores repetidos; por eso se informa "
            "también la clase modal de la tabla de frecuencias.\n"
            f"- Con n = {r['n']} observaciones, las medidas de forma (asimetría, curtosis) y las pruebas de normalidad "
            "tienen poca potencia: no rechazar la normalidad no demuestra que los datos sean normales.")
