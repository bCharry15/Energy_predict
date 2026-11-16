"""Visualizaciones Plotly con el tema de ENERGY PREDICT. Solo construyen figuras."""
from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from .theme import AMBER, BLUE, BLUE_SOFT, BORDER, GREEN, MUTED, RED, TEXT, VIOLET

GRID = "rgba(138,155,184,.24)"
# Texto secundario de las gráficas: más claro que MUTED para leerse sobre la superficie oscura.
TICK = "#A9B8D3"      # números de los ejes
EJE = "#C3CFE4"       # títulos de los ejes y leyenda
# Escala tipográfica de las gráficas (px con letra normal; «Letra más grande» la multiplica por ESCALA_GRANDE).
TIPO = {"base": 13, "titulo": 15, "aviso": 13, "eje": 13, "tick": 12, "leyenda": 12.5, "anotacion": 12}
ESCALA_GRANDE = 1.25
_AVISO_RE = re.compile(r"<br><span style='font-size:\d+px;color:[^']+'>(.*)</span>$")
PALETA = [BLUE, GREEN, AMBER, VIOLET, "#22D3EE", "#F472B6", "#94A3B8"]


# ------------------------------------------------------- procedencia de los datos
# Los meses de demostración llevan es_simulado = 1. Ninguna serie construida con ellos
# puede rotularse como «Real» o «Registrado».
def _procedencia(simulado) -> tuple[int, int]:
    """(meses simulados, meses totales) de una serie; (0, 0) si no se conoce el origen."""
    if simulado is None:
        return 0, 0
    marcas = np.asarray(simulado, dtype=float)
    return int(np.nansum(marcas > 0)), len(marcas)


def _rotulo(simulado, original: str) -> str:
    k, n = _procedencia(simulado)
    if k == 0:
        return original
    return "Histórico simulado" if k == n else f"Histórico ({k} de {n} meses simulados)"


def _corto(serie: str) -> str:
    """Nombre breve para títulos y ejes; el conteo de meses simulados ya va en el aviso."""
    return serie if serie in ("Real", "Histórico simulado") else "Histórico"


def _aviso(simulado, unidad: str = "meses") -> str:
    k, n = _procedencia(simulado)
    if k == 0:
        return ""
    if k == n:
        return "Datos simulados de ejemplo: no son mediciones reales"
    return f"{k} de {n} {unidad} son simulados de ejemplo, no mediciones reales"


def _aviso_derivado(simulado, que: str) -> str:
    """Aviso para resultados calculados a partir del histórico (modelos ajustados, escenarios)."""
    k, n = _procedencia(simulado)
    if k == 0:
        return ""
    if k == n:
        return f"{que} a partir de datos simulados de ejemplo"
    return f"{que} a partir de {n} meses: {k} simulados y {n - k} ingresados por usted"


def _con_aviso(titulo: str, aviso: str) -> str:
    return f"{titulo}<br><span style='font-size:{TIPO['aviso']}px;color:{AMBER}'>{aviso}</span>" if aviso else titulo


def _titulo(titulo: str, simulado) -> str:
    return _con_aviso(titulo, _aviso(simulado))


def _origen_meses(simulado, si: str = "Mes simulado (ejemplo)", no: str = "Mes ingresado por usted") -> np.ndarray | None:
    """Texto por punto para el hover; None si no hay datos simulados."""
    k, _ = _procedencia(simulado)
    if k == 0:
        return None
    return np.where(np.asarray(simulado, dtype=float) > 0, si, no)


def _base(fig: go.Figure, alto: int = 360, titulo: str | None = None, xt: str | None = None,
          yt: str | None = None, leyenda: bool = True) -> go.Figure:
    titulo = titulo or ""
    m = _AVISO_RE.search(titulo)
    principal, aviso = (titulo[:m.start()], m.group(1)) if m else (titulo, "")
    fig.update_layout(
        height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=TIPO["base"]),
        margin=dict(l=54, r=22, t=(82 if aviso else 52) if titulo else 24, b=46),
        title=dict(text=titulo, x=0.01, xanchor="left", font=dict(size=TIPO["titulo"], color="#FFFFFF")),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=EJE, size=TIPO["leyenda"])),
        hoverlabel=dict(bgcolor="#0E1626", bordercolor=BORDER, font=dict(color=TEXT, size=TIPO["base"])),
        hovermode="closest", bargap=0.06,
        # Datos para que theme.plot reparta título, aviso y leyenda según el ancho real de la columna.
        meta=dict(ep_titulo=principal, ep_aviso=aviso, ep_alto=alto),
    )
    fig.update_xaxes(title_text=xt, gridcolor=GRID, zeroline=False, linecolor=BORDER, automargin=True,
                     tickfont=dict(color=TICK, size=TIPO["tick"]), title_font=dict(color=EJE, size=TIPO["eje"]))
    fig.update_yaxes(title_text=yt, gridcolor=GRID, zeroline=False, linecolor=BORDER, automargin=True,
                     tickfont=dict(color=TICK, size=TIPO["tick"]), title_font=dict(color=EJE, size=TIPO["eje"]))
    return ajustar(fig)


def _partir(texto: str, max_car: int) -> list[str]:
    """Parte un texto en el menor número de líneas de como máximo `max_car` caracteres, sin cortar
    palabras y con longitudes parejas (evita dejar una palabra sola en la última línea)."""
    n = len(_partir_voraz(texto, max_car))
    for objetivo in range(-(-len(texto) // max(n, 1)), max_car + 1):
        lineas = _partir_voraz(texto, objetivo)
        if len(lineas) <= n:
            return lineas
    return _partir_voraz(texto, max_car)


def _partir_voraz(texto: str, max_car: int) -> list[str]:
    lineas, actual = [], ""
    for palabra in texto.split():
        if actual and len(actual) + 1 + len(palabra) > max_car:
            lineas.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    return lineas + ([actual] if actual else [])


def _escalar_fuentes(fig: go.Figure, f: float) -> None:
    lay = fig.layout
    for fuente in (lay.font, lay.title.font, lay.legend.font, lay.hoverlabel.font):
        if fuente.size:
            fuente.size = round(fuente.size * f, 1)
    for nombre in [k for k in lay.to_plotly_json() if k.startswith(("xaxis", "yaxis"))]:
        eje = lay[nombre]
        for fuente in (eje.tickfont, eje.title.font):
            if fuente.size:
                fuente.size = round(fuente.size * f, 1)
    for anot in lay.annotations or ():
        anot.font.size = round((anot.font.size or TIPO["anotacion"]) * f, 1)
    for traza in fig.data:
        if getattr(traza, "textfont", None) is not None:
            traza.textfont.size = round((traza.textfont.size or TIPO["tick"]) * f, 1)


def ajustar(fig: go.Figure, ancho_px: float = 800, grande: bool = False) -> go.Figure:
    """Coloca título, aviso de procedencia y leyenda en franjas propias sobre el área de la gráfica,
    partiendo las líneas para que se lean completas en `ancho_px`; con `grande` agranda todo el texto.
    Las fuentes se agrandan una sola vez aunque se llame varias veces (el estado queda en `meta`)."""
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else None
    if meta is None or "ep_alto" not in meta:
        return fig
    if grande and not meta.get("ep_grande"):
        _escalar_fuentes(fig, ESCALA_GRANDE)
        # Con letra grande caben menos fechas: se espacian para que no se pongan en vertical.
        for nombre in [k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")]:
            if fig.layout[nombre].nticks:
                fig.layout[nombre].nticks = max(4, int(fig.layout[nombre].nticks * 0.6))
    f = ESCALA_GRANDE if grande else 1.0
    s_t, s_a, s_l = TIPO["titulo"] * f, TIPO["aviso"] * f, TIPO["leyenda"] * f
    util = max(ancho_px - 28, 160)
    # Ancho medio de carácter medido en el navegador ≈ 0,50 em; se reserva un margen para no cortar.
    lin_t = _partir(meta["ep_titulo"], int(util / (0.56 * s_t))) if meta["ep_titulo"] else []
    lin_a = _partir(meta["ep_aviso"], int(util / (0.55 * s_a))) if meta["ep_aviso"] else []
    partes = list(lin_t) + [f"<span style='font-size:{s_a:g}px;color:{AMBER}'>{x}</span>" for x in lin_a]
    alto_titulo = (len(lin_t) * s_t * 1.3 + len(lin_a) * s_a * 1.35) if partes else 0
    # Leyenda horizontal debajo del título: se estima cuántas filas ocupa al ancho disponible.
    con_leyenda = [tr for tr in fig.data if tr.name and tr.showlegend is not False]
    filas, alto_leyenda = 0, 0.0
    if fig.layout.showlegend is not False and con_leyenda and fig.layout.legend.orientation == "h":
        max_car = max(int((util - 46) / (0.56 * s_l)), 12)
        x, lineas_fila = 0.0, 1
        filas = 1
        for tr in con_leyenda:
            # Un nombre más ancho que la gráfica se parte en varias líneas (Plotly no lo hace solo).
            partes_n = _partir(tr.name.replace("<br>", " "), max_car)
            tr.name = "<br>".join(partes_n)
            w = 46 + 0.56 * s_l * max(len(s) for s in partes_n)
            if x and x + w > util:
                alto_leyenda += lineas_fila * s_l * 1.3 + 8
                filas, x, lineas_fila = filas + 1, 0.0, 1
            x += w
            lineas_fila = max(lineas_fila, len(partes_n))
        alto_leyenda += lineas_fila * s_l * 1.3 + 8
    alto = int(meta["ep_alto"] * (1.12 if grande else 1.0))
    arriba = round(0.8 * s_t) + 4  # espacio sobre la primera línea, proporcional a la letra
    sup = int(arriba + alto_titulo + (8 if partes and filas else 0) + alto_leyenda + 12) if (partes or filas) else 24
    total = alto + max(0, sup - 52)
    fig.update_layout(height=total, margin=dict(t=sup, l=54, r=int(meta.get("ep_der", 22) * f), b=int(64 * f)),
                      title=dict(text="<br>".join(partes), yref="container", y=1 - arriba / total, yanchor="top",
                                 x=0.01, xref="container", xanchor="left"))
    if filas:
        y_ley = 1 - (arriba + alto_titulo + (8 if partes else 0)) / total
        fig.update_layout(legend=dict(yref="container", y=y_ley, yanchor="top",
                                      xref="container", x=0.01, xanchor="left"))
    fig.layout.meta = {**meta, "ep_grande": bool(grande or meta.get("ep_grande", False))}
    return fig


def _hline(fig, y, texto, color, dash="dash"):
    fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=1.3, annotation_text=texto,
                  annotation_position="top left",
                  annotation_font=dict(color=TICK if color == MUTED else color, size=TIPO["anotacion"]))


def _vline(fig, x, texto, color, dash="dash", pos="top right"):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.4, annotation_text=texto,
                  annotation_position=pos,
                  annotation_font=dict(color=TICK if color == MUTED else color, size=TIPO["anotacion"]))


# ------------------------------------------------------------------ dashboard
def historico(df, pron: dict | None = None, limite: float | None = None, etiqueta_pron: str = "") -> go.Figure:
    simulado = df["es_simulado"] if "es_simulado" in df else None
    origen = _origen_meses(simulado)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["etiqueta"], y=df["consumo_kwh"], mode="lines+markers", name=_rotulo(simulado, "Consumo registrado"),
        line=dict(color=BLUE, width=2.6, shape="spline", smoothing=0.5), marker=dict(size=5, color=BLUE),
        fill="tozeroy", fillcolor="rgba(59,130,246,.08)", customdata=origen,
        hovertemplate="%{x}<br><b>%{y:.1f} kWh</b>" + ("<br>%{customdata}" if origen is not None else "")
                      + "<extra></extra>"))
    if pron is not None:
        fig.add_trace(go.Scatter(
            x=[df["etiqueta"].iloc[-1], etiqueta_pron], y=[df["consumo_kwh"].iloc[-1], pron["yhat"]],
            mode="lines", line=dict(color=GREEN, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[etiqueta_pron], y=[pron["yhat"]], mode="markers", name="Proyección (IP 90%)",
            marker=dict(size=11, color=GREEN, line=dict(color="#FFFFFF", width=1.5)),
            error_y=dict(type="data", symmetric=False, array=[pron["hi"] - pron["yhat"]],
                         arrayminus=[pron["yhat"] - pron["lo"]], color=GREEN, thickness=1.6, width=6),
            hovertemplate="%{x}<br><b>%{y:.1f} kWh</b> (proyectado)<extra></extra>"))
    _hline(fig, float(df["consumo_kwh"].mean()), "Promedio", MUTED, "dot")
    if limite:
        _hline(fig, limite, "Límite", AMBER)
    lo = float(min(df["consumo_kwh"].min(), pron["lo"] if pron else np.inf))
    hi = float(max(df["consumo_kwh"].max(), pron["hi"] if pron else -np.inf, limite or -np.inf))
    margen = (hi - lo) * 0.15 or 10
    fig.update_yaxes(range=[max(lo - margen, 0), hi + margen])
    fig.update_xaxes(type="category", nticks=14)
    return _base(fig, 380, _titulo("Histórico de consumo y proyección", simulado), yt="kWh")


def real_vs_predicho(wf, simulado=None) -> go.Figure:
    """`simulado`: marca es_simulado de cada mes de `wf`, en el mismo orden."""
    serie = _rotulo(simulado, "Real")
    origen = _origen_meses(simulado)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wf["etiqueta"], y=wf["real"], mode="lines+markers", name=serie,
                             line=dict(color=BLUE, width=2.4), marker=dict(size=5), customdata=origen,
                             hovertemplate="%{x}<br>" + serie + ": <b>%{y:.1f} kWh</b>"
                                           + ("<br>%{customdata}" if origen is not None else "") + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=wf["etiqueta"], y=wf["predicho"], mode="lines+markers", name="Predicho (fuera de muestra)",
                             line=dict(color=GREEN, width=2.2, dash="dash"), marker=dict(size=5, symbol="diamond"),
                             hovertemplate="%{x}<br>Predicho: <b>%{y:.1f} kWh</b><extra></extra>"))
    fig.update_xaxes(type="category", nticks=12)
    titulo = "Consumo real vs. predicho (walk-forward)" if serie == "Real" else f"{_corto(serie)} vs. predicho (walk-forward)"
    return _base(fig, 380, _titulo(titulo, simulado), yt="kWh")


def barras_equipos(tabla, top: int = 10) -> go.Figure:
    t = tabla.head(top).iloc[::-1]
    colores = [BLUE] * len(t)
    if len(colores):
        colores[-1] = GREEN
    # Origen de los aparatos mostrados; los de demostración llevan es_simulado = 1.
    simulado = t["es_simulado"] if "es_simulado" in t else None
    origen = _origen_meses(simulado, "Electrodoméstico de ejemplo (simulado)", "Electrodoméstico ingresado por usted")
    pct = t["participacion_pct"].to_numpy(float)
    fig = go.Figure(go.Bar(
        x=t["consumo_kwh"], y=t["nombre"], orientation="h", marker=dict(color=colores, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pct], textposition="outside", textfont=dict(color=TICK, size=TIPO["tick"]),
        customdata=pct if origen is None else np.column_stack([pct, origen]),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} kWh/mes · "
                      + ("%{customdata:.1f}%" if origen is None else "%{customdata[0]:.1f}%<br>%{customdata[1]}")
                      + "<extra></extra>"))
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(range=[0, float(t["consumo_kwh"].max()) * 1.15 if len(t) else 1], nticks=4, tickangle=0)
    fig = _base(fig, max(300, 34 * len(t) + 90),
                _con_aviso("Participación por electrodoméstico", _aviso(simulado, "electrodomésticos")),
                xt="kWh / mes", leyenda=False)
    # Margen derecho reservado para la etiqueta de porcentaje de la barra más larga («25.6%»).
    fig.layout.meta = {**fig.layout.meta, "ep_der": 58}
    return ajustar(fig)


def dona_equipos(tabla, top: int = 6) -> go.Figure:
    t = tabla.head(top)
    etiquetas, valores = list(t["nombre"]), list(t["consumo_kwh"])
    resto = float(tabla["consumo_kwh"].iloc[top:].sum())
    if resto > 0:
        etiquetas.append("Otros")
        valores.append(resto)
    fig = go.Figure(go.Pie(labels=etiquetas, values=valores, hole=0.62, sort=False,
                           marker=dict(colors=PALETA, line=dict(color="#0E1626", width=2)),
                           textinfo="percent", textfont=dict(color="#FFFFFF", size=TIPO["tick"]),
                           hovertemplate="<b>%{label}</b><br>%{value:.1f} kWh · %{percent}<extra></extra>"))
    fig = _base(fig, 360, "Distribución del consumo estimado")
    fig.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0))
    return fig


# ---------------------------------------------------------------- estadística
def histograma(x, k: int, titulo: str, xt: str, dists: list[dict] | None = None, media: float | None = None,
               mediana: float | None = None, simulado=None) -> go.Figure:
    """`simulado`: marca es_simulado de cada valor de `x`, en el mismo orden."""
    x = np.asarray(x, dtype=float)
    conteos, bordes = np.histogram(x, bins=k)
    ancho = bordes[1] - bordes[0]
    densidad = dists is not None
    alturas = conteos / (len(x) * ancho) if densidad else conteos
    n_sim, n = _procedencia(simulado)
    hover = "Clase centrada en %{x:.1f}<br>Frecuencia: <b>%{customdata[0]}</b>"
    if 0 < n_sim < n:  # mezcla: cuántos meses de cada clase son de ejemplo y cuántos del usuario
        sim_clase, _ = np.histogram(x[np.asarray(simulado, dtype=float) > 0], bins=bordes)
        hover += "<br>%{customdata[1]} simulados · %{customdata[2]} ingresados por usted"
    else:
        sim_clase = conteos if n_sim else np.zeros_like(conteos)
    fig = go.Figure(go.Bar(
        x=(bordes[:-1] + bordes[1:]) / 2, y=alturas, width=ancho * 0.96, name=_rotulo(simulado, "Datos"),
        marker=dict(color="rgba(59,130,246,.55)", line=dict(color=BLUE, width=1)),
        customdata=np.column_stack([conteos, sim_clase, conteos - sim_clase]),
        hovertemplate=hover + "<extra></extra>"))
    if dists:
        xs = np.linspace(bordes[0] - ancho, bordes[-1] + ancho, 300)
        for i, d in enumerate(dists):
            fig.add_trace(go.Scatter(x=xs, y=d["dist"].pdf(xs), mode="lines", name=d["nombre"],
                                     line=dict(color=[GREEN, AMBER, VIOLET][i % 3], width=2.2,
                                               dash="solid" if i == 0 else "dot"),
                                     hovertemplate=d["nombre"] + ": %{y:.4f}<extra></extra>"))
    if media is not None:
        _vline(fig, media, "Media", GREEN)
    if mediana is not None:
        _vline(fig, mediana, "Mediana", AMBER, "dot", "top left")
    return _base(fig, 380, _titulo(titulo, simulado), xt=xt, yt="Densidad" if densidad else "Frecuencia")


def caja(x, nombre: str, simulado=None) -> go.Figure:
    fig = go.Figure(go.Box(x=np.asarray(x, dtype=float), name="", boxpoints="all", jitter=0.4, pointpos=0,
                           marker=dict(color=BLUE_SOFT, size=5, opacity=0.7), line=dict(color=BLUE),
                           fillcolor="rgba(59,130,246,.18)", boxmean=True,
                           hovertemplate="%{x:.1f}<extra></extra>"))
    return _base(fig, 240, _titulo("Diagrama de caja (cuartiles y atípicos)", simulado), xt=nombre, leyenda=False)


# ------------------------------------------------------------------ regresión
def dispersion(x, y, etiquetas, xt: str, pendiente: float, intercepto: float, r: float,
               simulado=None) -> go.Figure:
    """`simulado`: marca es_simulado de cada observación, en el mismo orden."""
    x = np.asarray(x, dtype=float)
    origen = _origen_meses(simulado)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=_rotulo(simulado, "Observaciones"), text=etiquetas,
                             customdata=origen,
                             marker=dict(size=9, color=BLUE, opacity=0.85, line=dict(color="#0E1626", width=1)),
                             hovertemplate="%{text}<br>x = %{x:.2f}<br><b>%{y:.1f} kWh</b>"
                                           + ("<br>%{customdata}" if origen is not None else "") + "<extra></extra>"))
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=intercepto + pendiente * xs, mode="lines",
                             name=f"Recta de regresión simple (r = {r:.2f})", line=dict(color=GREEN, width=2.6),
                             hoverinfo="skip"))
    return _base(fig, 400, _titulo("Diagrama de dispersión y línea de regresión", simulado), xt=xt,
                 yt="Consumo (kWh)")


def ajuste_holdout(df, h: dict) -> go.Figure:
    simulado = df["es_simulado"] if "es_simulado" in df else None
    serie = _rotulo(simulado, "Real")
    origen = _origen_meses(simulado)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["etiqueta"], y=df["consumo_kwh"], mode="lines+markers", name=serie,
                             line=dict(color=BLUE, width=2.2), marker=dict(size=5), customdata=origen,
                             hovertemplate="%{x}<br>" + serie + ": <b>%{y:.1f} kWh</b>"
                                           + ("<br>%{customdata}" if origen is not None else "") + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["etiqueta"].iloc[h["idx_train"]], y=h["pred_train"], mode="lines",
                             name="Ajuste (entrenamiento)", line=dict(color=MUTED, width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=df["etiqueta"].iloc[h["idx_test"]], y=h["pred_test"], mode="lines+markers",
                             name="Predicción (prueba)", line=dict(color=GREEN, width=2.4),
                             marker=dict(size=7, symbol="diamond")))
    fig.add_vrect(x0=h["n_train"] - 0.5, x1=len(df) - 0.5, fillcolor="rgba(34,197,94,.06)", line_width=0,
                  annotation_text="Conjunto de prueba", annotation_position="top left",
                  annotation_font=dict(color=GREEN, size=TIPO["anotacion"]))
    fig.update_xaxes(type="category", nticks=14)
    titulo = ("Valores reales vs. predichos (partición cronológica)" if serie == "Real"
              else f"{_corto(serie)} vs. predicciones (partición cronológica)")
    return _base(fig, 380, _titulo(titulo, simulado), yt="kWh")


def real_vs_pred_scatter(real, pred, titulo: str | None = None, simulado=None) -> go.Figure:
    """`simulado`: marca es_simulado de cada mes de prueba, en el mismo orden que `real`."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    lo, hi = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))
    m = (hi - lo) * 0.08 or 1
    serie = _rotulo(simulado, "Real")
    origen = _origen_meses(simulado)
    if titulo is None:
        titulo = ("Predicho vs. real (conjunto de prueba)" if serie == "Real"
                  else f"Predicho vs. {_corto(serie).lower()} (conjunto de prueba)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo - m, hi + m], y=[lo - m, hi + m], mode="lines", name="Predicción perfecta",
                             line=dict(color=MUTED, dash="dash", width=1.4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=real, y=pred, mode="markers", name="Meses de prueba", customdata=origen,
                             marker=dict(size=10, color=GREEN, line=dict(color="#0E1626", width=1)),
                             hovertemplate=serie + " %{x:.1f} · Predicho %{y:.1f}"
                                           + ("<br>%{customdata}" if origen is not None else "") + "<extra></extra>"))
    return _base(fig, 380, _titulo(titulo, simulado), xt=f"{_corto(serie)} (kWh)", yt="Predicho (kWh)")


def residuos_vs_ajustados(ajustados, residuos, simulado=None) -> go.Figure:
    """`simulado`: marcas es_simulado de los meses con que se ajustó el modelo."""
    fig = go.Figure(go.Scatter(x=ajustados, y=residuos, mode="markers",
                               marker=dict(size=8, color=BLUE, opacity=0.85),
                               hovertemplate="Ajustado %{x:.1f}<br>Residuo %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_dash="dash", line_width=1.2)
    return _base(fig, 330, _con_aviso("Residuos vs. valores ajustados", _aviso_derivado(simulado, "Calculado")),
                 xt="Valor ajustado (kWh)", yt="Residuo (kWh)", leyenda=False)


def qq_plot(residuos, simulado=None) -> go.Figure:
    """`simulado`: marcas es_simulado de los meses con que se ajustó el modelo."""
    (osm, osr), (pend, inter, _) = stats.probplot(np.asarray(residuos, float), dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=8, color=BLUE), name="Residuos",
                             hovertemplate="Teórico %{x:.2f}<br>Observado %{y:.2f}<extra></extra>"))
    xs = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=xs, y=inter + pend * xs, mode="lines", line=dict(color=GREEN, width=2),
                             name="Referencia normal", hoverinfo="skip"))
    return _base(fig, 330, _con_aviso("Gráfico Q-Q normal de los residuos", _aviso_derivado(simulado, "Calculado")),
                 xt="Cuantiles teóricos", yt="Cuantiles observados", leyenda=False)


# ---------------------------------------------------------------- probabilidad
def cola_superior(pdf, x_min: float, x_max: float, limite: float, p: float, titulo: str, xt: str,
                  nombre: str, simulado=None) -> go.Figure:
    """`simulado`: marcas es_simulado del histórico con que se ajustó la distribución."""
    xs = np.linspace(x_min, x_max, 400)
    ys = pdf(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nombre, line=dict(color=BLUE, width=2.4),
                             fill="tozeroy", fillcolor="rgba(59,130,246,.08)", hoverinfo="skip"))
    cola = xs >= limite
    if cola.any():
        fig.add_trace(go.Scatter(x=np.concatenate([[limite], xs[cola]]),
                                 y=np.concatenate([[float(pdf(np.array([limite]))[0])], ys[cola]]),
                                 mode="lines", name=f"P(X > límite) = {p * 100:.1f}%",
                                 line=dict(color=AMBER, width=0), fill="tozeroy",
                                 fillcolor="rgba(245,158,11,.45)", hoverinfo="skip"))
    _vline(fig, limite, f"Límite {limite:,.0f}".replace(",", "."), AMBER)
    return _base(fig, 360, _con_aviso(titulo, _aviso_derivado(simulado, "Modelo ajustado")), xt=xt,
                 yt="Densidad de probabilidad")


# ----------------------------------------------------------------- Monte Carlo
def mc_histograma(sims, limite: float | None, p5: float, p50: float, p95: float, titulo: str, xt: str,
                  bins: int = 60, simulado=None) -> go.Figure:
    """`sims`: escenarios generados por la simulación. `simulado`: marcas del histórico de entrada."""
    sims = np.asarray(sims, float)
    conteos, bordes = np.histogram(sims, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    colores = ["rgba(245,158,11,.75)" if (limite is not None and c > limite) else "rgba(59,130,246,.65)"
               for c in centros]
    fig = go.Figure(go.Bar(x=centros, y=conteos / len(sims) * 100, width=(bordes[1] - bordes[0]),
                           marker=dict(color=colores, line=dict(width=0)),
                           hovertemplate="%{x:,.1f}<br>%{y:.2f}% de los escenarios generados<extra></extra>"))
    _vline(fig, p50, "P50", GREEN, "solid")
    _vline(fig, p5, "P5", MUTED, "dot", "top left")
    _vline(fig, p95, "P95", MUTED, "dot")
    if limite is not None:
        _vline(fig, limite, "Límite", AMBER, "dash", "bottom right")
    return _base(fig, 400, _con_aviso(titulo, _aviso_derivado(simulado, "Escenarios generados")), xt=xt,
                 yt="% de escenarios", leyenda=False)


def convergencia(idx, medias, simulado=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=idx, y=medias, mode="lines", line=dict(color=GREEN, width=2),
                               hovertemplate="%{x} escenarios<br>Media = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=float(medias[-1]), line_dash="dot", line_color=MUTED, line_width=1.2)
    return _base(fig, 300, _con_aviso("Convergencia de la media de los escenarios",
                                      _aviso_derivado(simulado, "Escenarios generados")),
                 xt="Número de escenarios generados", yt="Media acumulada (kWh)", leyenda=False)


# --------------------------------------------------------------------- factura
def escenarios(nombres, valores, presupuesto: float | None) -> go.Figure:
    fig = go.Figure(go.Bar(x=nombres, y=valores, marker=dict(color=[GREEN, BLUE, AMBER], line=dict(width=0)),
                           text=["$" + f"{v:,.0f}".replace(",", ".") for v in valores], textposition="outside",
                           textfont=dict(color=TEXT, size=TIPO["tick"]), width=0.5,
                           hovertemplate="%{x}<br><b>$%{y:,.0f}</b><extra></extra>"))
    if presupuesto:
        _hline(fig, presupuesto, "Presupuesto", RED)
    fig.update_yaxes(range=[0, max(max(valores), presupuesto or 0) * 1.2])
    return _base(fig, 360, "Escenarios de factura del próximo mes", yt="$", leyenda=False)


def facturas_historicas(df, estimada: float, etiqueta_est: str, ultimos: int = 12) -> go.Figure:
    d = df.tail(ultimos)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["etiqueta"], y=d["valor_factura"], name="Facturas registradas",
                         marker=dict(color="rgba(59,130,246,.65)", line=dict(width=0)),
                         hovertemplate="%{x}<br><b>$%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=[etiqueta_est], y=[estimada], name="Estimada (escenario medio)",
                         marker=dict(color=GREEN, line=dict(width=0)),
                         hovertemplate="%{x}<br><b>$%{y:,.0f}</b> (estimada)<extra></extra>"))
    _hline(fig, float(d["valor_factura"].mean()), "Promedio del periodo", MUTED, "dot")
    fig.update_xaxes(type="category")
    return _base(fig, 360, "Comparación con facturas anteriores", yt="$")
