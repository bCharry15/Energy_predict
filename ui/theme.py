"""Identidad visual de ENERGY PREDICT: paleta, CSS, iconografía y componentes HTML."""
from __future__ import annotations

import html
import inspect

import numpy as np
import streamlit as st

# Paleta
BG = "#070B14"
SURFACE = "#0E1626"
SURFACE_2 = "#131D31"
BORDER = "#1F2B45"
TEXT = "#E6EDF7"
MUTED = "#8A9BB8"
BLUE = "#3B82F6"
BLUE_SOFT = "#60A5FA"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
VIOLET = "#A78BFA"
TONOS = {"blue": BLUE, "green": GREEN, "amber": AMBER, "red": RED, "violet": VIOLET, "muted": MUTED}

_ICON_PATHS = {
    "bolt": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "receipt": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "bars": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "medio": '<circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "chip": '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "shuffle": '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "wallet": '<rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/>',
    "down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
}


def icon(nombre: str, size: int = 18, color: str = "currentColor") -> str:
    cuerpo = _ICON_PATHS.get(nombre, _ICON_PATHS["info"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{cuerpo}</svg>')


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, .stApp, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(59,130,246,.10), transparent 60%), {BG}; color: {TEXT}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #060A12; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Navegación lateral: el grupo de radio se muestra como menú */
[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] div[role="radiogroup"] > label {{ padding: 9px 12px; border-radius: 10px; width: 100%;
  border: 1px solid transparent; transition: background .18s ease, border-color .18s ease; cursor: pointer; }}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: {SURFACE}; }}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{ background: rgba(59,130,246,.14);
  border-color: rgba(59,130,246,.35); }}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{ color: #fff; font-weight: 600; }}
[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: .92rem; color: {MUTED}; }}

.ep-brand {{ display:flex; align-items:center; gap:10px; margin: 2px 0 4px 2px; }}
.ep-brand-logo {{ width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, {BLUE}, #1D4ED8); box-shadow: 0 6px 18px rgba(59,130,246,.35); color:#fff; }}
.ep-brand-name {{ font-weight:700; letter-spacing:.06em; font-size:1.02rem; color:#fff; line-height:1.1; }}
.ep-brand-name span {{ color:{GREEN}; }}
.ep-brand-sub {{ font-size:.68rem; color:{MUTED}; letter-spacing:.04em; }}
.ep-navtitle {{ font-size:.68rem; letter-spacing:.14em; color:{MUTED}; text-transform:uppercase; margin: 18px 0 6px 4px; }}

.ep-header {{ display:flex; align-items:center; gap:14px; margin-bottom: 1.2rem; animation: epFade .45s ease both; }}
.ep-header-icon {{ width:46px; height:46px; border-radius:13px; background:{SURFACE_2}; border:1px solid {BORDER};
  display:flex; align-items:center; justify-content:center; color:{BLUE_SOFT}; flex-shrink:0; }}
.ep-header h1 {{ font-size:1.55rem; font-weight:700; margin:0; padding:0; color:#fff; letter-spacing:-.01em; }}
.ep-header p {{ margin:2px 0 0 0; color:{MUTED}; font-size:.92rem; }}

.ep-card {{ background: linear-gradient(180deg, {SURFACE_2}, {SURFACE}); border:1px solid {BORDER}; border-radius:16px;
  padding:16px 18px; height:100%; min-height:118px; margin-bottom: 14px; animation: epFade .5s ease both;
  transition: transform .18s ease, border-color .18s ease; }}
.ep-card:hover {{ transform: translateY(-2px); border-color:#2C3C60; }}
.ep-card-top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
.ep-card-label {{ color:{MUTED}; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600; }}
.ep-card-icon {{ width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; }}
.ep-card-value {{ font-size:1.65rem; font-weight:700; color:#fff; letter-spacing:-.02em; line-height:1.15; }}
.ep-card-value small {{ font-size:.85rem; color:{MUTED}; font-weight:500; margin-left:4px; }}
.ep-card-foot {{ margin-top:6px; font-size:.8rem; color:{MUTED}; }}
.ep-risk {{ display:inline-flex; align-items:center; gap:6px; margin-top:8px; padding:3px 10px 3px 8px; border-radius:999px;
  font-size:.82rem; font-weight:700; border:1px solid; }}
.ep-table-wrap {{ overflow-x:auto; border:1px solid {BORDER}; border-radius:12px; background:{SURFACE}; margin-bottom:.4rem; }}
.ep-table {{ width:100%; border-collapse:collapse; font-size:.86rem; line-height:1.45; color:{TEXT}; }}
.ep-table th {{ text-align:left; font-weight:600; color:#C3CFE4; background:{SURFACE_2}; padding:9px 12px;
  border-bottom:1px solid #2C3C60; vertical-align:bottom; }}
.ep-table td {{ padding:8px 12px; border-bottom:1px solid {BORDER}; vertical-align:top; }}
.ep-table tr:last-child td {{ border-bottom:none; }}
.ep-table .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.ep-table td.num {{ white-space:nowrap; }}
.ep-pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:.74rem; font-weight:600; margin-right:6px; }}

.ep-section {{ font-size:1.02rem; font-weight:600; color:#fff; margin: 1.1rem 0 .6rem 0; display:flex; align-items:center; gap:8px; }}
.ep-section::before {{ content:""; width:4px; height:16px; border-radius:2px; background:{BLUE}; display:inline-block; }}
.ep-note {{ border:1px solid {BORDER}; border-left:3px solid {BLUE}; background:{SURFACE}; border-radius:12px;
  padding:12px 14px; color:#C9D5EA; font-size:.88rem; margin: .4rem 0 1rem 0; line-height:1.5; }}
.ep-note b {{ color:#fff; }}
.ep-eq {{ font-family:'JetBrains Mono', ui-monospace, monospace; background:{SURFACE}; border:1px solid {BORDER};
  border-radius:12px; padding:14px 16px; color:{BLUE_SOFT}; font-size:.95rem; overflow-x:auto; margin-bottom: 1rem; }}
.ep-badge {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-size:.74rem;
  font-weight:600; border:1px solid; }}
.ep-rec {{ border:1px solid {BORDER}; border-left-width:3px; background:{SURFACE}; border-radius:12px; padding:13px 16px;
  margin-bottom:10px; animation: epFade .5s ease both; }}
.ep-rec-title {{ font-weight:600; color:#fff; font-size:.96rem; display:flex; align-items:center; gap:8px; }}
.ep-rec-body {{ color:#B9C6DD; font-size:.87rem; margin-top:5px; line-height:1.5; }}
.ep-rec-meta {{ margin-top:8px; font-size:.76rem; color:{MUTED}; }}

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius:10px;
  border:1px solid {BORDER}; background:{SURFACE_2}; color:{TEXT}; font-weight:600; transition: all .18s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  border-color:{BLUE}; color:#fff; }}
[data-testid="stFormSubmitButton"] > button[kind="primary"], .stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {BLUE}, #2563EB); border-color: transparent; color:#fff; }}
[data-testid="stForm"] {{ border:1px solid {BORDER}; border-radius:16px; background:{SURFACE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom:1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding: 8px 14px; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:#fff; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background:{SURFACE}; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
[data-testid="stPlotlyChart"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:6px 6px 0 6px;
  animation: epFade .55s ease both; }}
@keyframes epFade {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
@media (max-width: 900px) {{ .ep-card-value {{ font-size:1.35rem; }} .ep-header h1 {{ font-size:1.25rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }} }}
</style>
"""


CSS_GRANDE = """
<style>
html {{ font-size: {base}px; }}
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown {{ font-size: 1.02rem; line-height: 1.6; }}
[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: 1.02rem; }}
[data-testid="stSidebar"] div[role="radiogroup"] > label {{ padding: 11px 12px; }}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  font-size: 1.05rem; padding: .7rem 1.3rem; min-height: 3rem; }}
.stApp input, .stApp [data-baseweb="select"] {{ font-size: 1.05rem; }}
[data-testid="stWidgetLabel"] p {{ font-size: 1.02rem; font-weight: 600; color: #fff; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.05rem; font-weight: 600; }}
.ep-table {{ font-size: 1rem; }}
.ep-card-label {{ font-size: .82rem; }} .ep-card-foot {{ font-size: .9rem; }} .ep-note {{ font-size: .98rem; }}
.ep-rec-title {{ font-size: 1.08rem; }} .ep-rec-body {{ font-size: .98rem; }}
.ep-help {{ background: rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.35); border-radius:14px;
  padding:14px 18px; margin: 0 0 1.2rem 0; color:#DDF7E7; font-size:1.08rem; line-height:1.55; }}
.ep-help b {{ color:#fff; }}
.ep-hero {{ border-radius:20px; padding:26px 28px; margin-bottom:1.2rem; border:1px solid {borde};
  background: linear-gradient(135deg, #0F1D3A, #0B1324); animation: epFade .5s ease both; }}
.ep-hero h2 {{ font-size:1.9rem; color:#fff; margin:0 0 .4rem 0; padding:0; font-weight:700; }}
.ep-hero p {{ font-size:1.2rem; color:#D5DFF0; margin:0; line-height:1.55; }}
.ep-hero .ep-big {{ font-size:2.6rem; font-weight:700; color:#fff; letter-spacing:-.02em; }}
.ep-hero .ep-big small {{ font-size:1.2rem; color:#A9B8D3; font-weight:500; }}
.ep-light {{ display:inline-block; width:18px; height:18px; border-radius:50%; margin-right:10px; vertical-align:-2px; }}
.ep-step {{ background: linear-gradient(180deg, #131D31, #0E1626); border:1px solid {borde}; border-radius:16px;
  padding:18px 20px; min-height:190px; margin-bottom:10px; }}
.ep-step-num {{ width:40px; height:40px; border-radius:50%; background:#3B82F6; color:#fff; font-weight:700;
  font-size:1.25rem; display:flex; align-items:center; justify-content:center; margin-bottom:10px; }}
.ep-step h3 {{ font-size:1.25rem; color:#fff; margin:0 0 6px 0; padding:0; }}
.ep-step p {{ font-size:1.05rem; color:#C3CFE4; margin:0; }}
</style>
"""

# Explicación en lenguaje sencillo que se muestra bajo el título de cada pantalla.
AYUDA = {
    "Resumen general": "Aquí está todo en un vistazo: cuánto consume la casa, cuánto se espera para el próximo mes y cuánto costaría.",
    "Mi consumo mensual": "Aquí se anota, mes a mes, lo que dice el recibo de la luz: los kilovatios-hora (kWh) consumidos y el valor pagado.",
    "Mis electrodomésticos": "Aquí se anotan los aparatos de la casa y cuántas horas al día están encendidos. El programa calcula cuánta energía gasta cada uno.",
    "Análisis estadístico": "Resume el historial con números: el consumo promedio, el más alto, el más bajo y qué tanto cambia de un mes a otro.",
    "Regresión lineal": "Busca una fórmula que relaciona el consumo con las horas de uso y la cantidad de aparatos, y la usa para predecir el próximo mes.",
    "Análisis de probabilidad": "Responde: ¿qué tan probable es gastar más de cierto límite? 0 % significa imposible y 100 % significa seguro.",
    "Simulación de Monte Carlo": "El computador imagina miles de veces cómo podría ser el próximo mes y cuenta en cuántas se supera el límite.",
    "Mi factura": "Calcula cuánto se pagaría el próximo mes según la tarifa y el estrato, en tres casos: consumo bajo, normal y alto.",
    "Consejos de ahorro": "Consejos calculados con los datos de esta casa: qué aparatos gastan más y cuánto se ahorraría usándolos menos.",
    "Configuración": "Aquí se escriben la tarifa del recibo, el estrato de la vivienda y los límites que no se quieren superar.",
}


def inject_css(letra_grande: bool = False) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(CSS_GRANDE.format(base=20 if letra_grande else 17, borde=BORDER), unsafe_allow_html=True)


def help_box(texto_html: str) -> None:
    """Explicación en lenguaje sencillo (admite <b>)."""
    st.markdown(f'<div class="ep-help">{texto_html}</div>', unsafe_allow_html=True)


def hero(html_interno: str) -> None:
    st.markdown(f'<div class="ep-hero">{html_interno}</div>', unsafe_allow_html=True)


def step(numero: int, titulo: str, texto: str) -> None:
    st.markdown(f'<div class="ep-step"><div class="ep-step-num">{numero}</div><h3>{html.escape(titulo)}</h3>'
                f'<p>{html.escape(texto)}</p></div>', unsafe_allow_html=True)


def ir_a(pagina: str) -> None:
    """Callback de los botones de navegación: cambia el módulo seleccionado en el menú."""
    st.session_state["nav"] = pagina


def brand() -> None:
    st.markdown(
        f'<div class="ep-brand"><div class="ep-brand-logo">{icon("bolt", 20)}</div><div>'
        '<div class="ep-brand-name">ENERGY <span>PREDICT</span></div>'
        '<div class="ep-brand-sub">Predicción de consumo eléctrico</div></div></div>'
        '<div class="ep-navtitle">Menú</div>', unsafe_allow_html=True)


def header(titulo: str, subtitulo: str, icono: str = "grid") -> None:
    st.markdown(
        f'<div class="ep-header"><div class="ep-header-icon">{icon(icono, 22)}</div><div>'
        f'<h1>{html.escape(titulo)}</h1><p>{html.escape(subtitulo)}</p></div></div>', unsafe_allow_html=True)
    if titulo in AYUDA:
        help_box("<b>En palabras sencillas:</b> " + html.escape(AYUDA[titulo]))


def origen(simulado, derivado: bool = False, unidad: str = "meses") -> str:
    """Texto breve sobre el origen de un resultado; cadena vacía si no hay datos de ejemplo.

    `simulado`: marcas es_simulado de los datos usados. `derivado`: el resultado se calcula a partir
    de ellos (modelo, probabilidad, escenarios) en lugar de ser los datos mismos.
    """
    marcas = [bool(v) for v in simulado]
    k, n = sum(marcas), len(marcas)
    if k == 0:
        return ""
    if k == n:
        return "A partir de datos simulados de ejemplo" if derivado else "Datos simulados de ejemplo"
    if derivado:
        return f"A partir de {k} {unidad} simulados y {n - k} ingresados por usted"
    return f"{k} {unidad} simulados · {n - k} ingresados por usted"


def _origen_html(texto: str) -> str:
    return (f'<div style="display:flex;align-items:center;gap:5px;margin-top:6px;font-size:.74rem;'
            f'font-weight:600;color:{AMBER};">{icon("info", 12)}<span>{html.escape(texto)}</span></div>')


def marca_origen(texto: str) -> None:
    """Aviso de origen junto a un resultado que no es tarjeta ni gráfica (tablas, ecuaciones, listas)."""
    if texto:
        st.markdown(_origen_html(texto), unsafe_allow_html=True)


RIESGO = {"bajo": ("green", "check", "Riesgo bajo"), "medio": ("amber", "medio", "Riesgo medio"),
          "alto": ("red", "alert", "Riesgo alto")}


def nivel_riesgo(p: float, medio: float = 0.2, alto: float = 0.5) -> str:
    """Clasificación del semáforo de riesgo a partir de una probabilidad (umbrales de la app)."""
    return "alto" if p >= alto else "medio" if p >= medio else "bajo"


def tono_riesgo(nivel: str) -> str:
    return RIESGO[nivel][0]


def _riesgo_html(nivel: str) -> str:
    tono, ic, texto = RIESGO[nivel]
    c = TONOS[tono]
    return (f'<div><span class="ep-risk" style="color:{c};border-color:{c}66;background:{c}1A;">'
            f'{icon(ic, 14)}<span>{texto}</span></span></div>')


def card(label: str, valor: str, unidad: str = "", pie: str = "", icono: str = "bolt", tono: str = "blue",
         pill: str | None = None, pill_tono: str = "green", origen: str = "", riesgo: str = "") -> None:
    """Tarjeta de resultado. `riesgo` ("bajo", "medio" o "alto") añade la etiqueta del semáforo en texto
    y fija el color de la tarjeta según ese nivel, para que el riesgo no dependa solo del color."""
    if riesgo:
        tono = tono_riesgo(riesgo)
    color = TONOS.get(tono, BLUE)
    pill_html = ""
    if pill:
        pc = TONOS.get(pill_tono, GREEN)
        pill_html = f'<span class="ep-pill" style="color:{pc};background:{pc}22;">{html.escape(pill)}</span>'
    unidad_html = f"<small>{html.escape(unidad)}</small>" if unidad else ""
    origen_html = _origen_html(origen) if origen and valor != "—" else ""
    st.markdown(
        f'<div class="ep-card"><div class="ep-card-top"><span class="ep-card-label">{html.escape(label)}</span>'
        f'<span class="ep-card-icon" style="color:{color};background:{color}1F;">{icon(icono, 16)}</span></div>'
        f'<div class="ep-card-value">{html.escape(valor)}{unidad_html}</div>'
        f'{_riesgo_html(riesgo) if riesgo and valor != "—" else ""}'
        f'<div class="ep-card-foot">{pill_html}{html.escape(pie)}</div>{origen_html}</div>', unsafe_allow_html=True)


def section(titulo: str) -> None:
    st.markdown(f'<div class="ep-section">{html.escape(titulo)}</div>', unsafe_allow_html=True)


def note(texto_html: str, tono: str = "blue") -> None:
    """Nota explicativa. `texto_html` admite etiquetas <b> y debe venir ya escapado si incluye datos."""
    color = TONOS.get(tono, BLUE)
    st.markdown(f'<div class="ep-note" style="border-left-color:{color};">{texto_html}</div>', unsafe_allow_html=True)


def equation(texto: str) -> None:
    st.markdown(f'<div class="ep-eq">{html.escape(texto)}</div>', unsafe_allow_html=True)


def badge(texto: str, tono: str = "amber", icono: str = "info") -> str:
    color = TONOS.get(tono, AMBER)
    return (f'<span class="ep-badge" style="color:{color};border-color:{color}55;background:{color}14;">'
            f'{icon(icono, 13)}{html.escape(texto)}</span>')


def recommendation(rec: dict) -> None:
    conf = {"alta": (RED, "alert", "Prioridad alta"), "media": (AMBER, "info", "Prioridad media"),
            "baja": (BLUE, "info", "Informativa"), "positiva": (GREEN, "check", "Positiva")}
    color, ic, etiqueta = conf.get(rec["nivel"], conf["baja"])
    meta = f'{etiqueta} · {html.escape(rec["categoria"])}'
    if rec.get("ahorro_kwh"):
        meta += f' · Ahorro potencial: {rec["ahorro_kwh"]:.1f} kWh/mes'
    st.markdown(
        f'<div class="ep-rec" style="border-left-color:{color};"><div class="ep-rec-title">'
        f'<span style="color:{color};display:flex;">{icon(ic, 16)}</span>{html.escape(rec["titulo"])}</div>'
        f'<div class="ep-rec-body">{html.escape(rec["detalle"])}</div>'
        f'<div class="ep-rec-meta">{meta}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ utilidades
def _version_streamlit() -> tuple[int, int]:
    try:
        partes = st.__version__.split(".")
        return int(partes[0]), int(partes[1])
    except Exception:
        return (0, 0)


def _ancho_nuevo(func) -> bool:
    """Las versiones recientes de Streamlit usan width="stretch" en lugar de use_container_width."""
    try:
        return _version_streamlit() >= (1, 50) and "width" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


# Ancho útil del área principal (px) medido en una pantalla de 1280 px con el menú abierto.
# Se usa para partir títulos y avisos de las gráficas y que se lean completos en columnas angostas.
ANCHO_PRINCIPAL = 730


def plot(fig, key: str | None = None, ancho: float = 1.0) -> None:
    """Muestra una figura Plotly a todo el ancho conservando el tema propio (theme=None).

    `ancho`: fracción del área principal que ocupa la columna (0,5 en dos columnas iguales, 0,4 en la
    menor de [3, 2]...). Con «Letra más grande» activada se agrandan ejes, leyendas y anotaciones.
    """
    from ui import charts  # import local: charts depende de theme

    charts.ajustar(fig, ANCHO_PRINCIPAL * ancho, bool(st.session_state.get("letra_grande", False)))
    cfg = {"displayModeBar": False, "responsive": True}
    if _ancho_nuevo(st.plotly_chart):
        st.plotly_chart(fig, width="stretch", theme=None, config=cfg, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg, key=key)


def table(df, height: int | None = None) -> None:
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    if _ancho_nuevo(st.dataframe):
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def _celda(v) -> str:
    if isinstance(v, float):
        return "—" if v != v else np.format_float_positional(v, trim="-")
    return html.escape(str(v))


def tabla(df) -> None:
    """Tabla de resultados en HTML: su letra crece con «Letra más grande» y el texto largo se ajusta
    en varias líneas en lugar de cortarse (a diferencia de st.dataframe, que se dibuja en canvas)."""
    # Columnas numéricas o de valores numéricos ya formateados («< 0.001», «—»): alineadas a la derecha y sin partir.
    numericas = {c for c in df.columns if df[c].dtype.kind in "iuf"
                 or df[c].astype(str).str.fullmatch(r"[<>≤≥]?\s?[−\-+$]?[\d.,]+%?|—").all()}
    cab = "".join(f'<th class="{"num" if c in numericas else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    filas = "".join("<tr>" + "".join(f'<td class="{"num" if c in numericas else ""}">{_celda(v)}</td>'
                                     for c, v in zip(df.columns, fila)) + "</tr>"
                    for fila in df.itertuples(index=False))
    st.markdown(f'<div class="ep-table-wrap"><table class="ep-table"><thead><tr>{cab}</tr></thead>'
                f'<tbody>{filas}</tbody></table></div>', unsafe_allow_html=True)


def flash(tipo: str, mensaje: str) -> None:
    """Guarda un mensaje para mostrarlo después de st.rerun()."""
    st.session_state["_flash"] = (tipo, mensaje)


def show_flash() -> None:
    dato = st.session_state.pop("_flash", None)
    if dato:
        tipo, mensaje = dato
        {"success": st.success, "error": st.error, "warning": st.warning}.get(tipo, st.info)(mensaje)


def requiere(df, minimo: int, que: str) -> bool:
    """Devuelve True si hay datos suficientes; si no, muestra un aviso claro."""
    if len(df) >= minimo:
        return True
    st.warning(f"Se necesitan al menos {minimo} registros de consumo para {que}. "
               f"Actualmente hay {len(df)}. Agregue registros en «Gestión del consumo» o restaure los "
               "datos de demostración en «Configuración».")
    return False


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"
