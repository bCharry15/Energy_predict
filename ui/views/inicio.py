"""Pantalla de inicio en lenguaje sencillo: resultado principal y guía de 3 pasos."""
from __future__ import annotations

import html

import streamlit as st

from core import analysis, business, config
from core.config import money
from ui import theme


def render(df, equipos, s) -> None:
    theme.hero("<h2>Bienvenido a ENERGY PREDICT</h2>"
               "<p>Este programa le ayuda a saber <b>cuánta energía eléctrica gastará su casa el próximo mes</b>, "
               f"<b>cuánto pagará</b> y <b>cómo ahorrar</b>. Está preparado para viviendas de {config.CIUDAD}.</p>")

    if len(df) >= 3:
        r = analysis.resumen_general(df, s)
        pron = r["pronostico"]
        mes = config.periodo_label(pron["periodo"])
        p = r["p_limite"]
        if p < 0.2:
            color, frase = theme.GREEN, "Es poco probable que supere su límite"
        elif p < 0.5:
            color, frase = theme.AMBER, "Hay cierto riesgo de superar su límite"
        else:
            color, frase = theme.RED, "Es probable que supere su límite"
        theme.hero(
            f"<p>Para <b>{html.escape(mes)}</b> se espera que la casa consuma cerca de</p>"
            f"<div class='ep-big'>{pron['yhat']:.0f} <small>kWh (kilovatios-hora)</small></div>"
            f"<p>y que el recibo de la luz sea de aproximadamente</p>"
            f"<div class='ep-big'>{money(r['factura_estimada'])}</div>"
            f"<p style='margin-top:14px'><span class='ep-light' style='background:{color}'></span>"
            f"<b>{frase}</b> de {s['limite_kwh']:.0f} kWh: la probabilidad es de {p * 100:.0f} de cada 100.</p>"
            f"<p style='margin-top:8px;font-size:1rem;color:#A9B8D3'>En 90 de cada 100 casos el consumo estará entre "
            f"{pron['lo']:.0f} y {pron['hi']:.0f} kWh. El mes pasado ({html.escape(df['etiqueta'].iloc[-1])}) fue de "
            f"{r['actual']:.0f} kWh y el promedio de la casa es {r['promedio']:.0f} kWh.</p>")
    else:
        theme.help_box("Todavía no hay suficientes meses anotados para hacer una predicción. "
                       "<b>Se necesitan al menos 3 meses.</b> Siga los pasos de abajo para empezar.")

    theme.section("¿Cómo se usa? Solo tres pasos")
    c1, c2, c3 = st.columns(3)
    with c1:
        theme.step(1, "Anote sus aparatos", "Escriba qué electrodomésticos hay en la casa y cuántas horas al día "
                   "están encendidos. El programa calcula cuánto gasta cada uno.")
        st.button("Ir a Mis electrodomésticos", key="ini_b1", type="primary", on_click=theme.ir_a,
                  args=("1. Mis electrodomésticos",))
    with c2:
        theme.step(2, "Anote el consumo del mes", "Copie del recibo de la luz los kilovatios-hora (kWh) y el valor "
                   "pagado. Entre más meses anote, mejor será la predicción.")
        st.button("Ir a Mi consumo mensual", key="ini_b2", type="primary", on_click=theme.ir_a,
                  args=("2. Mi consumo mensual",))
    with c3:
        theme.step(3, "Mire el resultado", "Vea cuánto pagaría el próximo mes y los consejos para ahorrar. "
                   "Todo se calcula solo, sin hacer cuentas.")
        st.button("Ir a Mi factura", key="ini_b3", type="primary", on_click=theme.ir_a, args=("3. Mi factura",))

    tabla = business.tabla_electrodomesticos(equipos)
    if not tabla.empty:
        top = tabla.iloc[0]
        theme.help_box(f"<b>Dato útil:</b> el aparato que más energía gasta en esta casa es "
                       f"<b>{html.escape(str(top['nombre']))}</b>, con {top['consumo_kwh']:.0f} kWh al mes "
                       f"({top['participacion_pct']:.0f} de cada 100 kWh del consumo calculado).")

    theme.section(f"Datos de {config.CIUDAD} que usa el programa")
    estrato = int(s.get("estrato", 4))
    sub = config.SUBSIDIO[estrato]
    con = config.CONTRIBUCION[estrato]
    regla = (f"recibe un descuento (subsidio) de hasta {sub * 100:.0f} % en los primeros "
             f"{config.SUBSISTENCIA_KWH:.0f} kWh de cada mes" if sub > 0 else
             f"paga un recargo (contribución) del {con * 100:.0f} % sobre todo el consumo" if con > 0 else
             "paga la tarifa completa, sin descuento ni recargo")
    theme.note(f"• Empresa de energía: <b>{config.OPERADOR}</b>.<br>"
               f"• Precio de cada kWh configurado: <b>{money(s['tarifa_kwh'])}</b>. Cambia cada mes; puede "
               "actualizarlo en «Configuración» con el valor de su recibo.<br>"
               f"• Vivienda de <b>estrato {estrato}</b>: {regla}.<br>"
               f"• Popayán está a más de 1.000 metros de altura, por eso el consumo básico subsidiado es de "
               f"{config.SUBSISTENCIA_KWH:.0f} kWh al mes.")
    if len(df) and int(df["es_simulado"].sum()):
        theme.note("<b>Importante:</b> los meses que aparecen cargados son un <b>ejemplo simulado</b> de una casa de "
                   "Popayán, para que el programa se pueda probar de inmediato. No son mediciones reales. Puede "
                   "borrarlos en «Configuración» y anotar los suyos.", "amber")
