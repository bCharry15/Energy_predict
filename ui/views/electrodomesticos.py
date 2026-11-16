"""C. Gestión de electrodomésticos (pantalla «Mis electrodomésticos»)."""
from __future__ import annotations

import sqlite3

import streamlit as st

from core import business, database
from core.config import money
from ui import charts, theme


def _campos(pref: str, nombre="", potencia=100.0, horas=1.0, dias=30, cantidad=1):
    nom = st.text_input("Nombre del aparato", value=nombre, max_chars=60, key=f"{pref}_nom")
    c1, c2 = st.columns(2)
    with c1:
        hor = st.number_input("¿Cuántas horas al día está encendido o conectado?", min_value=0.0, max_value=24.0,
                              value=float(horas), step=0.25, format="%.2f", key=f"{pref}_hor",
                              help="Ejemplos: media hora = 0,50 · 15 minutos = 0,25 · todo el día = 24")
        dias_ = st.number_input("¿Cuántos días al mes se usa?", min_value=0, max_value=31, value=int(dias), step=1,
                                key=f"{pref}_dias")
    with c2:
        cant = st.number_input("¿Cuántos aparatos iguales hay?", min_value=1, max_value=500, value=int(cantidad),
                               step=1, key=f"{pref}_cant")
        pot = st.number_input("Potencia en vatios (W)", min_value=0.0, max_value=20000.0, value=float(potencia),
                              step=10.0, format="%.0f", key=f"{pref}_pot",
                              help="Está en la etiqueta o placa del aparato. Si no la conoce, deje el valor sugerido.")
    return nom.strip(), pot, hor, dias_, cant


def render(df, equipos, s) -> None:
    theme.header("Mis electrodomésticos", "Aparatos de la casa, horas de uso y energía que gasta cada uno", "chip")
    tabla = business.tabla_electrodomesticos(equipos)
    t1, t2, t3 = st.tabs(["Agregar un aparato", "Mi lista de aparatos", "Cambiar o borrar"])

    with t1:
        nombres = [c[0] for c in business.CATALOGO] + [business.OTRO]
        tipo = st.selectbox("Paso 1 · Elija el aparato de la lista", nombres, key="eq_tipo",
                            help="Si no aparece, elija la última opción y escriba el nombre.")
        idx = nombres.index(tipo)
        if tipo == business.OTRO:
            base = ("", 100, 1.0, 30)
        else:
            base = business.CATALOGO[idx]
            st.caption(f"Valores habituales para «{tipo}»: {base[1]} W, {base[2]:g} horas al día. Puede cambiarlos.")
        with st.form(f"form_add_equipo_{idx}"):
            st.markdown("**Paso 2 · Indique cuánto se usa y pulse el botón**")
            datos = _campos(f"eqadd{idx}", base[0], base[1], base[2], base[3], 1)
            enviar = st.form_submit_button("Guardar aparato", type="primary")
        if enviar:
            try:
                business.validar_electrodomestico(*datos)
                database.insertar_electrodomestico(*datos)
                kwh = business.consumo_electrodomestico(*datos[1:])
                theme.flash("success", f"«{datos[0]}» quedó guardado. Gasta aproximadamente {kwh:.1f} kWh al mes, "
                                       f"unos {money(kwh * s['tarifa_kwh'])}.")
                st.rerun()
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))
        theme.note("<b>¿Cómo se calcula?</b> Energía del mes (kWh) = potencia (W) × horas al día × días al mes × "
                   "cantidad ÷ 1000. Ejemplo: un televisor de 100 W encendido 5 horas diarias durante 30 días gasta "
                   "100 × 5 × 30 ÷ 1000 = <b>15 kWh</b>.")

    with t2:
        if tabla.empty:
            st.info("Todavía no hay aparatos anotados. Agregue el primero en la pestaña «Agregar un aparato».")
        else:
            total = float(tabla["consumo_kwh"].sum())
            top = tabla.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                theme.card("Energía calculada", f"{total:.1f}", "kWh/mes", f"{len(tabla)} aparatos anotados", "bolt")
            with c2:
                theme.card("Costo aproximado", money(total * s["tarifa_kwh"]), "", f"A {money(s['tarifa_kwh'])} por kWh, sin subsidio", "receipt", "green")
            with c3:
                theme.card("El que más gasta", str(top["nombre"])[:22], "", f"{top['consumo_kwh']:.1f} kWh · {top['participacion_pct']:.1f}%", "alert", "amber")
            with c4:
                if len(df):
                    real = float(df["consumo_kwh"].iloc[-1])
                    theme.card("Comparado con el recibo", f"{total / real * 100:.0f}", "%",
                               f"El recibo de {df['etiqueta'].iloc[-1]} fue {real:.1f} kWh", "target", "violet")
                else:
                    theme.card("Comparado con el recibo", "—", "", "Aún no hay recibos anotados", "target", "muted")
            izq, der = st.columns([3, 2])
            with izq:
                theme.plot(charts.barras_equipos(tabla, 12), key="eq_bar", ancho=0.6)
            with der:
                theme.plot(charts.dona_equipos(tabla), key="eq_dona", ancho=0.4)
            vista = tabla[["nombre", "potencia_w", "horas_dia", "dias_mes", "cantidad", "consumo_kwh",
                           "participacion_pct", "es_simulado"]].copy()
            vista["costo"] = (vista["consumo_kwh"] * s["tarifa_kwh"]).round(0)
            vista["consumo_kwh"] = vista["consumo_kwh"].round(2)
            vista["participacion_pct"] = vista["participacion_pct"].round(1)
            vista["es_simulado"] = vista["es_simulado"].map({1: "Ejemplo", 0: "Mío"})
            vista = vista[["nombre", "potencia_w", "horas_dia", "dias_mes", "cantidad", "consumo_kwh",
                           "participacion_pct", "costo", "es_simulado"]]
            vista.columns = ["Aparato", "Potencia (W)", "Horas al día", "Días al mes", "Cantidad",
                             "Energía (kWh/mes)", "Parte del total (%)", "Costo ($/mes)", "Origen"]
            theme.table(vista)
            theme.note("La energía de cada aparato es un <b>cálculo aproximado</b> con la potencia y las horas "
                       "anotadas; no es una medición. En la nevera cuente solo las horas en que el motor trabaja "
                       "(unas 8 al día).")

    with t3:
        if equipos.empty:
            st.info("No hay aparatos para cambiar.")
        else:
            ids = equipos["id"].tolist()
            nombres_eq = dict(zip(equipos["id"], equipos["nombre"]))
            sel = st.selectbox("Elija el aparato", ids, format_func=lambda i: nombres_eq[i], key="edit_eq_sel")
            e = equipos[equipos["id"] == sel].iloc[0]
            with st.form(f"form_edit_equipo_{sel}"):
                st.markdown("**Cambiar los datos**")
                datos = _campos(f"eqedit{sel}", e["nombre"], e["potencia_w"], e["horas_dia"], e["dias_mes"], e["cantidad"])
                guardar = st.form_submit_button("Guardar cambios", type="primary")
            if guardar:
                try:
                    business.validar_electrodomestico(*datos)
                    database.actualizar_electrodomestico(sel, *datos)
                    theme.flash("success", f"«{datos[0]}» quedó actualizado.")
                    st.rerun()
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
            with st.form(f"form_del_equipo_{sel}"):
                st.markdown("**Borrar este aparato**")
                confirmar = st.checkbox(f"Sí, quiero borrar «{e['nombre']}»", key=f"del_eq_ok_{sel}")
                borrar = st.form_submit_button("Borrar aparato")
            if borrar:
                if confirmar:
                    database.eliminar_electrodomestico(sel)
                    theme.flash("success", f"«{e['nombre']}» se borró de la lista.")
                    st.rerun()
                else:
                    st.error("Marque la casilla «Sí, quiero borrar» y vuelva a pulsar el botón.")
