"""B. Gestión del consumo (CRUD + importación/exportación CSV)."""
from __future__ import annotations

import sqlite3

import streamlit as st

from core import business, config, data_processing, database
from core.config import money
from ui import theme


def _periodo(anio: int, mes_nombre: str) -> str:
    return f"{int(anio):04d}-{config.MESES.index(mes_nombre) + 1:02d}"


def _campos(pref: str, anio: int, mes: int, consumo: float, factura: float, equipos: int, horas: float):
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("Año", min_value=2000, max_value=2100, value=int(anio), step=1, key=f"{pref}_anio")
    with c2:
        m = st.selectbox("Mes", config.MESES, index=int(mes) - 1, key=f"{pref}_mes")
    c1, c2 = st.columns(2)
    with c1:
        kwh = st.number_input("Consumo del mes en kWh (aparece en el recibo)", min_value=0.0, max_value=20000.0, value=float(consumo), step=1.0,
                              format="%.1f", key=f"{pref}_kwh")
        n = st.number_input("Cantidad de aparatos en la casa", min_value=0, max_value=500, value=int(equipos), step=1,
                            key=f"{pref}_n")
    with c2:
        fac = st.number_input("Valor pagado en el recibo ($)", min_value=0.0, value=float(factura), step=1000.0,
                              format="%.0f", key=f"{pref}_fac")
        h = st.number_input("Horas promedio de uso de los aparatos al día", min_value=0.0, max_value=24.0,
                            value=float(horas), step=0.1, format="%.1f", key=f"{pref}_h")
    return _periodo(a, m), kwh, fac, n, h


def render(df, equipos, s) -> None:
    theme.header("Mi consumo mensual", "Anote, cambie o borre el consumo de cada mes", "calendar")
    t1, t2, t3, t4 = st.tabs(["Agregar un mes", "Ver los meses anotados", "Cambiar o borrar", "Importar / exportar"])

    with t1:
        sig = config.next_periodo(df["periodo"].iloc[-1]) if len(df) else "2026-01"
        y, m = (int(v) for v in sig.split("-"))
        ult = df.iloc[-1] if len(df) else None
        sug = business.sugerencia_registro(equipos)
        if sug:
            fac_sug = round(business.factura_s(sug["consumo_kwh"], s) / 100) * 100
            theme.help_box(f"Los valores ya vienen <b>llenos con lo calculado a partir de sus aparatos</b> "
                           f"({sug['consumo_kwh']:.1f} kWh). Si tiene el recibo de la luz a la mano, "
                           "<b>cámbielos por los del recibo</b>, que son más exactos, y pulse «Guardar registro».")
        else:
            fac_sug = 0.0
            sug = {"consumo_kwh": 0.0, "num_electrodomesticos": int(ult["num_electrodomesticos"]) if ult is not None else 10,
                   "horas_uso": float(ult["horas_uso"]) if ult is not None else 5.0}
        with st.form("form_add_consumo", clear_on_submit=False):
            st.markdown("**Mes nuevo**")
            periodo, kwh, fac, n, h = _campos("add", y, m, sug["consumo_kwh"], fac_sug,
                                              sug["num_electrodomesticos"], sug["horas_uso"])
            enviar = st.form_submit_button("Guardar registro", type="primary")
        if enviar:
            try:
                business.validar_consumo(periodo, kwh, fac, n, h)
                if database.existe_periodo(periodo):
                    raise ValueError(f"Ya existe un registro para {config.periodo_label(periodo)}. Edítelo en la pestaña «Editar / eliminar».")
                database.insertar_consumo(periodo, kwh, fac, n, h)
                theme.flash("success", f"Registro de {config.periodo_label(periodo)} guardado ({kwh:.1f} kWh).")
                st.rerun()
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))

    with t2:
        if df.empty:
            st.info("Aún no hay meses anotados. Use la pestaña «Agregar un mes».")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                theme.card("Registros", str(len(df)), "meses", f"{df['etiqueta'].iloc[0]} – {df['etiqueta'].iloc[-1]}", "calendar")
            with c2:
                theme.card("Consumo total", f"{df['consumo_kwh'].sum():,.0f}".replace(",", "."), "kWh", "Suma del histórico", "bolt", "violet")
            with c3:
                theme.card("Facturación total", money(df["valor_factura"].sum()), "", "Suma del histórico", "receipt", "green")
            with c4:
                sim = int(df["es_simulado"].sum())
                theme.card("Datos simulados", str(sim), f"de {len(df)}", "Registros de demostración", "info",
                           "amber" if sim else "green")
            vista = df.sort_values("periodo", ascending=False)[
                ["etiqueta", "consumo_kwh", "valor_factura", "num_electrodomesticos", "horas_uso",
                 "tarifa_implicita", "variacion_pct", "es_simulado"]].copy()
            vista["tarifa_implicita"] = vista["tarifa_implicita"].round(1)
            vista["variacion_pct"] = vista["variacion_pct"].round(1)
            vista["es_simulado"] = vista["es_simulado"].map({1: "Ejemplo", 0: "Mío"})
            vista.columns = ["Mes", "Consumo (kWh)", "Factura ($)", "Electrodomésticos", "Horas/día",
                             "Precio pagado por kWh ($)", "Variación (%)", "Origen"]
            theme.table(vista, height=460)

    with t3:
        if df.empty:
            st.info("No hay registros para editar.")
        else:
            opciones = df.sort_values("periodo", ascending=False)
            ids = opciones["id"].tolist()
            etiquetas = dict(zip(opciones["id"], opciones["etiqueta"] + " · " + opciones["consumo_kwh"].round(1).astype(str) + " kWh"))
            sel = st.selectbox("Registro", ids, format_func=lambda i: etiquetas[i], key="edit_consumo_sel")
            fila = df[df["id"] == sel].iloc[0]
            y, m = (int(v) for v in fila["periodo"].split("-"))
            with st.form(f"form_edit_consumo_{sel}"):
                st.markdown("**Editar registro**")
                periodo, kwh, fac, n, h = _campos(f"edit{sel}", y, m, fila["consumo_kwh"], fila["valor_factura"],
                                                  fila["num_electrodomesticos"], fila["horas_uso"])
                guardar = st.form_submit_button("Guardar cambios", type="primary")
            if guardar:
                try:
                    business.validar_consumo(periodo, kwh, fac, n, h)
                    if database.existe_periodo(periodo, excluir_id=sel):
                        raise ValueError(f"Ya existe otro registro para {config.periodo_label(periodo)}.")
                    database.actualizar_consumo(sel, periodo, kwh, fac, n, h)
                    theme.flash("success", f"Registro de {config.periodo_label(periodo)} actualizado.")
                    st.rerun()
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
            with st.form(f"form_del_consumo_{sel}"):
                st.markdown("**Eliminar registro**")
                confirmar = st.checkbox(f"Confirmo que deseo eliminar el registro de {fila['etiqueta']}",
                                        key=f"del_consumo_ok_{sel}")
                borrar = st.form_submit_button("Eliminar registro")
            if borrar:
                if confirmar:
                    database.eliminar_consumo(sel)
                    theme.flash("success", f"Registro de {fila['etiqueta']} eliminado.")
                    st.rerun()
                else:
                    st.error("Marque la casilla de confirmación para eliminar el registro.")

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            theme.section("Exportar")
            st.download_button("Descargar histórico (CSV)", data_processing.exportar_csv(df) if len(df) else b"",
                               file_name="energy_predict_consumo.csv", mime="text/csv", disabled=df.empty)
            st.download_button("Descargar plantilla CSV", data_processing.plantilla_csv(),
                               file_name="plantilla_consumo.csv", mime="text/csv")
            st.caption("Columnas: periodo (AAAA-MM), consumo_kwh, valor_factura, num_electrodomesticos, horas_uso.")
        with c2:
            theme.section("Importar datos propios")
            archivo = st.file_uploader("Archivo CSV", type=["csv"], key="csv_up")
            reemplazar = st.checkbox("Reemplazar todo el histórico actual (incluye los datos simulados)", key="csv_rep")
            if archivo is not None:
                registros, errores = data_processing.leer_csv(archivo.getvalue())
                if errores:
                    st.error("Se encontraron problemas:\n\n" + "\n".join(f"- {e}" for e in errores[:10]))
                if registros:
                    st.success(f"{len(registros)} filas válidas listas para importar.")
                    if st.button("Importar filas válidas", type="primary", key="csv_go"):
                        if reemplazar:
                            with database.connect() as c:
                                c.execute("DELETE FROM consumo")
                        database.upsert_consumo(registros)
                        theme.flash("success", f"Se importaron {len(registros)} registros.")
                        st.rerun()
