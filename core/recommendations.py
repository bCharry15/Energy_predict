"""Recomendaciones basadas en reglas calculadas sobre los datos (no son mensajes aleatorios).

Cada recomendación indica la evidencia numérica que la activó.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import business, descriptive
from .config import money

ORDEN = {"alta": 0, "media": 1, "baja": 2, "positiva": 3}


def _rec(nivel, titulo, detalle, ahorro_kwh=None, ahorro_dinero=None, categoria="Consumo"):
    return {"nivel": nivel, "titulo": titulo, "detalle": detalle, "ahorro_kwh": ahorro_kwh,
            "ahorro_dinero": ahorro_dinero, "categoria": categoria}


def generar(df: pd.DataFrame, equipos: pd.DataFrame, settings: dict, pron: dict | None = None,
            p_limite: float | None = None) -> list[dict]:
    recs: list[dict] = []
    tarifa = settings["tarifa_kwh"]
    limite = settings["limite_kwh"]

    # ---------------------------------------------------- histórico de consumo
    if len(df) >= 2:
        x = df["consumo_kwh"].to_numpy(float)
        ultimo, etiqueta = x[-1], df["etiqueta"].iloc[-1]
        previos = x[:-1]
        if len(previos) >= 5:
            media, sd = previos.mean(), previos.std(ddof=1)
            z = (ultimo - media) / sd if sd > 0 else 0.0
            dif_pct = (ultimo - media) / media * 100
            if z > 2:
                recs.append(_rec("alta", f"Incremento inusual en {etiqueta}",
                                 f"El consumo fue de {ultimo:.1f} kWh, {dif_pct:.1f}% por encima del promedio "
                                 f"histórico ({media:.1f} kWh) y a {z:.1f} desviaciones estándar. Revise equipos "
                                 "nuevos, fugas eléctricas o cambios de hábitos ese mes.", categoria="Anomalía"))
            elif dif_pct > 5:
                recs.append(_rec("media", "Consumo actual por encima del histórico",
                                 f"{etiqueta}: {ultimo:.1f} kWh frente a un promedio previo de {media:.1f} kWh "
                                 f"(+{dif_pct:.1f}%, z = {z:.2f}). Aún dentro de la variación habitual, pero "
                                 "conviene vigilar el próximo mes.", categoria="Comparación histórica"))
            elif dif_pct < -5:
                recs.append(_rec("positiva", "Consumo actual por debajo del histórico",
                                 f"{etiqueta}: {ultimo:.1f} kWh, {abs(dif_pct):.1f}% menos que el promedio previo "
                                 f"({media:.1f} kWh). Ahorro aproximado de {money((media - ultimo) * tarifa)} "
                                 "frente a un mes típico.", categoria="Comparación histórica"))
            else:
                recs.append(_rec("baja", "Consumo actual en línea con el histórico",
                                 f"{etiqueta}: {ultimo:.1f} kWh frente a un promedio previo de {media:.1f} kWh "
                                 f"({dif_pct:+.1f}%).", categoria="Comparación histórica"))
        var = (x[-1] - x[-2]) / x[-2] * 100
        if var > 10:
            recs.append(_rec("media", "Aumento marcado respecto al mes anterior",
                             f"El consumo subió {var:.1f}% ({x[-2]:.1f} → {x[-1]:.1f} kWh). El costo adicional "
                             f"estimado es {money((x[-1] - x[-2]) * tarifa)}.", categoria="Anomalía"))

        # Tendencia de los últimos 12 meses (pendiente de consumo vs. tiempo).
        ult = x[-12:]
        if len(ult) >= 8:
            lr = stats.linregress(np.arange(len(ult)), ult)
            if lr.pvalue < 0.05 and lr.slope > 0:
                recs.append(_rec("media", "Tendencia creciente del consumo",
                                 f"En los últimos {len(ult)} meses el consumo aumenta en promedio "
                                 f"{lr.slope:.2f} kWh por mes (p = {lr.pvalue:.3f}). De mantenerse, en 6 meses "
                                 f"se pagarían unos {money(lr.slope * 6 * tarifa)} más al mes.",
                                 categoria="Tendencia"))
            elif lr.pvalue < 0.05 and lr.slope < 0:
                recs.append(_rec("positiva", "Tendencia decreciente del consumo",
                                 f"En los últimos {len(ult)} meses el consumo baja en promedio "
                                 f"{abs(lr.slope):.2f} kWh por mes (p = {lr.pvalue:.3f}).", categoria="Tendencia"))

        # Sensibilidad a las horas de uso (regresión simple consumo ~ horas).
        if len(df) >= 8 and df["horas_uso"].nunique() > 2:
            lr = stats.linregress(df["horas_uso"].to_numpy(float), x)
            if lr.pvalue < 0.05 and lr.slope > 0:
                recs.append(_rec("media", "Reducir las horas de uso tiene efecto medible",
                                 f"Según el histórico, cada hora diaria adicional de uso se asocia con "
                                 f"{lr.slope:.1f} kWh más al mes (r = {lr.rvalue:.2f}). Reducir media hora diaria "
                                 f"equivaldría a unos {lr.slope / 2:.1f} kWh menos. Es una asociación estadística, "
                                 "no una garantía causal.", ahorro_kwh=lr.slope / 2,
                                 ahorro_dinero=lr.slope / 2 * tarifa, categoria="Hábitos"))

    # --------------------------------------------------------------- pronóstico
    if pron is not None:
        if p_limite is not None and p_limite >= 0.5:
            recs.append(_rec("alta", "Es probable superar el límite de consumo",
                             f"La probabilidad estimada de superar {limite:.0f} kWh en {pron['periodo']} es "
                             f"{p_limite * 100:.1f}% (predicción: {pron['yhat']:.1f} kWh). Para volver al límite "
                             f"habría que reducir unos {max(pron['yhat'] - limite, 0):.1f} kWh.",
                             ahorro_kwh=max(pron["yhat"] - limite, 0), categoria="Pronóstico"))
        elif p_limite is not None and p_limite >= 0.2:
            recs.append(_rec("media", "Riesgo moderado de superar el límite",
                             f"Probabilidad de superar {limite:.0f} kWh el próximo mes: {p_limite * 100:.1f}% "
                             f"(predicción: {pron['yhat']:.1f} kWh).", categoria="Pronóstico"))
        elif p_limite is not None:
            recs.append(_rec("positiva", "Bajo riesgo de superar el límite",
                             f"Probabilidad de superar {limite:.0f} kWh el próximo mes: {p_limite * 100:.1f}%.",
                             categoria="Pronóstico"))
        fact = business.factura_s(pron["yhat"], settings)
        if fact > settings["presupuesto"]:
            exceso = fact - settings["presupuesto"]
            recs.append(_rec("alta", "La factura proyectada supera el presupuesto",
                             f"Factura estimada {money(fact)} frente a un presupuesto de "
                             f"{money(settings['presupuesto'])}. Para ajustarse habría que ahorrar unos "
                             f"{exceso / tarifa:.1f} kWh.", ahorro_kwh=exceso / tarifa, ahorro_dinero=exceso,
                             categoria="Factura"))

    # ---------------------------------------------------------- electrodomésticos
    tabla = business.tabla_electrodomesticos(equipos)
    if not tabla.empty and tabla["consumo_kwh"].sum() > 0:
        total = tabla["consumo_kwh"].sum()
        top = tabla.iloc[0]
        top3 = tabla.head(3)
        recs.append(_rec("media" if top["participacion_pct"] >= 25 else "baja",
                         f"Mayor consumidor: {top['nombre']}",
                         f"Aporta {top['consumo_kwh']:.1f} kWh/mes ({top['participacion_pct']:.1f}% del inventario). "
                         f"Los tres primeros ({', '.join(top3['nombre'])}) suman "
                         f"{top3['participacion_pct'].sum():.1f}% del consumo estimado.",
                         categoria="Electrodomésticos"))
        # Oportunidades: reducir un 20% el uso de los equipos de mayor consumo (uso no continuo).
        for _, e in tabla.head(4).iterrows():
            if e["horas_dia"] >= 24 or e["consumo_kwh"] <= 0:
                continue
            if any(p in str(e["nombre"]).lower() for p in ("nevera", "refriger", "congelador")):
                continue   # una nevera no se puede "usar menos horas": se excluye de este consejo
            ahorro = e["consumo_kwh"] * 0.20
            minutos = e["horas_dia"] * 0.20 * 60
            if ahorro * tarifa < 0.005 * total * tarifa:
                continue
            recs.append(_rec("media", f"Oportunidad de ahorro: {e['nombre']}",
                             f"Reducir su uso un 20% (≈ {minutos:.0f} min menos al día) ahorraría "
                             f"{ahorro:.1f} kWh/mes, unos {money(ahorro * tarifa)} al mes y "
                             f"{money(ahorro * tarifa * 12)} al año.", ahorro_kwh=ahorro,
                             ahorro_dinero=ahorro * tarifa, categoria="Ahorro"))
        continuos = tabla[(tabla["horas_dia"] >= 20) & (tabla["participacion_pct"] >= 2)]
        for _, e in continuos.iterrows():
            recs.append(_rec("baja", f"Equipo de uso continuo: {e['nombre']}",
                             f"Funciona {e['horas_dia']:.0f} h/día y consume {e['consumo_kwh']:.1f} kWh/mes "
                             f"({money(e['consumo_kwh'] * tarifa)}). Verifique si puede apagarse en horas sin uso.",
                             categoria="Electrodomésticos"))
        if len(df) >= 1:
            real = float(df["consumo_kwh"].iloc[-1])
            brecha = (real - total) / real * 100
            if brecha > 20:
                recs.append(_rec("media", "El inventario no explica todo el consumo",
                                 f"Los equipos registrados suman {total:.1f} kWh/mes, pero el último consumo fue "
                                 f"{real:.1f} kWh ({brecha:.0f}% sin explicar). Puede haber equipos sin registrar, "
                                 "consumo en espera (stand-by) u horas subestimadas.", categoria="Inventario"))
            elif brecha < -20:
                recs.append(_rec("baja", "El inventario sobreestima el consumo",
                                 f"Los equipos registrados suman {total:.1f} kWh/mes frente a {real:.1f} kWh "
                                 "medidos. Revise potencias y horas de uso declaradas.", categoria="Inventario"))

    if len(df) >= 6:
        mask = descriptive.atipicos_iqr(df["consumo_kwh"])
        if mask.any():
            meses = ", ".join(df.loc[mask, "etiqueta"].tolist()[:5])
            recs.append(_rec("baja", "Meses atípicos en el histórico",
                             f"Según el criterio 1.5·IQR, son atípicos: {meses}. Verifique si corresponden a "
                             "eventos puntuales; pueden distorsionar promedios y modelos.", categoria="Anomalía"))

    recs.sort(key=lambda r: ORDEN[r["nivel"]])
    return recs


def ahorro_potencial(recs: list[dict]) -> tuple[float, float]:
    """Suma del ahorro de las oportunidades por electrodoméstico (categoría 'Ahorro')."""
    kwh = sum(r["ahorro_kwh"] or 0 for r in recs if r["categoria"] == "Ahorro")
    dinero = sum(r["ahorro_dinero"] or 0 for r in recs if r["categoria"] == "Ahorro")
    return kwh, dinero
