"""Orquestación: combina datos, pronóstico y probabilidad para las vistas de resumen."""
from __future__ import annotations

import pandas as pd

from . import business, probability, regression


def resumen_general(df: pd.DataFrame, settings: dict) -> dict:
    """Indicadores del dashboard. Requiere al menos 3 registros."""
    x = df["consumo_kwh"]
    pron = regression.pronostico(df)
    p_lim = probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], settings["limite_kwh"])
    lim_presupuesto = business.kwh_para_presupuesto(settings["presupuesto"], settings["tarifa_kwh"],
                                                    settings["cargo_fijo"], int(settings.get("estrato", 4)))
    p_pres = probability.prob_predictiva(pron["yhat"], pron["se"], pron["dof"], lim_presupuesto)
    actual, anterior = float(x.iloc[-1]), float(x.iloc[-2])
    return {
        "actual": actual,
        "anterior": anterior,
        "variacion_pct": (actual - anterior) / anterior * 100,
        "promedio": float(x.mean()),
        "pronostico": pron,
        "factura_estimada": business.factura_s(pron["yhat"], settings),
        "p_limite": p_lim,
        "p_presupuesto": p_pres,
        "residuos_normales": regression.residuos_normales(pron["modelo"]),
    }
