"""Modelos probabilísticos para el consumo mensual X (variable aleatoria continua).

Se distinguen tres formas de calcular P(X > límite):
  1. Empírica (frecuentista): proporción de meses históricos que superaron el límite.
  2. Paramétrica marginal: distribución ajustada por máxima verosimilitud al histórico
     (Normal, Lognormal o Gamma; se elige por AIC y se contrasta la bondad de ajuste).
  3. Predictiva condicional: distribución t alrededor de la predicción de la regresión para
     el próximo mes (condicionada a los predictores). Solo válida si los residuos son ≈ normales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NOMBRES = {"norm": "Normal", "lognorm": "Lognormal", "gamma": "Gamma"}


def shapiro(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    w, p = stats.shapiro(x)
    return float(w), float(p)


def ajustar_distribuciones(x) -> list[dict]:
    """Ajusta Normal, Lognormal y Gamma por MLE y las ordena por AIC (menor = mejor).

    Nota: el valor p de Kolmogorov-Smirnov es optimista porque los parámetros se estiman
    con los mismos datos (problema de Lilliefors); se usa solo como indicio.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 5 or np.ptp(x) == 0:
        raise ValueError("Se necesitan al menos 5 registros con valores distintos para ajustar distribuciones.")
    resultados = []
    for clave in ("norm", "lognorm", "gamma"):
        dist = getattr(stats, clave)
        try:
            if clave == "norm":
                params = dist.fit(x)
            else:
                if np.any(x <= 0):
                    continue
                params = dist.fit(x, floc=0)
            congelada = dist(*params)
            ll = float(np.sum(congelada.logpdf(x)))
            if not np.isfinite(ll):
                continue
            ks, p = stats.kstest(x, congelada.cdf)
            resultados.append({"clave": clave, "nombre": NOMBRES[clave], "dist": congelada, "params": params,
                               "loglik": ll, "aic": 2 * 2 - 2 * ll, "ks": float(ks), "ks_p": float(p)})
        except Exception:
            continue
    if not resultados:
        raise ValueError("No fue posible ajustar ninguna distribución a los datos.")
    return sorted(resultados, key=lambda r: r["aic"])


def tabla_ajustes(ajustes: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Distribución": a["nombre"], "Log-verosimilitud": a["loglik"], "AIC": a["aic"],
        "KS (D)": a["ks"], "Valor p KS": a["ks_p"],
        "Ajuste": "Aceptable" if a["ks_p"] > 0.05 else "Rechazado"} for a in ajustes])


def descripcion_parametros(a: dict) -> str:
    p = a["params"]
    if a["clave"] == "norm":
        return f"μ = {p[0]:.2f}, σ = {p[1]:.2f}"
    if a["clave"] == "lognorm":
        return f"μ(log) = {np.log(p[2]):.3f}, σ(log) = {p[0]:.3f}"
    return f"forma k = {p[0]:.2f}, escala θ = {p[2]:.2f}"


def wilson(k: int, n: int, confianza: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción."""
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(0.5 + confianza / 2)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    semi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(max(centro - semi, 0.0)), float(min(centro + semi, 1.0))


def prob_empirica(x, limite: float) -> dict:
    x = np.asarray(x, dtype=float)
    k, n = int(np.sum(x > limite)), len(x)
    lo, hi = wilson(k, n)
    return {"p": k / n if n else float("nan"), "k": k, "n": n, "lo": lo, "hi": hi}


def prob_parametrica(dist, limite: float) -> float:
    """P(X > límite) = 1 − F(límite) con la distribución ajustada."""
    return float(dist.sf(limite))


def prob_predictiva(yhat: float, se_pred: float, dof: int, limite: float) -> float:
    """P(Y_nuevo > límite) con (Y − ŷ)/se ~ t(n−p)."""
    if se_pred <= 0:
        return float(yhat > limite)
    return float(stats.t.sf((limite - yhat) / se_pred, dof))


def prob_aumento_empirica(x) -> dict:
    """Proporción histórica de meses en que el consumo subió respecto al mes anterior."""
    d = np.diff(np.asarray(x, dtype=float))
    k, n = int(np.sum(d > 0)), len(d)
    lo, hi = wilson(k, n)
    return {"p": k / n if n else float("nan"), "k": k, "n": n, "lo": lo, "hi": hi}
