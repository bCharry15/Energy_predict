"""Simulación de Monte Carlo del consumo y la factura del próximo mes.

Modelos disponibles (todos con parámetros estimados del histórico o supuestos explícitos):
  - "regresion": consumo = b0 + Σ bj·xj + ε, propagando tres fuentes de incertidumbre:
        (a) coeficientes  b ~ Normal multivariada(b̂, Cov(b̂))
        (b) entradas      horas de uso ~ Normal(valor esperado, σ histórica), truncada a [0, 24]
        (c) ruido         ε remuestreado de los residuos (bootstrap) o ~ Normal(0, s²)
  - "distribucion": muestreo de la distribución paramétrica ajustada al histórico.
  - "bootstrap": remuestreo con reemplazo del consumo histórico (sin supuesto de forma).
La tarifa se simula como Normal(tarifa, (cv·tarifa)²) truncada en valores positivos.
"""
from __future__ import annotations

import numpy as np

from .regression import OLSResult

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def simular_regresion(res: OLSResult, x_centro: dict, sd_entradas: dict, n_sims: int,
                      rng: np.random.Generator, ruido: str = "bootstrap",
                      incertidumbre_parametros: bool = True) -> np.ndarray:
    k = len(res.features)
    X = np.empty((n_sims, k))
    for j, f in enumerate(res.features):
        sd = float(sd_entradas.get(f, 0.0) or 0.0)
        col = rng.normal(x_centro[f], sd, n_sims) if sd > 0 else np.full(n_sims, float(x_centro[f]))
        if f == "horas_uso":
            col = np.clip(col, 0.0, 24.0)
        X[:, j] = col
    A = np.column_stack([np.ones(n_sims), X])
    if incertidumbre_parametros:
        betas = rng.multivariate_normal(res.coef, res.cov, size=n_sims, method="svd")
    else:
        betas = np.tile(res.coef, (n_sims, 1))
    media = np.einsum("ij,ij->i", A, betas)
    if ruido == "bootstrap":
        # Residuos reescalados por sqrt(n/(n-p)) para compensar la subestimación de la varianza.
        e = rng.choice(res.residuals * np.sqrt(res.n / res.dof), size=n_sims, replace=True)
    else:
        e = rng.normal(0.0, res.sigma, n_sims)
    return np.clip(media + e, 0.0, None)


def simular_distribucion(dist, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(dist.rvs(size=n_sims, random_state=rng), 0.0, None)


def simular_bootstrap(x, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.asarray(x, dtype=float), size=n_sims, replace=True)


def simular_tarifa(tarifa: float, cv: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    if cv <= 0:
        return np.full(n_sims, float(tarifa))
    return np.clip(rng.normal(tarifa, tarifa * cv, n_sims), tarifa * 0.01, None)


def prob_superar(sims: np.ndarray, limite: float) -> dict:
    """Probabilidad estimada y su error estándar de Monte Carlo: sqrt(p(1−p)/N)."""
    n = len(sims)
    p = float(np.mean(sims > limite))
    return {"p": p, "ee": float(np.sqrt(p * (1 - p) / n))}


def resumen(sims: np.ndarray) -> dict:
    n = len(sims)
    sd = float(np.std(sims, ddof=1))
    pcts = np.percentile(sims, PERCENTILES)
    return {"n": n, "media": float(np.mean(sims)), "desviacion": sd, "ee_media": sd / np.sqrt(n),
            "minimo": float(np.min(sims)), "maximo": float(np.max(sims)),
            "percentiles": {p: float(v) for p, v in zip(PERCENTILES, pcts)}}


def intervalo(sims: np.ndarray, confianza: float = 0.90) -> tuple[float, float]:
    a = (1 - confianza) / 2 * 100
    lo, hi = np.percentile(sims, [a, 100 - a])
    return float(lo), float(hi)


def media_acumulada(sims: np.ndarray, puntos: int = 200):
    """Convergencia de la media (ley de los grandes números)."""
    n = len(sims)
    idx = np.unique(np.linspace(10, n, min(puntos, n), dtype=int))
    acum = np.cumsum(sims)
    return idx, acum[idx - 1] / idx
