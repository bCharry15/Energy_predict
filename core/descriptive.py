"""Estadística descriptiva del consumo (variable aleatoria X = consumo mensual en kWh)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def resumen(x) -> dict:
    """Medidas de tendencia central, dispersión y forma (varianza y desviación MUESTRALES, n-1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("No hay datos para analizar.")
    media = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else float("nan")
    sd = float(np.sqrt(var)) if n > 1 else float("nan")
    q1, q3 = (float(v) for v in np.percentile(x, [25, 75]))
    return {
        "n": n,
        "media": media,
        "mediana": float(np.median(x)),
        "moda": moda(x),
        "varianza": var,
        "desviacion": sd,
        "minimo": float(np.min(x)),
        "maximo": float(np.max(x)),
        "rango": float(np.max(x) - np.min(x)),
        "cv": float(sd / media * 100) if media != 0 and n > 1 else float("nan"),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "asimetria": float(stats.skew(x, bias=False)) if n > 2 else float("nan"),
        "curtosis": float(stats.kurtosis(x, bias=False)) if n > 3 else float("nan"),
        "error_estandar": float(sd / np.sqrt(n)) if n > 1 else float("nan"),
    }


def moda(x, decimales: int = 0) -> list[float]:
    """Moda sobre valores redondeados. Devuelve [] si ningún valor se repite (no hay moda).

    El consumo es una variable continua: la moda de datos sin agrupar solo tiene sentido
    si algún valor se repite; de lo contrario se usa la clase modal (ver tabla de frecuencias).
    """
    vals = pd.Series(np.round(np.asarray(x, dtype=float), decimales))
    conteo = vals.value_counts()
    if conteo.empty or conteo.iloc[0] < 2:
        return []
    return sorted(float(v) for v in conteo[conteo == conteo.iloc[0]].index)


def n_clases_sturges(n: int) -> int:
    """Regla de Sturges: k = 1 + log2(n)."""
    return max(int(np.ceil(1 + np.log2(n))), 3) if n > 1 else 1


def tabla_frecuencias(x, k: int | None = None) -> pd.DataFrame:
    """Distribución de frecuencias con k clases de igual amplitud (Sturges por defecto)."""
    x = np.asarray(x, dtype=float)
    k = k or n_clases_sturges(len(x))
    fi, bordes = np.histogram(x, bins=k)
    fr = fi / fi.sum()
    return pd.DataFrame({
        "Clase": [f"[{bordes[i]:.1f} – {bordes[i + 1]:.1f}{']' if i == k - 1 else ')'}" for i in range(k)],
        "Marca de clase": (bordes[:-1] + bordes[1:]) / 2,
        "Frecuencia (fi)": fi,
        "Frec. relativa (%)": fr * 100,
        "Frec. acumulada (Fi)": np.cumsum(fi),
        "Frec. rel. acumulada (%)": np.cumsum(fr) * 100,
    })


def clase_modal(tabla: pd.DataFrame) -> str:
    return str(tabla.loc[tabla["Frecuencia (fi)"].idxmax(), "Clase"])


def atipicos_iqr(x) -> np.ndarray:
    """Máscara booleana de valores atípicos según el criterio de Tukey (1.5·IQR)."""
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
