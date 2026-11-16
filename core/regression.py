"""Regresión lineal por mínimos cuadrados ordinarios (MCO).

- El ajuste y la inferencia (errores estándar, valores p, intervalos de predicción) se
  implementan con álgebra matricial para que las fórmulas sean verificables.
- La evaluación usa scikit-learn y NUNCA mide el error con los mismos datos de entrenamiento:
  (1) partición cronológica entrenamiento/prueba y (2) validación walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config

TARGET = "consumo_kwh"


@dataclass
class OLSResult:
    features: list[str]
    coef: np.ndarray          # [intercepto, b1, ..., bk]
    cov: np.ndarray           # matriz de covarianza de los coeficientes
    xtx_inv: np.ndarray
    sigma2: float             # varianza residual s² = SCE / (n - p)
    n: int
    p: int                    # nº de parámetros (incluye intercepto)
    r2: float
    r2_adj: float
    fitted: np.ndarray
    residuals: np.ndarray

    @property
    def dof(self) -> int:
        return self.n - self.p

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma2))

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def t_values(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.coef / self.se

    @property
    def p_values(self) -> np.ndarray:
        return 2 * stats.t.sf(np.abs(self.t_values), self.dof)

    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        k = len(self.features)
        X = X.reshape(len(X), 0) if k == 0 else X.reshape(-1, k)
        return np.column_stack([np.ones(len(X)), X])

    def predict(self, X) -> np.ndarray:
        return self._design(X) @ self.coef

    def se_prediccion(self, X) -> np.ndarray:
        """Error estándar de una observación nueva: s·sqrt(1 + x0ᵀ(XᵀX)⁻¹x0)."""
        A = self._design(X)
        h = np.einsum("ij,jk,ik->i", A, self.xtx_inv, A)
        return np.sqrt(self.sigma2 * (1 + h))

    def intervalo_prediccion(self, X, confianza: float = 0.90):
        """Intervalo de predicción (t de Student con n-p g.l.). Requiere residuos ≈ normales."""
        y = self.predict(X)
        se = self.se_prediccion(X)
        tcrit = stats.t.ppf(0.5 + confianza / 2, self.dof)
        return y, y - tcrit * se, y + tcrit * se, se

    def ecuacion(self) -> str:
        partes = [f"{self.coef[0]:.2f}"]
        for b, f in zip(self.coef[1:], self.features):
            partes.append(f"{'+' if b >= 0 else '−'} {abs(b):.3f}·{f}")
        return "consumo_kwh = " + " ".join(partes)

    def tabla_coeficientes(self) -> pd.DataFrame:
        nombres = ["Intercepto"] + [config.FEATURE_LABELS.get(f, f) for f in self.features]
        return pd.DataFrame({"Parámetro": nombres, "Coeficiente": self.coef, "Error estándar": self.se,
                             "t": self.t_values, "Valor p": self.p_values})


def fit_ols(X, y, features: list[str]) -> OLSResult:
    """Ajuste MCO: b = (XᵀX)⁻¹Xᵀy. Con features=[] ajusta el modelo de solo intercepto (media)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    X = np.asarray(X, dtype=float).reshape(n, len(features))
    A = np.column_stack([np.ones(n), X])
    p = A.shape[1]
    if n < p + 2:
        raise ValueError(f"Datos insuficientes: se necesitan al menos {p + 2} registros para "
                         f"estimar {p} parámetros.")
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("Alguna variable predictora es constante o es combinación lineal de otras; "
                         "no se puede estimar el modelo. Cambie la selección de variables.")
    xtx_inv = np.linalg.inv(A.T @ A)
    coef = xtx_inv @ A.T @ y
    fitted = A @ coef
    resid = y - fitted
    sce = float(resid @ resid)
    sct = float(((y - y.mean()) ** 2).sum())
    sigma2 = sce / (n - p)
    r2 = 1 - sce / sct if sct > 0 else float("nan")
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p) if sct > 0 else float("nan")
    return OLSResult(list(features), coef, sigma2 * xtx_inv, xtx_inv, sigma2, n, p, r2, r2_adj, fitted, resid)


def ajustar(df: pd.DataFrame, features: list[str]) -> OLSResult:
    return fit_ols(df[features].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float), features)


# ---------------------------------------------------------------- evaluación
def _metricas(y, pred) -> dict:
    return {"r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def evaluar_holdout(df: pd.DataFrame, features: list[str], frac_prueba: float = 0.25) -> dict:
    """Partición CRONOLÓGICA: se entrena con los primeros meses y se evalúa con los últimos.

    No se baraja porque los datos son una serie temporal: evaluar con meses anteriores a los
    de entrenamiento filtraría información del futuro.
    """
    n = len(df)
    n_test = max(int(round(n * frac_prueba)), 3)
    n_train = n - n_test
    if n_train < len(features) + 3:
        raise ValueError("No hay suficientes registros para separar entrenamiento y prueba con "
                         "estas variables. Agregue más meses o reduzca las variables.")
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    if features:
        modelo = LinearRegression().fit(train[features].to_numpy(float), train[TARGET].to_numpy(float))
        pred_train = modelo.predict(train[features].to_numpy(float))
        pred_test = modelo.predict(test[features].to_numpy(float))
    else:
        pred_train = np.full(n_train, train[TARGET].mean())
        pred_test = np.full(n_test, train[TARGET].mean())
    # Referencia ingenua: predecir siempre la media de entrenamiento.
    base = np.full(n_test, train[TARGET].mean())
    return {
        "n_train": n_train, "n_test": n_test,
        "train": _metricas(train[TARGET], pred_train),
        "test": _metricas(test[TARGET], pred_test),
        "baseline": _metricas(test[TARGET], base),
        "pred_train": pred_train, "pred_test": pred_test,
        "idx_train": train.index.to_numpy(), "idx_test": test.index.to_numpy(),
    }


def walk_forward(df: pd.DataFrame, features: list[str], min_train: int | None = None) -> pd.DataFrame:
    """Validación de origen móvil: para cada mes i se entrena con los meses < i y se predice i.

    Cada predicción es fuera de muestra (el mes predicho nunca participa en el ajuste).
    """
    n = len(df)
    min_train = min_train or max(len(features) + 4, min(12, n // 2))
    filas = []
    X = df[features].to_numpy(float) if features else np.empty((n, 0))
    y = df[TARGET].to_numpy(float)
    for i in range(min_train, n):
        try:
            if features:
                pred = float(LinearRegression().fit(X[:i], y[:i]).predict(X[i:i + 1])[0])
            else:
                pred = float(y[:i].mean())
        except Exception:
            continue
        filas.append({"periodo": df["periodo"].iloc[i], "etiqueta": df["etiqueta"].iloc[i],
                      "real": y[i], "predicho": pred})
    return pd.DataFrame(filas, columns=["periodo", "etiqueta", "real", "predicho"])


def metricas_walk_forward(wf: pd.DataFrame) -> dict | None:
    if len(wf) < 2:
        return None
    return _metricas(wf["real"], wf["predicho"])


# --------------------------------------------------------------- diagnóstico
def durbin_watson(resid) -> float:
    e = np.asarray(resid, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def breusch_pagan(resid, X) -> tuple[float, float]:
    """Prueba de Breusch-Pagan (versión de Koenker): LM = n·R² de e² sobre X ~ χ²(k)."""
    e2 = np.asarray(resid, dtype=float) ** 2
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, e2, rcond=None)
    fitted = A @ beta
    sct = ((e2 - e2.mean()) ** 2).sum()
    r2 = 1 - ((e2 - fitted) ** 2).sum() / sct if sct > 0 else 0.0
    lm = n * r2
    return float(lm), float(stats.chi2.sf(lm, k))


def vif(X) -> np.ndarray:
    """Factor de inflación de la varianza de cada predictor: 1 / (1 - R²_j)."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    out = np.ones(k)
    if k < 2:
        return out
    for j in range(k):
        otros = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(otros, X[:, j], rcond=None)
        res = X[:, j] - otros @ beta
        sct = ((X[:, j] - X[:, j].mean()) ** 2).sum()
        r2 = 1 - (res @ res) / sct if sct > 0 else 0.0
        out[j] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return out


def diagnosticos(res: OLSResult, X) -> list[dict]:
    """Verificación de los supuestos del modelo lineal (α = 0.05)."""
    out = []
    e = res.residuals
    if len(e) >= 3 and np.ptp(e) > 0:
        w, p = stats.shapiro(e)
        out.append({"Supuesto": "Normalidad de los residuos", "Prueba": "Shapiro-Wilk",
                    "Estadístico": float(w), "Valor p": float(p), "cumple": bool(p > 0.05),
                    "Interpretación": "No se rechaza la normalidad" if p > 0.05 else
                    "Se rechaza la normalidad: los intervalos y probabilidades basados en t no son fiables"})
    dw = durbin_watson(e)
    ok = 1.5 <= dw <= 2.5
    out.append({"Supuesto": "Independencia de los residuos", "Prueba": "Durbin-Watson",
                "Estadístico": dw, "Valor p": float("nan"), "cumple": bool(ok),
                "Interpretación": "Sin evidencia de autocorrelación (≈ 2)" if ok else
                "Posible autocorrelación de primer orden (regla práctica: fuera de 1.5–2.5)"})
    if res.features:
        lm, p = breusch_pagan(e, X)
        out.append({"Supuesto": "Varianza constante (homocedasticidad)", "Prueba": "Breusch-Pagan",
                    "Estadístico": lm, "Valor p": p, "cumple": bool(p > 0.05),
                    "Interpretación": "No se rechaza la homocedasticidad" if p > 0.05 else
                    "Hay evidencia de heterocedasticidad"})
        if len(res.features) > 1:
            v = vif(X)
            vmax = float(np.max(v))
            out.append({"Supuesto": "Ausencia de multicolinealidad", "Prueba": "VIF máximo",
                        "Estadístico": vmax, "Valor p": float("nan"), "cumple": bool(vmax < 5),
                        "Interpretación": "Predictores poco correlacionados (VIF < 5)" if vmax < 5 else
                        "Predictores muy correlacionados (VIF ≥ 5): coeficientes inestables"})
    return out


def residuos_normales(res: OLSResult) -> bool | None:
    e = res.residuals
    if len(e) < 3 or np.ptp(e) == 0:
        return None
    return bool(stats.shapiro(e)[1] > 0.05)


# ----------------------------------------------------------------- pronóstico
def valores_proximo_mes(df: pd.DataFrame, features: list[str]) -> dict:
    """Supuesto por defecto para el próximo mes: horas = media de los últimos 3 meses;
    electrodomésticos = último valor registrado; tendencia = n + 1."""
    base = {"horas_uso": float(df["horas_uso"].tail(3).mean()),
            "num_electrodomesticos": float(df["num_electrodomesticos"].iloc[-1]),
            "tendencia": float(len(df) + 1)}
    return {f: base[f] for f in features}


def features_utilizables(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Reduce la selección si hay pocos datos o variables constantes (hasta solo intercepto)."""
    feats = list(config.DEFAULT_FEATURES if features is None else features)
    feats = [f for f in feats if df[f].nunique() > 1]
    while feats and len(df) < len(feats) + 6:
        feats.pop()
    while feats:
        try:
            ajustar(df, feats)
            break
        except ValueError:
            feats.pop()
    return feats


def pronostico(df: pd.DataFrame, features: list[str] | None = None, x_next: dict | None = None,
               confianza: float = 0.90) -> dict:
    """Predicción del próximo mes con el modelo reajustado sobre TODO el histórico.

    (La calidad predictiva se informa aparte con holdout y walk-forward.) Si no hay datos
    suficientes para los predictores, se recurre al modelo de solo intercepto (la media).
    """
    feats = features_utilizables(df, features)
    res = ajustar(df, feats)
    x = valores_proximo_mes(df, feats)
    if x_next:
        x.update({k: float(v) for k, v in x_next.items() if k in feats})
    fila = np.array([[x[f] for f in feats]], dtype=float)
    y, lo, hi, se = res.intervalo_prediccion(fila, confianza)
    return {"modelo": res, "features": feats, "x": x, "yhat": float(y[0]), "lo": float(max(lo[0], 0.0)),
            "hi": float(hi[0]), "se": float(se[0]), "dof": res.dof, "confianza": confianza,
            "periodo": config.next_periodo(df["periodo"].iloc[-1])}


def cuantil_predictivo(pron: dict, q: float) -> float:
    """Cuantil q de la distribución predictiva t del próximo mes."""
    return float(max(pron["yhat"] + stats.t.ppf(q, pron["dof"]) * pron["se"], 0.0))
