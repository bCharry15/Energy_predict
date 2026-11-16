"""Procesamiento de datos: carga, variables derivadas e importación/exportación CSV."""
from __future__ import annotations

import io
import re

import pandas as pd

from . import config, database
from .business import validar_consumo

PERIODO_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena cronológicamente y añade variables derivadas."""
    df = df.sort_values("periodo").reset_index(drop=True).copy()
    df["tendencia"] = range(1, len(df) + 1)                # mes nº 1..n
    df["etiqueta"] = [config.periodo_label(p) for p in df["periodo"]]
    df["tarifa_implicita"] = df["valor_factura"] / df["consumo_kwh"]
    df["variacion_pct"] = df["consumo_kwh"].pct_change() * 100
    return df


def cargar_consumo() -> pd.DataFrame:
    return preparar(database.listar_consumo())


def exportar_csv(df: pd.DataFrame) -> bytes:
    return df[database.CONSUMO_COLS].to_csv(index=False).encode("utf-8")


def plantilla_csv() -> bytes:
    ejemplo = pd.DataFrame([{"periodo": "2026-01", "consumo_kwh": 210.5, "valor_factura": 189000,
                             "num_electrodomesticos": 11, "horas_uso": 6.5}])
    return ejemplo.to_csv(index=False).encode("utf-8")


def leer_csv(contenido: bytes) -> tuple[list[dict], list[str]]:
    """Valida un CSV del usuario. Devuelve (registros válidos, lista de errores)."""
    try:
        df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", dtype={"periodo": str})
    except Exception as exc:  # archivo ilegible
        return [], [f"No se pudo leer el archivo CSV: {exc}"]
    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = [c for c in database.CONSUMO_COLS if c not in df.columns]
    if faltan:
        return [], [f"Faltan columnas obligatorias: {', '.join(faltan)}"]
    registros, errores, vistos = [], [], set()
    for i, row in df.iterrows():
        fila = i + 2  # +1 encabezado, +1 base 1
        try:
            periodo = str(row["periodo"]).strip()[:7]
            valores = {c: float(row[c]) for c in database.CONSUMO_COLS[1:]}
            if any(pd.isna(v) for v in valores.values()):
                raise ValueError("hay celdas vacías")
            validar_consumo(periodo, **valores)
            if periodo in vistos:
                raise ValueError(f"el periodo {periodo} está repetido en el archivo")
            vistos.add(periodo)
            valores["num_electrodomesticos"] = int(valores["num_electrodomesticos"])
            registros.append({"periodo": periodo, **valores})
        except (ValueError, TypeError) as exc:
            errores.append(f"Fila {fila}: {exc}")
    return registros, errores
