"""Configuración global de ENERGY PREDICT (rutas, valores por defecto, etiquetas)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

APP_NAME = "ENERGY PREDICT"
APP_TAGLINE = "Sistema Inteligente de Predicción del Consumo Eléctrico Residencial"

# Valores por defecto de la configuración (se guardan en SQLite al iniciar).
# ----------------------------------------------------------------- Popayán
CIUDAD = "Popayán (Cauca)"
OPERADOR = "Compañía Energética de Occidente (CEO)"
# Popayán está a más de 1.000 m s. n. m.: el consumo de subsistencia es 130 kWh/mes
# (Resolución UPME 0355 de 2004). Por debajo de 1.000 m sería 173 kWh/mes.
SUBSISTENCIA_KWH = 130.0
# Subsidio máximo sobre el consumo de subsistencia (estratos 1, 2 y 3) y contribución
# de solidaridad sobre todo el consumo (estratos 5 y 6). El estrato 4 paga tarifa plena.
SUBSIDIO = {1: 0.60, 2: 0.50, 3: 0.15, 4: 0.0, 5: 0.0, 6: 0.0}
CONTRIBUCION = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.20, 6: 0.20}
# Costo unitario (CU) de referencia de CEO Cauca informado por la Superintendencia de
# Servicios Públicos en 2025: 1.048,22 $/kWh. La tarifa cambia cada mes: el usuario debe
# reemplazarla por la que aparece en su recibo.
TARIFA_REFERENCIA = 1048.0
FUENTE_TARIFA = "CU de referencia de CEO Cauca, Superservicios 2025 (1.048,22 $/kWh)"

DEMO_VERSION = 2.0   # cambia cuando cambian los datos de demostración

DEFAULT_SETTINGS = {
    "tarifa_kwh": TARIFA_REFERENCIA,  # $ por kWh (costo unitario, tarifa plena)
    "cargo_fijo": 0.0,       # $ por mes (cargos que no dependen del consumo)
    "limite_kwh": 170.0,     # límite de consumo mensual que el usuario no quiere superar
    "presupuesto": 160000.0, # presupuesto mensual de factura ($)
    "estrato": 3.0,          # estrato socioeconómico de la vivienda (1 a 6)
    "version_demo": DEMO_VERSION,
}

# Variables predictoras disponibles para la regresión.
FEATURE_LABELS = {
    "horas_uso": "Horas promedio de uso diario",
    "num_electrodomesticos": "Cantidad de electrodomésticos",
    "tendencia": "Tendencia temporal (mes nº)",
}
DEFAULT_FEATURES = ["horas_uso", "num_electrodomesticos"]

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

MIN_REGISTROS_ANALISIS = 6    # mínimo para estadística descriptiva / probabilidad
MIN_REGISTROS_REGRESION = 12  # mínimo razonable para separar entrenamiento y prueba


def db_path() -> Path:
    """Ruta de la base de datos (configurable con ENERGY_PREDICT_DB, útil en pruebas)."""
    custom = os.environ.get("ENERGY_PREDICT_DB")
    return Path(custom) if custom else DATA_DIR / "energy_predict.db"


def periodo_label(periodo: str) -> str:
    """'2025-03' -> 'Mar 2025'."""
    try:
        y, m = periodo.split("-")
        return f"{MESES[int(m) - 1]} {y}"
    except Exception:
        return str(periodo)


def next_periodo(periodo: str) -> str:
    y, m = (int(v) for v in periodo.split("-"))
    m += 1
    if m > 12:
        y, m = y + 1, 1
    return f"{y:04d}-{m:02d}"


def money(v: float) -> str:
    """Formato de moneda con separador de miles '.' (p. ej. $150.000)."""
    return "$" + f"{v:,.0f}".replace(",", ".")
