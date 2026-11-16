"""Lógica de negocio: validaciones, consumo de electrodomésticos y facturación."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

PERIODO_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# ------------------------------------------------------------- validaciones
def validar_consumo(periodo, consumo_kwh, valor_factura, num_electrodomesticos, horas_uso) -> None:
    if not PERIODO_RE.match(str(periodo)):
        raise ValueError("El periodo debe tener el formato AAAA-MM (por ejemplo 2026-03).")
    if not consumo_kwh > 0:
        raise ValueError("El consumo debe ser mayor que 0 kWh.")
    if consumo_kwh > 20000:
        raise ValueError("El consumo supera 20.000 kWh; revise el valor para una vivienda.")
    if valor_factura < 0:
        raise ValueError("El valor de la factura no puede ser negativo.")
    if num_electrodomesticos < 0 or int(num_electrodomesticos) != num_electrodomesticos:
        raise ValueError("La cantidad de electrodomésticos debe ser un entero mayor o igual a 0.")
    if num_electrodomesticos > 500:
        raise ValueError("La cantidad de electrodomésticos no puede superar 500.")
    if not 0 <= horas_uso <= 24:
        raise ValueError("Las horas promedio de uso diario deben estar entre 0 y 24.")


def validar_electrodomestico(nombre, potencia_w, horas_dia, dias_mes, cantidad) -> None:
    if not str(nombre).strip():
        raise ValueError("El nombre del electrodoméstico es obligatorio.")
    if len(str(nombre)) > 60:
        raise ValueError("El nombre no puede superar 60 caracteres.")
    if not potencia_w > 0:
        raise ValueError("La potencia debe ser mayor que 0 W.")
    if not 0 <= horas_dia <= 24:
        raise ValueError("Las horas de uso diario deben estar entre 0 y 24.")
    if not 0 <= dias_mes <= 31:
        raise ValueError("Los días de utilización mensual deben estar entre 0 y 31.")
    if cantidad < 1 or int(cantidad) != cantidad:
        raise ValueError("La cantidad de unidades debe ser un entero mayor o igual a 1.")
    if potencia_w > 20000 or cantidad > 500:
        raise ValueError("La potencia no puede superar 20.000 W ni la cantidad 500 unidades.")


# -------------------------------------------------------- electrodomésticos
def consumo_electrodomestico(potencia_w: float, horas_dia: float, dias_mes: float, cantidad: float) -> float:
    """Consumo (kWh) = (Potencia en W × horas de uso × días de utilización × cantidad) / 1000."""
    return potencia_w * horas_dia * dias_mes * cantidad / 1000.0


def tabla_electrodomesticos(df: pd.DataFrame) -> pd.DataFrame:
    """Añade consumo estimado (kWh/mes) y participación (%) y ordena de mayor a menor."""
    df = df.copy()
    if df.empty:
        df["consumo_kwh"] = []
        df["participacion_pct"] = []
        return df
    df["consumo_kwh"] = consumo_electrodomestico(df["potencia_w"], df["horas_dia"], df["dias_mes"], df["cantidad"])
    total = df["consumo_kwh"].sum()
    df["participacion_pct"] = df["consumo_kwh"] / total * 100 if total > 0 else 0.0
    return df.sort_values("consumo_kwh", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------- facturación
def factura(consumo_kwh, tarifa_kwh, cargo_fijo: float = 0.0, estrato: int = 4,
            subsistencia: float = config.SUBSISTENCIA_KWH):
    """Factura estimada con el esquema colombiano de subsidios y contribuciones.

        factura = consumo × tarifa × (1 + contribución)
                  − subsidio × tarifa × min(consumo, consumo de subsistencia) + cargo fijo

    Estratos 1, 2 y 3: subsidio (60 %, 50 %, 15 %) solo sobre los primeros 130 kWh en Popayán.
    Estrato 4: tarifa plena. Estratos 5 y 6: contribución del 20 % sobre todo el consumo.
    Acepta números o arreglos de NumPy (para la simulación de Monte Carlo).
    """
    estrato = int(estrato)
    c = np.asarray(consumo_kwh, dtype=float)
    t = np.asarray(tarifa_kwh, dtype=float)
    total = (c * t * (1 + config.CONTRIBUCION[estrato])
             - config.SUBSIDIO[estrato] * t * np.minimum(c, subsistencia) + cargo_fijo)
    return float(total) if total.ndim == 0 else total


def subsidio_aplicado(consumo_kwh: float, tarifa_kwh: float, estrato: int,
                      subsistencia: float = config.SUBSISTENCIA_KWH) -> float:
    """Valor en pesos del subsidio (positivo) o de la contribución (negativo)."""
    estrato = int(estrato)
    return (config.SUBSIDIO[estrato] * tarifa_kwh * min(consumo_kwh, subsistencia)
            - config.CONTRIBUCION[estrato] * tarifa_kwh * consumo_kwh)


def kwh_para_presupuesto(presupuesto: float, tarifa_kwh: float, cargo_fijo: float = 0.0, estrato: int = 4,
                         subsistencia: float = config.SUBSISTENCIA_KWH) -> float:
    """Consumo (kWh) a partir del cual la factura supera el presupuesto (inversa de `factura`)."""
    if tarifa_kwh <= 0:
        raise ValueError("La tarifa debe ser mayor que 0.")
    estrato = int(estrato)
    neto = presupuesto - cargo_fijo
    if neto <= 0:
        return 0.0
    sub, con = config.SUBSIDIO[estrato], config.CONTRIBUCION[estrato]
    tramo1 = tarifa_kwh * (1 + con - sub)                 # precio dentro del consumo de subsistencia
    if neto <= tramo1 * subsistencia:
        return neto / tramo1
    return (neto + sub * tarifa_kwh * subsistencia) / (tarifa_kwh * (1 + con))


def factura_s(consumo_kwh, s: dict):
    """Factura con la configuración guardada (tarifa, cargo fijo y estrato)."""
    return factura(consumo_kwh, s["tarifa_kwh"], s["cargo_fijo"], int(s.get("estrato", 4)))


def sugerencia_registro(equipos: pd.DataFrame) -> dict | None:
    """Valores sugeridos para un registro mensual a partir del inventario de electrodomésticos."""
    if equipos is None or equipos.empty:
        return None
    t = tabla_electrodomesticos(equipos)
    # nº de electrodomésticos = tipos de equipo registrados; horas = promedio simple de sus horas diarias
    return {"consumo_kwh": round(float(t["consumo_kwh"].sum()), 1), "num_electrodomesticos": min(len(t), 500),
            "horas_uso": round(min(float(t["horas_dia"].mean()), 24.0), 1)}


# Potencias TÍPICAS de referencia (W) y horas de uso diario habituales. Son valores
# orientativos: la potencia real está en la placa o etiqueta de cada aparato.
CATALOGO = [
    ("Nevera", 150, 8.0, 30), ("Ducha eléctrica", 3500, 0.25, 30), ("Televisor", 100, 5.0, 30),
    ("Bombillo LED", 9, 5.0, 30), ("Bombillo ahorrador", 20, 5.0, 30), ("Computador portátil", 65, 6.0, 26),
    ("Computador de escritorio", 200, 6.0, 26), ("Lavadora", 500, 1.0, 12), ("Plancha", 1100, 1.0, 6),
    ("Licuadora", 400, 0.1, 30), ("Microondas", 1200, 0.15, 30), ("Olla arrocera", 700, 0.5, 26),
    ("Estufa eléctrica (un puesto)", 1200, 1.0, 30), ("Horno eléctrico", 1500, 0.5, 8), ("Cafetera", 800, 0.25, 30),
    ("Ventilador", 60, 4.0, 20), ("Calentador de ambiente", 1500, 2.0, 15), ("Secador de pelo", 1500, 0.15, 20),
    ("Equipo de sonido", 80, 2.0, 20), ("Consola de videojuegos", 150, 2.0, 20), ("Aspiradora", 1000, 0.5, 8),
    ("Router WiFi", 10, 24.0, 30), ("Cargador de celular", 10, 4.0, 30),
]
OTRO = "Otro (escribir el nombre)"
