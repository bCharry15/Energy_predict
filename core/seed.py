"""Generación de datos SINTÉTICOS de demostración ambientados en Popayán.

Los datos NO son mediciones reales de ninguna vivienda: se generan con un modelo conocido
para que la aplicación funcione desde el primer inicio. Todos se marcan con es_simulado = 1.
Lo que SÍ es real son las reglas de facturación de Popayán (consumo de subsistencia de
130 kWh, subsidios por estrato) y la tarifa de referencia de CEO (ver core/config.py).

Vivienda simulada: estrato 3, clima templado, sin aire acondicionado, con ducha eléctrica.
Modelo generador (supuesto explícito):
    consumo = 40 + 14·horas_uso + 4·n_equipos + estacionalidad + ruido,  ruido ~ N(0, 6²)
    factura = factura_estrato_3(consumo, tarifa_mes) · (1 + e),          e ~ N(0, 0.015²)
"""
from __future__ import annotations

import numpy as np

from . import business, config

SEED = 20261114
N_MESES = 36
INICIO = (2023, 11)  # primer periodo simulado -> último: 2026-10
ESTRATO_DEMO = 3

# nombre, potencia (W), horas/día, días/mes, cantidad
ELECTRODOMESTICOS_DEMO = [
    ("Nevera", 150, 8.0, 30, 1),
    ("Ducha eléctrica", 3500, 0.25, 30, 1),
    ("Televisor", 100, 5.0, 30, 1),
    ("Bombillos LED", 9, 5.0, 30, 8),
    ("Computador portátil", 65, 6.0, 26, 1),
    ("Olla arrocera", 700, 0.5, 26, 1),
    ("Router WiFi", 10, 24.0, 30, 1),
    ("Plancha", 1100, 1.0, 6, 1),
    ("Lavadora", 500, 1.0, 12, 1),
    ("Microondas", 1200, 0.15, 30, 1),
    ("Cargadores de celular", 10, 4.0, 30, 3),
    ("Equipo de sonido", 80, 2.0, 20, 1),
    ("Licuadora", 400, 0.1, 30, 1),
]


def generar_consumo_demo(n: int = N_MESES, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    y, m = INICIO
    registros = []
    estac_horas = {12: 0.8, 1: 0.5, 6: 0.4, 7: 0.4}   # vacaciones: más horas en casa
    estac_kwh = {12: 5.0, 1: 2.0}                      # luces navideñas, visitas
    for i in range(n):
        equipos = 9 + int(i // 10) + int(rng.integers(0, 2))
        horas = 4.6 + estac_horas.get(m, 0.0) + rng.normal(0, 0.6)
        horas = float(np.clip(round(horas, 1), 3.0, 9.0))
        consumo = 40 + 14 * horas + 4 * equipos + estac_kwh.get(m, 0.0) + rng.normal(0, 6)
        consumo = round(float(max(consumo, 50.0)), 1)
        tarifa = 930 + (config.TARIFA_REFERENCIA - 930) * i / (n - 1)
        factura = business.factura(consumo, tarifa, 0.0, ESTRATO_DEMO) * (1 + rng.normal(0, 0.015))
        registros.append({
            "periodo": f"{y:04d}-{m:02d}",
            "consumo_kwh": consumo,
            "valor_factura": float(round(factura / 100) * 100),
            "num_electrodomesticos": equipos,
            "horas_uso": horas,
            "es_simulado": 1,
        })
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return registros


def generar_electrodomesticos_demo() -> list[dict]:
    return [
        {"nombre": n, "potencia_w": float(p), "horas_dia": float(h),
         "dias_mes": int(d), "cantidad": int(c), "es_simulado": 1}
        for n, p, h, d, c in ELECTRODOMESTICOS_DEMO
    ]
