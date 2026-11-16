"""Capa de persistencia (SQLite). Solo SQL y acceso a datos; sin lógica estadística."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd

from . import config, seed

SCHEMA = """
CREATE TABLE IF NOT EXISTS consumo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    periodo TEXT NOT NULL UNIQUE,              -- 'AAAA-MM'
    consumo_kwh REAL NOT NULL CHECK (consumo_kwh > 0),
    valor_factura REAL NOT NULL CHECK (valor_factura >= 0),
    num_electrodomesticos INTEGER NOT NULL CHECK (num_electrodomesticos >= 0),
    horas_uso REAL NOT NULL CHECK (horas_uso >= 0 AND horas_uso <= 24),
    es_simulado INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS electrodomesticos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    potencia_w REAL NOT NULL CHECK (potencia_w > 0),
    horas_dia REAL NOT NULL CHECK (horas_dia >= 0 AND horas_dia <= 24),
    dias_mes INTEGER NOT NULL CHECK (dias_mes >= 0 AND dias_mes <= 31),
    cantidad INTEGER NOT NULL CHECK (cantidad >= 1),
    es_simulado INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS configuracion (
    clave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""

CONSUMO_COLS = ["periodo", "consumo_kwh", "valor_factura", "num_electrodomesticos", "horas_uso"]
EQUIPO_COLS = ["nombre", "potencia_w", "horas_dia", "dias_mes", "cantidad"]


@contextmanager
def connect():
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(seed_demo: bool = True) -> None:
    """Crea el esquema y, la primera vez, carga los datos de demostración."""
    with connect() as c:
        c.executescript(SCHEMA)
        previas = {r["clave"]: r["valor"] for r in c.execute("SELECT clave, valor FROM configuracion")}
        primera_vez = not previas
        antigua = bool(previas) and float(previas.get("version_demo", 1)) < config.DEMO_VERSION
        for k, v in config.DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO configuracion (clave, valor) VALUES (?, ?)", (k, str(v)))
        del_usuario = (c.execute("SELECT COUNT(*) FROM consumo WHERE es_simulado=0").fetchone()[0]
                       + c.execute("SELECT COUNT(*) FROM electrodomesticos WHERE es_simulado=0").fetchone()[0])
    if seed_demo and primera_vez:
        cargar_demo()
    elif antigua:
        # Base creada con una versión anterior: si solo tiene datos de demostración se
        # regeneran (versión Popayán); si el usuario ya cargó datos propios, se respetan.
        if seed_demo and del_usuario == 0:
            restablecer_demo()
        else:
            set_setting("version_demo", config.DEMO_VERSION)


def cargar_demo() -> None:
    """Inserta los datos sintéticos (sin duplicar periodos ya existentes)."""
    with connect() as c:
        for r in seed.generar_consumo_demo():
            c.execute(
                "INSERT OR IGNORE INTO consumo (periodo, consumo_kwh, valor_factura,"
                " num_electrodomesticos, horas_uso, es_simulado) VALUES (?,?,?,?,?,1)",
                (r["periodo"], r["consumo_kwh"], r["valor_factura"],
                 r["num_electrodomesticos"], r["horas_uso"]))
        hay = c.execute("SELECT COUNT(*) FROM electrodomesticos WHERE es_simulado = 1").fetchone()[0]
        if not hay:
            for e in seed.generar_electrodomesticos_demo():
                c.execute(
                    "INSERT INTO electrodomesticos (nombre, potencia_w, horas_dia, dias_mes,"
                    " cantidad, es_simulado) VALUES (?,?,?,?,?,1)",
                    (e["nombre"], e["potencia_w"], e["horas_dia"], e["dias_mes"], e["cantidad"]))


# ----------------------------------------------------------------- consumo
def listar_consumo() -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query("SELECT * FROM consumo ORDER BY periodo", c)


def insertar_consumo(periodo, consumo_kwh, valor_factura, num_electrodomesticos, horas_uso,
                     es_simulado: int = 0) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO consumo (periodo, consumo_kwh, valor_factura, num_electrodomesticos,"
            " horas_uso, es_simulado) VALUES (?,?,?,?,?,?)",
            (periodo, float(consumo_kwh), float(valor_factura), int(num_electrodomesticos),
             float(horas_uso), int(es_simulado)))
        return int(cur.lastrowid)


def actualizar_consumo(id_, periodo, consumo_kwh, valor_factura, num_electrodomesticos, horas_uso) -> None:
    """Al editar un registro pasa a considerarse dato del usuario (es_simulado = 0)."""
    with connect() as c:
        c.execute(
            "UPDATE consumo SET periodo=?, consumo_kwh=?, valor_factura=?, num_electrodomesticos=?,"
            " horas_uso=?, es_simulado=0 WHERE id=?",
            (periodo, float(consumo_kwh), float(valor_factura), int(num_electrodomesticos),
             float(horas_uso), int(id_)))


def eliminar_consumo(id_) -> None:
    with connect() as c:
        c.execute("DELETE FROM consumo WHERE id=?", (int(id_),))


def existe_periodo(periodo: str, excluir_id: int | None = None) -> bool:
    with connect() as c:
        row = c.execute("SELECT id FROM consumo WHERE periodo=?", (periodo,)).fetchone()
    return row is not None and (excluir_id is None or int(row["id"]) != int(excluir_id))


def upsert_consumo(registros: list[dict]) -> int:
    with connect() as c:
        for r in registros:
            c.execute(
                "INSERT INTO consumo (periodo, consumo_kwh, valor_factura, num_electrodomesticos,"
                " horas_uso, es_simulado) VALUES (?,?,?,?,?,0)"
                " ON CONFLICT(periodo) DO UPDATE SET consumo_kwh=excluded.consumo_kwh,"
                " valor_factura=excluded.valor_factura,"
                " num_electrodomesticos=excluded.num_electrodomesticos,"
                " horas_uso=excluded.horas_uso, es_simulado=0",
                (r["periodo"], r["consumo_kwh"], r["valor_factura"],
                 r["num_electrodomesticos"], r["horas_uso"]))
    return len(registros)


# ------------------------------------------------------- electrodomésticos
def listar_electrodomesticos() -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query("SELECT * FROM electrodomesticos ORDER BY id", c)


def insertar_electrodomestico(nombre, potencia_w, horas_dia, dias_mes, cantidad, es_simulado: int = 0) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO electrodomesticos (nombre, potencia_w, horas_dia, dias_mes, cantidad,"
            " es_simulado) VALUES (?,?,?,?,?,?)",
            (nombre, float(potencia_w), float(horas_dia), int(dias_mes), int(cantidad), int(es_simulado)))
        return int(cur.lastrowid)


def actualizar_electrodomestico(id_, nombre, potencia_w, horas_dia, dias_mes, cantidad) -> None:
    with connect() as c:
        c.execute(
            "UPDATE electrodomesticos SET nombre=?, potencia_w=?, horas_dia=?, dias_mes=?,"
            " cantidad=?, es_simulado=0 WHERE id=?",
            (nombre, float(potencia_w), float(horas_dia), int(dias_mes), int(cantidad), int(id_)))


def eliminar_electrodomestico(id_) -> None:
    with connect() as c:
        c.execute("DELETE FROM electrodomesticos WHERE id=?", (int(id_),))


# ------------------------------------------------------------ mantenimiento
def borrar_simulados() -> None:
    with connect() as c:
        c.execute("DELETE FROM consumo WHERE es_simulado=1")
        c.execute("DELETE FROM electrodomesticos WHERE es_simulado=1")


def borrar_todo() -> None:
    with connect() as c:
        c.execute("DELETE FROM consumo")
        c.execute("DELETE FROM electrodomesticos")


def restablecer_demo() -> None:
    borrar_todo()
    cargar_demo()
    with connect() as c:
        for k, v in config.DEFAULT_SETTINGS.items():
            c.execute("INSERT OR REPLACE INTO configuracion (clave, valor) VALUES (?, ?)", (k, str(v)))


# ------------------------------------------------------------ configuración
def get_settings() -> dict:
    with connect() as c:
        rows = c.execute("SELECT clave, valor FROM configuracion").fetchall()
    out = dict(config.DEFAULT_SETTINGS)
    for r in rows:
        try:
            out[r["clave"]] = float(r["valor"])
        except ValueError:
            pass
    return out


def set_setting(clave: str, valor: float) -> None:
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO configuracion (clave, valor) VALUES (?, ?)", (clave, str(float(valor))))
