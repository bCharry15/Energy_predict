"""Pruebas unitarias de la lógica: fórmulas verificadas contra valores conocidos y contra
implementaciones de referencia (NumPy, SciPy, scikit-learn)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from core import (analysis, business, config, data_processing, descriptive, montecarlo, probability,
                  recommendations, regression, seed)


# ------------------------------------------------------------ base de datos
def test_demo_se_carga_y_esta_marcada_como_simulada(db, df):
    assert len(df) == 36 and df["es_simulado"].all()
    assert df["periodo"].is_monotonic_increasing and df["periodo"].is_unique
    eq = db.listar_electrodomesticos()
    assert len(eq) == len(seed.ELECTRODOMESTICOS_DEMO) and eq["es_simulado"].all()


def test_init_db_es_idempotente(db):
    db.init_db(); db.init_db()
    assert len(db.listar_consumo()) == 36


def test_datos_demo_reproducibles():
    assert seed.generar_consumo_demo() == seed.generar_consumo_demo()


def test_crud_consumo_y_persistencia(db):
    rid = db.insertar_consumo("2026-11", 210.0, 190000, 12, 6.5)
    assert db.existe_periodo("2026-11") and not db.existe_periodo("2026-11", excluir_id=rid)
    db.actualizar_consumo(rid, "2026-11", 222.2, 200000, 12, 7.0)
    fila = db.listar_consumo().set_index("id").loc[rid]
    assert fila["consumo_kwh"] == 222.2 and fila["es_simulado"] == 0
    db.eliminar_consumo(rid)
    assert not db.existe_periodo("2026-11")


def test_periodo_duplicado_rechazado_por_la_base(db):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.insertar_consumo("2026-10", 100, 1, 1, 1)


def test_borrar_simulados_conserva_datos_del_usuario(db):
    db.insertar_consumo("2026-11", 210.0, 190000, 12, 6.5)
    db.borrar_simulados()
    assert db.listar_consumo()["periodo"].tolist() == ["2026-11"]
    assert db.listar_electrodomesticos().empty


def test_configuracion_persistente(db):
    db.set_setting("tarifa_kwh", 1234.5)
    assert db.get_settings()["tarifa_kwh"] == 1234.5


# ------------------------------------------------------------------ negocio
def test_formula_consumo_electrodomestico():
    # (1000 W × 2 h × 30 días × 1) / 1000 = 60 kWh
    assert business.consumo_electrodomestico(1000, 2, 30, 1) == 60
    assert business.consumo_electrodomestico(9, 5, 30, 10) == pytest.approx(13.5)


def test_participacion_suma_100(db):
    t = business.tabla_electrodomesticos(db.listar_electrodomesticos())
    assert t["participacion_pct"].sum() == pytest.approx(100)
    assert t["consumo_kwh"].is_monotonic_decreasing


@pytest.mark.parametrize("args", [("2026-13", 100, 1, 1, 1), ("2026-1", 100, 1, 1, 1), ("2026-01", 0, 1, 1, 1),
                                  ("2026-01", 100, -1, 1, 1), ("2026-01", 100, 1, 1.5, 1), ("2026-01", 100, 1, 1, 25)])
def test_validacion_consumo_rechaza(args):
    with pytest.raises(ValueError):
        business.validar_consumo(*args)


@pytest.mark.parametrize("args", [("", 100, 1, 30, 1), ("TV", 0, 1, 30, 1), ("TV", 100, 25, 30, 1),
                                  ("TV", 100, 1, 32, 1), ("TV", 100, 1, 30, 0)])
def test_validacion_electrodomestico_rechaza(args):
    with pytest.raises(ValueError):
        business.validar_electrodomestico(*args)


def test_factura_y_presupuesto_son_inversas():
    f = business.factura(200, 900, 5000)                     # estrato 4 por defecto: tarifa plena
    assert f == 185000
    assert business.kwh_para_presupuesto(f, 900, 5000) == pytest.approx(200)


def test_factura_popayan_por_estrato():
    """Subsidio solo sobre los primeros 130 kWh (Popayán, > 1.000 m); contribución del 20 % en estratos 5 y 6."""
    cu = 1000.0
    assert business.factura(150, cu, 0, 4) == 150000
    assert business.factura(150, cu, 0, 1) == pytest.approx(150000 - 0.60 * 130 * cu)     # 72.000
    assert business.factura(150, cu, 0, 2) == pytest.approx(150000 - 0.50 * 130 * cu)     # 85.000
    assert business.factura(150, cu, 0, 3) == pytest.approx(150000 - 0.15 * 130 * cu)     # 130.500
    assert business.factura(100, cu, 0, 2) == pytest.approx(50000)                         # todo subsidiado
    assert business.factura(150, cu, 0, 5) == business.factura(150, cu, 0, 6) == pytest.approx(180000)
    assert business.subsidio_aplicado(150, cu, 3) == pytest.approx(19500)
    assert business.subsidio_aplicado(150, cu, 6) == pytest.approx(-30000)
    # vectorizada (Monte Carlo) y creciente con el consumo
    v = business.factura(np.array([100.0, 130.0, 150.0]), cu, 0, 3)
    assert v == pytest.approx([85000, 110500, 130500]) and (np.diff(v) > 0).all()


@pytest.mark.parametrize("estrato", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("consumo", [40.0, 130.0, 131.0, 220.0])
def test_inversa_de_la_factura_por_estrato(estrato, consumo):
    f = business.factura(consumo, 1048, 3000, estrato)
    assert business.kwh_para_presupuesto(f, 1048, 3000, estrato) == pytest.approx(consumo)
    assert business.kwh_para_presupuesto(2000, 1048, 3000, estrato) == 0.0


def test_sugerencia_de_registro_y_catalogo(db):
    sug = business.sugerencia_registro(db.listar_electrodomesticos())
    t = business.tabla_electrodomesticos(db.listar_electrodomesticos())
    assert sug["consumo_kwh"] == pytest.approx(t["consumo_kwh"].sum(), abs=0.05) and sug["num_electrodomesticos"] == 13
    assert business.sugerencia_registro(db.listar_electrodomesticos().iloc[0:0]) is None
    for nombre, pot, horas, dias in business.CATALOGO:
        business.validar_electrodomestico(nombre, pot, horas, dias, 1)
    assert len({c[0] for c in business.CATALOGO}) == len(business.CATALOGO)


def test_facturas_demo_siguen_las_reglas_de_estrato_3(df):
    """Las facturas simuladas equivalen a un precio medio entre 85 % y 100 % de la tarifa plena."""
    assert df["tarifa_implicita"].between(0.80 * 930, 1.03 * config.TARIFA_REFERENCIA).all()
    assert 140 < df["consumo_kwh"].mean() < 160


def test_base_antigua_se_actualiza_a_popayan(db):
    with db.connect() as c:                     # simula una base creada con la versión anterior
        c.execute("DELETE FROM configuracion WHERE clave IN ('version_demo', 'estrato')")
        c.execute("UPDATE configuracion SET valor='900.0' WHERE clave='tarifa_kwh'")
        c.execute("UPDATE consumo SET consumo_kwh = 999")
    db.init_db()
    assert db.get_settings()["tarifa_kwh"] == config.TARIFA_REFERENCIA and db.listar_consumo()["consumo_kwh"].max() < 300
    with db.connect() as c:                     # con datos propios del usuario NO se toca nada
        c.execute("DELETE FROM configuracion WHERE clave = 'version_demo'")
    db.insertar_consumo("2026-11", 151.0, 140000, 12, 5.0)
    db.set_setting("tarifa_kwh", 777.0)
    db.init_db()
    assert db.get_settings()["tarifa_kwh"] == 777.0 and len(db.listar_consumo()) == 37


def test_csv_valida_filas():
    csv = (b"periodo,consumo_kwh,valor_factura,num_electrodomesticos,horas_uso\n"
           b"2027-01,180.5,160000,12,5.5\n2027-01,190,1,1,1\n2027-03,abc,1,1,1\n2027-04,100,1,1,30\n")
    ok, err = data_processing.leer_csv(csv)
    assert [r["periodo"] for r in ok] == ["2027-01"] and len(err) == 3
    assert data_processing.leer_csv(b"a,b\n1,2\n")[1][0].startswith("Faltan columnas")


def test_csv_ida_y_vuelta(df):
    ok, err = data_processing.leer_csv(data_processing.exportar_csv(df))
    assert not err and len(ok) == len(df)


# -------------------------------------------------------------- descriptiva
def test_descriptiva_valores_conocidos():
    r = descriptive.resumen([2, 4, 4, 4, 5, 5, 7, 9])
    assert r["media"] == 5 and r["mediana"] == 4.5 and r["moda"] == [4.0]
    assert r["varianza"] == pytest.approx(32 / 7)          # muestral (n - 1)
    assert r["desviacion"] == pytest.approx(np.sqrt(32 / 7))
    assert r["cv"] == pytest.approx(np.sqrt(32 / 7) / 5 * 100)
    assert (r["minimo"], r["maximo"], r["rango"]) == (2, 9, 7)


def test_descriptiva_coincide_con_pandas(df):
    x = df["consumo_kwh"]
    r = descriptive.resumen(x)
    assert r["varianza"] == pytest.approx(x.var()) and r["desviacion"] == pytest.approx(x.std())
    assert r["asimetria"] == pytest.approx(x.skew()) and r["curtosis"] == pytest.approx(x.kurt())


def test_moda_no_aplica_sin_repetidos():
    assert descriptive.moda([1.0, 2.0, 3.0]) == []
    assert descriptive.moda([1, 1, 2, 2, 3]) == [1.0, 2.0]


def test_tabla_frecuencias_consistente(df):
    t = descriptive.tabla_frecuencias(df["consumo_kwh"])
    assert len(t) == descriptive.n_clases_sturges(36) == 7
    assert t["Frecuencia (fi)"].sum() == 36 and t["Frec. acumulada (Fi)"].iloc[-1] == 36
    assert t["Frec. rel. acumulada (%)"].iloc[-1] == pytest.approx(100)


# ---------------------------------------------------------------- regresión
def test_ols_recupera_recta_exacta():
    x = np.arange(10.0)
    r = regression.fit_ols(x, 3 + 2 * x, ["x"])
    assert r.coef == pytest.approx([3, 2]) and r.r2 == pytest.approx(1)


def test_ols_coincide_con_sklearn_y_scipy(df):
    feats = ["horas_uso", "num_electrodomesticos"]
    r = regression.ajustar(df, feats)
    sk = LinearRegression().fit(df[feats], df["consumo_kwh"])
    assert r.coef[0] == pytest.approx(sk.intercept_) and r.coef[1:] == pytest.approx(sk.coef_)
    assert r.r2 == pytest.approx(sk.score(df[feats], df["consumo_kwh"]))
    simple = regression.ajustar(df, ["horas_uso"])
    lr = stats.linregress(df["horas_uso"], df["consumo_kwh"])
    assert simple.coef == pytest.approx([lr.intercept, lr.slope])
    assert simple.se[1] == pytest.approx(lr.stderr) and simple.se[0] == pytest.approx(lr.intercept_stderr)
    assert simple.p_values[1] == pytest.approx(lr.pvalue)
    assert simple.r2 == pytest.approx(lr.rvalue ** 2)


def test_intervalo_de_prediccion_formula_clasica(df):
    """Regresión simple: se = s·sqrt(1 + 1/n + (x0 - x̄)² / Sxx)."""
    r = regression.ajustar(df, ["horas_uso"])
    x = df["horas_uso"].to_numpy()
    x0 = 7.0
    esperado = r.sigma * np.sqrt(1 + 1 / len(x) + (x0 - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    y, lo, hi, se = r.intervalo_prediccion([[x0]], 0.90)
    assert se[0] == pytest.approx(esperado)
    assert hi[0] - y[0] == pytest.approx(stats.t.ppf(0.95, len(x) - 2) * esperado)


def test_modelo_solo_intercepto_es_la_media(df):
    r = regression.ajustar(df, [])
    y = df["consumo_kwh"]
    assert r.coef[0] == pytest.approx(y.mean()) and r.sigma == pytest.approx(y.std())
    assert r.se_prediccion(np.empty((1, 0)))[0] == pytest.approx(y.std() * np.sqrt(1 + 1 / len(y)))


def test_holdout_es_cronologico_y_sin_fuga(df):
    feats = ["horas_uso", "num_electrodomesticos"]
    h = regression.evaluar_holdout(df, feats, 0.25)
    assert h["n_train"] + h["n_test"] == len(df) and h["n_test"] == 9
    assert h["idx_train"].max() < h["idx_test"].min()
    # Cambiar los valores de prueba no debe alterar el modelo entrenado.
    df2 = df.copy()
    df2.loc[h["idx_test"], "consumo_kwh"] = 999.0
    h2 = regression.evaluar_holdout(df2, feats, 0.25)
    assert h2["pred_test"] == pytest.approx(h["pred_test"])
    assert h2["train"]["r2"] == pytest.approx(h["train"]["r2"])


def test_walk_forward_no_usa_el_futuro(df):
    feats = ["horas_uso"]
    wf = regression.walk_forward(df, feats)
    df2 = df.copy()
    df2.loc[df2.index[-1], "consumo_kwh"] = 999.0
    wf2 = regression.walk_forward(df2, feats)
    assert wf2["predicho"].to_numpy() == pytest.approx(wf["predicho"].to_numpy())
    assert len(wf) == len(df) - 12


def test_modelo_supera_a_la_media_en_datos_demo(df):
    h = regression.evaluar_holdout(df, config.DEFAULT_FEATURES)
    assert h["test"]["rmse"] < h["baseline"]["rmse"]


def test_recupera_parametros_del_generador(df):
    """El generador sintético usa 14 kWh/hora y 4 kWh/equipo: el IC 95% debe contenerlos."""
    r = regression.ajustar(df, config.DEFAULT_FEATURES)
    t = stats.t.ppf(0.975, r.dof)
    for real, b, se in zip((14, 4), r.coef[1:], r.se[1:]):
        assert b - t * se <= real <= b + t * se


def test_diagnosticos_y_estadisticos(df):
    r = regression.ajustar(df, config.DEFAULT_FEATURES)
    X = df[config.DEFAULT_FEATURES].to_numpy(float)
    d = {x["Prueba"]: x for x in regression.diagnosticos(r, X)}
    assert set(d) == {"Shapiro-Wilk", "Durbin-Watson", "Breusch-Pagan", "VIF máximo"}
    assert 0 < d["Durbin-Watson"]["Estadístico"] < 4
    assert regression.durbin_watson([1, -1, 1, -1]) == pytest.approx(3.0)
    # VIF con dos predictores: 1 / (1 - r²)
    rho = np.corrcoef(X.T)[0, 1]
    assert regression.vif(X) == pytest.approx([1 / (1 - rho ** 2)] * 2)


def test_predictores_colineales_o_constantes_se_rechazan(df):
    d = df.copy()
    d["horas_uso"] = 5.0
    with pytest.raises(ValueError):
        regression.ajustar(d, ["horas_uso"])
    assert regression.features_utilizables(d) == ["num_electrodomesticos"]
    with pytest.raises(ValueError):
        regression.fit_ols(np.arange(3.0), [1, 2, 3], ["x"])


def test_pronostico_degrada_con_pocos_datos(df):
    p = regression.pronostico(df.head(4))
    assert p["features"] == [] and p["yhat"] == pytest.approx(df["consumo_kwh"].head(4).mean())
    p = regression.pronostico(df)
    assert p["periodo"] == "2026-11" and p["lo"] < p["yhat"] < p["hi"]


# -------------------------------------------------------------- probabilidad
def test_probabilidad_normal_conocida():
    assert probability.prob_parametrica(stats.norm(200, 20), 200) == pytest.approx(0.5)
    assert probability.prob_parametrica(stats.norm(200, 20), 239.2) == pytest.approx(0.025, abs=1e-3)
    assert probability.prob_predictiva(200, 10, 30, 200) == pytest.approx(0.5)
    assert probability.prob_predictiva(200, 10, 10**6, 216.45) == pytest.approx(0.05, abs=1e-3)


def test_probabilidad_empirica_y_wilson():
    r = probability.prob_empirica([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 8)
    assert (r["k"], r["n"], r["p"]) == (2, 10, 0.2)
    lo, hi = probability.wilson(2, 10)
    assert (lo, hi) == pytest.approx((0.0567, 0.5098), abs=1e-3)   # valores de referencia publicados
    assert probability.prob_aumento_empirica([1, 2, 1, 3])["p"] == pytest.approx(2 / 3)


def test_ajuste_identifica_la_distribucion_generadora():
    rng = np.random.default_rng(0)
    assert probability.ajustar_distribuciones(rng.normal(200, 15, 2000))[0]["clave"] == "norm"
    assert probability.ajustar_distribuciones(rng.lognormal(5, 0.5, 2000))[0]["clave"] == "lognorm"
    assert probability.ajustar_distribuciones(rng.gamma(3, 50, 2000))[0]["clave"] == "gamma"


def test_ajuste_mle_normal_usa_media_y_desviacion(df):
    x = df["consumo_kwh"].to_numpy()
    a = {r["clave"]: r for r in probability.ajustar_distribuciones(x)}
    assert a["norm"]["params"] == pytest.approx((x.mean(), x.std(ddof=0)))
    for r in a.values():
        assert 0 <= probability.prob_parametrica(r["dist"], 160) <= 1
        assert r["dist"].mean() == pytest.approx(x.mean(), rel=0.01)


def test_shapiro_detecta_no_normalidad():
    rng = np.random.default_rng(1)
    assert probability.shapiro(rng.normal(0, 1, 200))[1] > 0.05
    assert probability.shapiro(rng.exponential(1, 200))[1] < 0.05


# --------------------------------------------------------------- Monte Carlo
def test_montecarlo_reproducible_y_tamano(df):
    p = regression.pronostico(df)
    a = montecarlo.simular_regresion(p["modelo"], p["x"], {"horas_uso": 0.6}, 10000, np.random.default_rng(7))
    b = montecarlo.simular_regresion(p["modelo"], p["x"], {"horas_uso": 0.6}, 10000, np.random.default_rng(7))
    assert len(a) == 10000 and np.array_equal(a, b) and (a >= 0).all()


def test_montecarlo_converge_a_la_solucion_analitica(df):
    """Sin incertidumbre de entradas y con ruido normal, la simulación debe reproducir la
    distribución predictiva analítica de la regresión (media ŷ, desviación ≈ se de predicción)."""
    p = regression.pronostico(df)
    sims = montecarlo.simular_regresion(p["modelo"], p["x"], {}, 200000, np.random.default_rng(3), ruido="normal")
    r = montecarlo.resumen(sims)
    assert r["media"] == pytest.approx(p["yhat"], abs=0.15)
    assert r["desviacion"] == pytest.approx(p["se"], rel=0.02)
    exacta = stats.norm.sf(160, p["yhat"], p["se"])
    est = montecarlo.prob_superar(sims, 160)
    assert abs(est["p"] - exacta) < 4 * est["ee"] + 0.002


def test_montecarlo_distribucion_y_bootstrap(df):
    x = df["consumo_kwh"].to_numpy()
    rng = np.random.default_rng(5)
    s1 = montecarlo.simular_distribucion(stats.norm(200, 15), 100000, rng)
    assert s1.mean() == pytest.approx(200, abs=0.3) and s1.std() == pytest.approx(15, rel=0.02)
    assert montecarlo.prob_superar(s1, 200)["p"] == pytest.approx(0.5, abs=0.01)
    s2 = montecarlo.simular_bootstrap(x, 50000, rng)
    assert set(np.unique(s2)) <= set(x) and s2.mean() == pytest.approx(x.mean(), abs=0.5)


def test_montecarlo_resumen_y_error_estandar():
    sims = np.arange(1, 10001, dtype=float)
    r = montecarlo.resumen(sims)
    assert r["percentiles"][50] == pytest.approx(5000.5) and r["n"] == 10000
    assert montecarlo.intervalo(sims, 0.90) == pytest.approx((500.95, 9500.05))
    p = montecarlo.prob_superar(sims, 7500)
    assert p["p"] == 0.25 and p["ee"] == pytest.approx(np.sqrt(0.25 * 0.75 / 10000))
    idx, med = montecarlo.media_acumulada(sims)
    assert med[-1] == pytest.approx(sims.mean())


def test_tarifa_simulada(df):
    rng = np.random.default_rng(2)
    assert (montecarlo.simular_tarifa(900, 0, 100, rng) == 900).all()
    t = montecarlo.simular_tarifa(900, 0.03, 100000, rng)
    assert t.mean() == pytest.approx(900, abs=1) and t.std() == pytest.approx(27, rel=0.03) and (t > 0).all()


# ----------------------------------------------------------- recomendaciones
def _recs(df, db, **cambios):
    s = {**db.get_settings(), **cambios}
    r = analysis.resumen_general(df, s)
    return recommendations.generar(df, db.listar_electrodomesticos(), s, r["pronostico"], r["p_limite"])


def test_recomendaciones_dependen_de_los_datos(db, df):
    base = _recs(df, db)
    assert base == _recs(df, db)                                   # deterministas
    assert any("Nevera" in r["titulo"] for r in base)              # mayor consumidor real
    assert not any(r["titulo"].startswith("Incremento inusual") for r in base)

    d = df.copy()
    d.loc[d.index[-1], "consumo_kwh"] = 330.0                      # pico anómalo en el último mes
    con_pico = _recs(d, db)
    assert any(r["titulo"].startswith("Incremento inusual") and r["nivel"] == "alta" for r in con_pico)

    bajo = _recs(df, db, limite_kwh=100.0)                         # límite muy bajo -> alerta
    assert any(r["titulo"] == "Es probable superar el límite de consumo" for r in bajo)
    alto = _recs(df, db, limite_kwh=400.0)
    assert any(r["titulo"] == "Bajo riesgo de superar el límite" for r in alto)


def test_recomendacion_cambia_con_el_inventario(db, df):
    db.insertar_electrodomestico("Calentador", 4000, 3, 30, 1)
    recs = _recs(df, db)
    assert any(r["titulo"] == "Mayor consumidor: Calentador" for r in recs)
    kwh, dinero = recommendations.ahorro_potencial(recs)
    assert kwh > 0 and dinero == pytest.approx(kwh * db.get_settings()["tarifa_kwh"])


def test_recomendaciones_sin_datos(db):
    db.borrar_todo()
    assert recommendations.generar(data_processing.cargar_consumo(), db.listar_electrodomesticos(),
                                   db.get_settings()) == []


# ------------------------------------------------------------------ resumen
def test_resumen_general_coherente(db, df):
    s = db.get_settings()
    r = analysis.resumen_general(df, s)
    assert r["actual"] == df["consumo_kwh"].iloc[-1]
    assert r["variacion_pct"] == pytest.approx(df["variacion_pct"].iloc[-1])
    assert r["factura_estimada"] == pytest.approx(business.factura(r["pronostico"]["yhat"], s["tarifa_kwh"],
                                                                   s["cargo_fijo"], 3))
    assert 0 <= r["p_limite"] <= 1
    mas_alto = analysis.resumen_general(df, {**s, "limite_kwh": s["limite_kwh"] + 30})
    assert mas_alto["p_limite"] < r["p_limite"]                    # monotonía de la probabilidad
    assert config.money(150000) == "$150.000" and config.next_periodo("2026-12") == "2027-01"
