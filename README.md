# ⚡ ENERGY PREDICT

**Sistema Inteligente de Predicción del Consumo Eléctrico Residencial**

Aplicación web (Python + Streamlit) que analiza el consumo eléctrico de una vivienda, predice el consumo del próximo mes y calcula probabilidades asociadas al gasto energético. Proyecto académico para la asignatura **Probabilidad y Estadística** (Ingeniería de Sistemas).

Conceptos aplicados: estadística descriptiva, variables aleatorias, varianza y desviación estándar, distribuciones de probabilidad, regresión lineal y simulación de Monte Carlo.

---

## 1. Instalación y ejecución

Requisitos: Python 3.10 o superior.

```bash
# 1. Entrar a la carpeta del proyecto
cd energy_predict

# 2. (Recomendado) crear un entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

Se abre en `http://localhost:8501`. En el primer inicio se crea la base de datos `data/energy_predict.db` con datos de demostración, de modo que el dashboard nunca aparece vacío.

### Ejecutar las pruebas

```bash
pytest -v
```

- `tests/test_core.py`: fórmulas, base de datos, regresión, probabilidad, Monte Carlo y recomendaciones.
- `tests/test_app_streamlit.py`: ejecuta la aplicación real con `streamlit.testing` y navega por las 11 pantallas (requiere Streamlit y Plotly instalados).
- `tests/test_ui_smoke.py`: alternativa que ejecuta todas las vistas con dobles de prueba cuando Streamlit no está instalado (se omite sola si lo está).

---

## 2. Pensada para Popayán y para cualquier persona

La aplicación abre en una pantalla de **Inicio** escrita en lenguaje sencillo, con letra grande (hay una casilla «Letra más grande» en el menú) y tres pasos:

1. **Mis electrodomésticos**: se elige el aparato de una lista (nevera, ducha eléctrica, televisor…), se indica cuántas horas al día está encendido y el programa calcula cuánto gasta.
2. **Mi consumo mensual**: se anota el mes. Los campos ya vienen llenos con lo calculado a partir de los aparatos; si se tiene el recibo, se cambian por sus valores.
3. **Mi factura** y **Consejos de ahorro**: el resultado, sin hacer cuentas.

Cada pantalla empieza con un recuadro «En palabras sencillas». Los módulos estadísticos están en el mismo menú con el prefijo «Análisis ·».

### Reglas reales de Popayán que usa el cálculo de la factura

| Dato | Valor | Fuente |
|---|---|---|
| Operador | Compañía Energética de Occidente (CEO) | — |
| Tarifa de referencia | 1.048 $/kWh (costo unitario de CEO Cauca, 2025) | Superintendencia de Servicios Públicos |
| Consumo de subsistencia | 130 kWh/mes (municipios a 1.000 m s. n. m. o más) | Resolución UPME 0355 de 2004 |
| Subsidio sobre el consumo de subsistencia | Hasta 60 % estrato 1, 50 % estrato 2, 15 % estrato 3 | Ley 142 de 1994 y modificatorias |
| Contribución | 20 % sobre todo el consumo en estratos 5 y 6 | Ley 142 de 1994 |

```
factura = consumo × tarifa × (1 + contribución) − subsidio × tarifa × min(consumo, 130) + cargos fijos
```

La tarifa cambia cada mes: debe reemplazarse en **Configuración** por la que aparece en el recibo. Las potencias de la lista de aparatos son valores típicos de referencia; la potencia real está en la placa de cada equipo.

### Datos de demostración (simulados)

> **Los meses y aparatos iniciales son un ejemplo sintético. No son mediciones reales de ninguna vivienda de Popayán.**

No se encontró un registro público del consumo mes a mes de hogares de Popayán, así que el ejemplo se genera en `core/seed.py` con semilla fija y un modelo conocido, para una vivienda de estrato 3, clima templado, sin aire acondicionado y con ducha eléctrica:

```
consumo = 40 + 14·horas_uso + 4·n_electrodomésticos + estacionalidad + ruido,   ruido ~ N(0, 6²)
factura = factura de estrato 3 (reglas de Popayán) con la tarifa del mes · (1 + e),   e ~ N(0, 0.015²)
```

Son 36 meses (nov-2023 a oct-2026, promedio cercano a 150 kWh) y 13 electrodomésticos. Cada registro simulado lleva la marca `es_simulado = 1` y la aplicación lo indica en el menú y en las tablas. Para usar datos propios: anótelos, impórtelos desde CSV o elimine los simulados en **Configuración**.

Si ya había ejecutado la versión anterior, la base se actualiza sola a esta versión siempre que solo contenga datos de ejemplo; los datos que usted haya ingresado se respetan.

---

## 3. Pantallas

| Pantalla | Qué hace |
|---|---|
| **Inicio** | Resultado principal en lenguaje sencillo, semáforo de riesgo y guía de tres pasos |
| **1. Mis electrodomésticos** | Lista de aparatos para elegir, horas de uso, energía y participación de cada uno |
| **2. Mi consumo mensual** | Agregar, cambiar y borrar meses; valores sugeridos desde los aparatos; importar y exportar CSV |
| **3. Mi factura** | Escenarios bajo, medio y alto por estrato, con el desglose del subsidio o la contribución |
| **4. Consejos de ahorro** | Reglas calculadas sobre los datos: equipos de mayor consumo, incrementos inusuales, tendencia y ahorro |
| **Análisis · Resumen general** | Dashboard: consumo actual, promedio, proyección, factura, variación %, probabilidades, histórico y real vs. predicho |
| **Análisis · Estadística** | Media, mediana, moda, varianza, desviación estándar, máximo, mínimo, CV, histograma y tabla de frecuencias |
| **Análisis · Regresión lineal** | Predictores, ecuación, R², MAE, RMSE, dispersión con recta, residuos, supuestos y predicción |
| **Análisis · Probabilidad** | P(consumo > límite), P(factura > presupuesto), P(aumento respecto al mes anterior) con tres enfoques |
| **Análisis · Monte Carlo** | 10.000 a 100.000 simulaciones del consumo y la factura; percentiles, intervalos y probabilidades |
| **Configuración** | Estrato, tarifa, cargo fijo, límite, presupuesto y administración de los datos |

---

## 4. Metodología estadística

### 4.1 Estadística descriptiva (`core/descriptive.py`)
- Varianza y desviación estándar **muestrales** (divisor n − 1), porque los meses registrados son una muestra.
- Coeficiente de variación = s / x̄ × 100.
- La moda se calcula sobre valores redondeados y solo se informa si algún valor se repite; en una variable continua se informa además la **clase modal**.
- Número de clases por la regla de Sturges: k = 1 + log₂(n).

### 4.2 Regresión lineal (`core/regression.py`)
- Mínimos cuadrados ordinarios: **b = (XᵀX)⁻¹Xᵀy**, con errores estándar, estadísticos t y valores p.
- Predictores disponibles: horas promedio de uso diario, cantidad de electrodomésticos y tendencia temporal. La relación lineal es físicamente razonable porque energía = potencia × tiempo.
- **Evaluación sin reutilizar los datos de entrenamiento:**
  1. *Partición cronológica*: se entrena con los primeros meses y se evalúa con los más recientes. No se barajan los datos porque son una serie temporal.
  2. *Validación walk-forward*: cada mes se predice con un modelo entrenado solo con los meses anteriores.
  3. Las métricas se comparan con un modelo de referencia que predice siempre la media.
- **Supuestos verificados:** normalidad de residuos (Shapiro-Wilk), independencia (Durbin-Watson), homocedasticidad (Breusch-Pagan) y multicolinealidad (VIF).
- Intervalo de predicción: ŷ ± t(n−p) · s · √(1 + x₀ᵀ(XᵀX)⁻¹x₀).
- Con pocos registros el sistema reduce predictores y, en último caso, usa el modelo de solo intercepto (la media histórica).

### 4.3 Probabilidad (`core/probability.py`)
La regresión estima el valor **esperado** del consumo; la probabilidad mide la **incertidumbre** alrededor de él. Se muestran tres cálculos de P(X > límite):

| Enfoque | Modelo | Se refiere a |
|---|---|---|
| Empírico | Proporción de meses que superaron el límite, con intervalo de Wilson | Un mes cualquiera |
| Paramétrico | Normal, Lognormal o Gamma ajustadas por máxima verosimilitud; se elige por AIC y se contrasta con Kolmogorov-Smirnov | Un mes cualquiera |
| Predictivo | t de Student centrada en la predicción de la regresión | El próximo mes |

**No se asume normalidad sin comprobarla:** se aplica Shapiro-Wilk y, si se rechaza, la aplicación lo advierte.

P(factura > B) se reduce a P(X > consumo equivalente a B), con tarifa determinista y las reglas del estrato.

### 4.4 Monte Carlo (`core/montecarlo.py`)
Tres modelos generadores:
1. **Regresión + incertidumbre** (recomendado): coeficientes ~ Normal multivariada(b̂, Cov(b̂)); horas de uso ~ Normal(media, σ histórica) truncada a [0, 24]; ruido por bootstrap de residuos o Normal(0, s²).
2. **Distribución ajustada** al histórico (la de menor AIC).
3. **Bootstrap** del histórico (no paramétrico).

La tarifa se simula como Normal(tarifa, (cv · tarifa)²). Cada probabilidad incluye su error estándar de Monte Carlo √(p(1−p)/N). La semilla es configurable para reproducir resultados.

### 4.5 Limitaciones
- Con 36 meses las pruebas de normalidad y de bondad de ajuste tienen poca potencia: no rechazar una hipótesis no la demuestra.
- El valor p de Kolmogorov-Smirnov es optimista porque los parámetros se estiman con los mismos datos.
- El conjunto de prueba es pequeño, de modo que R², MAE y RMSE fuera de muestra tienen alta incertidumbre.
- Las pendientes de la regresión son asociaciones estadísticas, no efectos causales.
- El modelo no incluye temperatura, número de habitantes ni cambios de equipos.
- La factura estimada aplica tarifa, subsidio o contribución del estrato y cargos fijos. El recibo real puede diferir por alumbrado público, aseo, intereses y ajustes tarifarios del mes.
- El consumo por electrodoméstico es teórico (potencia nominal × horas declaradas), no una medición.

---

## 5. Arquitectura

```
energy_predict/
├── app.py                     # Punto de entrada: navegación y manejo de errores
├── requirements.txt
├── .streamlit/config.toml     # Tema oscuro
├── core/                      # Sin dependencias de la interfaz
│   ├── config.py              # Rutas, valores por defecto, formato
│   ├── database.py            # Base de datos SQLite (esquema y CRUD)
│   ├── seed.py                # Datos sintéticos de demostración
│   ├── data_processing.py     # Procesamiento de datos, importación/exportación CSV
│   ├── business.py            # Lógica de negocio: validaciones, electrodomésticos, factura
│   ├── descriptive.py         # Estadística descriptiva
│   ├── regression.py          # Modelos estadísticos: regresión, validación, diagnósticos
│   ├── probability.py         # Modelos probabilísticos
│   ├── montecarlo.py          # Simulaciones
│   ├── recommendations.py     # Recomendaciones basadas en reglas
│   └── analysis.py            # Indicadores del dashboard
├── ui/
│   ├── theme.py               # Identidad visual: paleta, CSS, iconos, tarjetas
│   ├── charts.py              # Visualizaciones Plotly
│   └── views/                 # Una vista por pantalla (inicio.py es la portada sencilla)
├── tests/
└── data/                      # Base de datos (se crea al iniciar)
```

La variable de entorno `ENERGY_PREDICT_DB` permite usar otra ruta para la base de datos.

### Esquema de la base de datos

- `consumo(id, periodo UNIQUE, consumo_kwh, valor_factura, num_electrodomesticos, horas_uso, es_simulado)`
- `electrodomesticos(id, nombre, potencia_w, horas_dia, dias_mes, cantidad, es_simulado)`
- `configuracion(clave, valor)`

---

## 6. Fórmulas principales

| Concepto | Fórmula |
|---|---|
| Consumo de un equipo | (W × horas/día × días/mes × cantidad) / 1000 |
| Media | x̄ = Σxᵢ / n |
| Varianza muestral | s² = Σ(xᵢ − x̄)² / (n − 1) |
| Coeficiente de variación | CV = s / x̄ × 100 |
| Coeficientes de regresión | b = (XᵀX)⁻¹Xᵀy |
| R² | 1 − SCE / SCT |
| MAE | Σ\|yᵢ − ŷᵢ\| / n |
| RMSE | √(Σ(yᵢ − ŷᵢ)² / n) |
| Probabilidad de superar un límite | P(X > L) = 1 − F(L) |
| Error estándar de Monte Carlo | √(p(1 − p) / N) |
| Factura estimada | consumo × tarifa × (1 + contribución) − subsidio × tarifa × min(consumo, 130) + cargo fijo |
