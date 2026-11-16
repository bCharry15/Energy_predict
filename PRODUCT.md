# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary: the course instructor and evaluators** of *Probabilidad y Estadística* (Ingeniería de Sistemas). They grade the project and judge it during the oral defense (*sustentación*). Their job is to verify that the statistical methods are correct, honestly reported and clearly explained, and that the application works end to end.

**Secondary: households in Popayán** — people without technical training, including older adults, who want to know how much electricity they will use next month, what the CEO bill will be, and how to save. They are the product's narrative and the reason the "plain language" screens exist; design decisions favor the evaluator when the two conflict, but never break the household path.

## Product Purpose

ENERGY PREDICT analyzes a home's monthly electricity consumption, predicts next month's kWh and bill, and computes probabilities tied to energy spending (exceeding a kWh limit or a budget, increasing over last month).

Success means a strong grade and a convincing *sustentación*: the evaluator can see, screen by screen, descriptive statistics, random variables, variance and standard deviation, probability distributions, linear regression and Monte Carlo simulation applied correctly to a believable local problem. Real-world adoption is not a goal of this project.

## Positioning

A statistics course project grounded in Popayán's real billing rules rather than a generic dataset: CEO (Compañía Energética de Occidente) as operator, 130 kWh subsistence consumption (UPME 0355/2004), stratum subsidies and contributions (Ley 142/1994). Every statistical result is translated into a household question ("¿cuánto pagaré?", "¿qué probabilidad hay de pasarme?") and every method states its assumptions, tests them and lists its limitations.

## Operating Context

- Runs locally with `streamlit run app.py` (http://localhost:8501); likely shown on a projector or laptop screen during the defense.
- Two navigation tiers in one sidebar: steps for the household ("Inicio", "1. Mis electrodomésticos", "2. Mi consumo mensual", "3. Mi factura", "4. Consejos de ahorro") and the evaluator-facing modules prefixed "Análisis ·" (Resumen general, Estadística, Regresión lineal, Probabilidad, Monte Carlo), plus "Configuración".
- Data lives in SQLite (`data/energy_predict.db`), seeded with 36 synthetic months and 13 appliances on first run; users can enter, edit, import/export CSV, or delete the simulated rows.
- Test suite (`pytest`) drives all 11 screens through `streamlit.testing`; results recorded in `RESULTADOS_PRUEBAS.txt`.

## Capabilities and Constraints

- Stack is fixed: Python + **Streamlit** + Plotly (pandas, numpy, scipy, scikit-learn). Design work happens within Streamlit's widgets plus injected CSS/HTML (`ui/theme.py`, `ui/charts.py`); no framework migration.
- Statistical capabilities: sample descriptive stats with Sturges classes; OLS regression with chronological split, walk-forward validation, mean baseline, Shapiro-Wilk / Durbin-Watson / Breusch-Pagan / VIF diagnostics and prediction intervals; empirical (Wilson), parametric (Normal/Lognormal/Gamma by AIC + KS) and predictive (Student t) probabilities; Monte Carlo with 10,000–100,000 runs, three generators, and MC standard errors; rule-based saving recommendations.
- Bill = consumo × tarifa × (1 + contribución) − subsidio × tarifa × min(consumo, 130) + cargos fijos. Tariff (default 1.048 $/kWh, CEO Cauca 2025) is user-editable because it changes monthly.
- Terminology in the UI is Spanish: kWh, estrato, recibo, tarifa, subsidio, contribución, «En palabras sencillas».

## Brand Commitments

- Name: **ENERGY PREDICT**, subtitle "Sistema Inteligente de Predicción del Consumo Eléctrico Residencial".
- All interface copy in Spanish. Household screens use plain language and open with an «En palabras sencillas» box; jargon is confined to the «Análisis ·» modules, where it is explained.
- Simulated data is always labeled as simulated (`es_simulado`), in the sidebar and in tables. It must never be presented as real measurements from any Popayán household.

## Evidence on Hand

- Real regulatory facts and sources: tariff, subsistence threshold, subsidy and contribution percentages (README §2).
- Synthetic demo data only: 36 months (nov-2023 to oct-2026, ~150 kWh mean) and 13 appliances generated in `core/seed.py` with a known model and fixed seed. There is **no real household consumption data**, no user testimonials and no field validation; future work must not imply otherwise.
- Documented methodology and limitations: README §4.
- Test evidence: `tests/`, `RESULTADOS_PRUEBAS.txt`.

## Product Principles

1. **Rigor visible on screen.** Every number shows how it was obtained: formula, assumptions, tests and uncertainty. The evaluator should never have to trust a result blindly.
2. **Honest uncertainty.** Predictions come with intervals and probabilities; limitations and failed assumptions are stated, not hidden.
3. **Statistics answer household questions.** Each method is framed by the everyday question it resolves, so the defense tells one coherent story.
4. **Simulated is never disguised as real.** Synthetic data stays labeled everywhere it appears.
5. **The simple path stays simple.** Steps 1–4 remain usable by someone with no statistics background, even though the evaluator is the primary audience.

## Accessibility & Inclusion

- Readable by older adults: optional larger text («Letra más grande»), strong contrast, and risk indicators (semáforo) that never rely on color alone — always paired with words and numbers.
- Plain-language Spanish on household screens; probabilities expressed as "X de cada 100".
