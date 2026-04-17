# CLAUDE.md

Punto de entrada del proyecto para Claude Code. Lo leo siempre que abres una sesión.

---

## Qué es este proyecto

Predicción de picos de stormflow (caudal de tormenta) para el Metropolitan Sewer District of Greater Cincinnati (MSD), usando una red neuronal temporal (TCN) sobre datos de la estación MC-CL-005. El objetivo operativo es pasar de un sistema reactivo a uno predictivo para anticipar Combined Sewer Overflows (CSOs). El proyecto es también el Trabajo Fin de Máster (TFM) de Aimar.

Estado general: modelo base funcional con NSE=0.82 a H=1. Limitaciones conocidas en predicción de extremos. Entrega del TFM en septiembre 2026.

---

## Cómo leerme

No leas todos los archivos al arrancar. Lee solo este `CLAUDE.md` y elige qué más abrir según la tarea:

- Si necesitas entender el problema hidrológico, las reglas del proyecto y los anti-patrones aprendidos → lee `AGENTS.md`.
- Si necesitas saber el estado ACTUAL del proyecto hoy → lee `docs/STATE.md`.
- Si necesitas el historial de iteraciones → lee `docs/EXPERIMENTS.md`.
- Si necesitas saber cómo trabajar con el repo (Colab, Drive, Git) → lee `docs/WORKFLOW.md`.
- Para código fuente, abre bajo demanda desde `src/` según el área que toques.
- Para métricas del modelo actual, lee `outputs/data_analysis/local_eval_metrics.json`.
- Los `iteration_X_analysis.md` en `outputs/` son análisis detallados de las iteraciones 1-10. Son histórico. No los leas todos, solo el que corresponda si preguntan por una iteración concreta.

---

## Perfil del usuario (Aimar)

- Ingeniero de telecomunicaciones con doble máster (telecom + IA).
- Base teórica fuerte en matemáticas, señales y ML. Práctica limitada en programación desde cero.
- Principiante en Git, terminal, DevOps. Hay que explicarle los comandos.
- Entiende código Python cuando se le explica línea a línea, pero le cuesta escribirlo solo.
- Idioma: español siempre. Terminología técnica en inglés cuando es estándar (loss function, backpropagation, etc.).
- Quiere aprender de verdad, no solo obtener respuestas. Prefiere explicaciones con intuición primero, luego formalismo.
- No suavizar correcciones. Si algo está mal, decirlo directo y explicar por qué.

---

## Estructura del repositorio

stormflow-prediction/
├── CLAUDE.md                      ← Este archivo. Punto de entrada para Claude Code.
├── AGENTS.md                      ← Contexto permanente del problema y reglas.
├── README.md                      ← README público del repo (breve).
├── requirements.txt               ← Dependencias Python.
├── .gitignore
├── evaluate_local.py              ← Script de evaluación local (genera plots + métricas).
├── audit.py                       ← [opcional, puede no estar] script de auditoría puntual.
├── configs/
│   └── default.yaml               ← Rutas de datos (apuntan a Drive/Colab).
├── docs/
│   ├── STATE.md                   ← Estado vivo del proyecto (se actualiza cada sesión).
│   ├── EXPERIMENTS.md             ← Historial de iteraciones.
│   └── WORKFLOW.md                ← Cómo trabajar con el repo (Colab, Git, Drive).
├── src/                           ← Código fuente modular.
│   ├── data/                      ← load.py, clean.py, events.py (vacío).
│   ├── features/                  ← engineering.py (creación de features).
│   ├── pipeline/                  ← split.py, normalize.py, sequences.py.
│   ├── models/                    ← tcn.py (TwoStageTCN), loss.py (TwoStageLoss, CompositeLoss).
│   ├── training/                  ← trainer.py (train_model, predict).
│   └── evaluation/                ← metrics.py (NSE, RMSE, severidad), diagnostics.py.
├── notebooks/                     ← Notebooks Colab descargados como .py.
│   ├── claude_train.py            ← Notebook principal de entrenamiento.
│   ├── horizon_comparison.py      ← Comparación v1 de horizontes H1/H3/H6.
│   └── horizon_comparison_v2.py   ← Comparación v2 (reentrenamiento H1_sinSF).
├── outputs/
│   ├── proposal.md                ← Propuesta original del modelo (histórico).
│   ├── iteration_1_analysis.md a iteration_10_analysis.md   ← Histórico detallado.
│   ├── data_analysis/             ← JSONs de métricas y diagnósticos.
│   │   ├── local_eval_metrics.json              ← MÉTRICAS AUTORITATIVAS del modelo actual.
│   │   ├── extreme_events_no_rain.json          ← Análisis de los 15 extremos sin lluvia.
│   │   ├── full_diagnostics.json                ← Diagnóstico completo histórico.
│   │   ├── permutation_importance.json          ← PI histórico.
│   │   ├── threshold_sweep_H1_conSF.json
│   │   ├── threshold_sweep_H1_sinSF.json
│   │   ├── test_metrics.json                    ← Histórico.
│   │   └── analysis_report.md                   ← Histórico.
│   ├── figures/
│   │   └── local_eval/            ← 56 plots generados por evaluate_local.py.
│   └── models/
│       └── tcn_best.pt            ← Peso histórico, NO es el modelo en producción.
└── MC-CL-005/                     ← Datos brutos (NO en GitHub, solo local).
├── 1parte/                    ← rain_list.tsf, flow_list.tsf, lstFlowTs.tsf, lstStormTs.tsf, eventos.
├── 2parte/                    ← ídem.
├── daily_temperatures_2006_2026.tsf
└── Pesos 13-04-2026/          ← 7 modelos entrenados (6 comparativos + 1 v2).

---

## Archivos autoritativos (source of truth)

- **Código fuente del pipeline**: `src/` en el repo local (sincronizado con GitHub).
- **Datos brutos**: `MC-CL-005/` en local (NO en GitHub por tamaño y confidencialidad).
- **Pesos del modelo actual**: `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_weights.pt` (v1) — el mejor por NSE global.
- **Parámetros de normalización del modelo actual**: `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_norm_params.json`.
- **Metadata del modelo actual**: `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_meta.json`.
- **Métricas del modelo actual**: `outputs/data_analysis/local_eval_metrics.json`.
- **Plots del modelo actual**: `outputs/figures/local_eval/`.
- **Notebooks de Colab (código real que entrenó los modelos)**: `notebooks/*.py`.

El `outputs/models/tcn_best.pt` es un peso histórico de una iteración anterior al Two-Stage. NO es el modelo en producción.

---

## Modelo actual en producción

- Nombre: `modelo_H1_sinSF` (v1).
- Arquitectura: TwoStageTCN (backbone compartido + clasificador + regresor con switch duro).
- Horizonte de predicción: 1 paso = 5 minutos.
- Sin stormflow como feature (22 features de entrada).
- Seq length: 72 pasos (6 horas de contexto).
- Entrenamiento: AdamW, lr=5e-4, batch=256, early stopping patience=10.
- Mejor época en entrenamiento: 6.
- Métricas test: NSE=0.819, RMSE=1.02 MGD, MAE=0.31 MGD, error de pico +12.8%.

Existe también una `v2` del mismo modelo (`modelo_H1_sinSF_v2_*`) entrenada el 16-abril-2026. Captura ligeramente mejor los picos que v1 pero v1 sigue siendo el mejor por NSE global.

---

## Reglas críticas (no romper)

1. **Datos NUNCA en GitHub.** `MC-CL-005/` no se commitea. Verificar que está en `.gitignore`.
2. **Split SIEMPRE cronológico.** Nunca aleatorio en series temporales.
3. **Normalización con estadísticas de train solamente.** Nunca calcular min/max/mean/std con val o test.
4. **`flow_total_mgd` NUNCA como feature.** Tiene r=0.9976 con stormflow, es un atajo confirmado en iter 11. Ya está excluida de `FEATURE_COLUMNS`.
5. **`log1p` en target es obligatorio.** Sin él, NSE cayó a -26.6 en iter 6 (catastrófico). No quitarlo.
6. **UN cambio por iteración.** Nunca tocar arquitectura + loss + features a la vez. Imposible diagnosticar qué ayudó.
7. **Hipótesis ANTES de cambiar.** No "probar a ver qué pasa". Si no hay hipótesis, no hay cambio.
8. **Colab SOLO para entrenar.** Todo análisis y plot se hace en VS Code local con `evaluate_local.py`.
9. **Comentarios en español, código en inglés (snake_case).** Type hints en funciones públicas.
10. **Máximo 3 archivos modificados por tarea.** Si hay que tocar más, separar en tareas.

---

## Workflow típico de una iteración

Flujo resumido (detalle completo en `docs/WORKFLOW.md`):

1. Decidir hipótesis concreta de cambio.
2. Modificar código en VS Code local (rama `main`).
3. `git add` + `git commit -m "exp(iterN): descripción breve"` + `git push`.
4. En Colab: `git pull`.
5. Ejecutar notebook de entrenamiento.
6. Guardar pesos y metadata en `Pesos 13-04-2026/` (o carpeta nueva con fecha actualizada).
7. Descargar pesos a local.
8. Ejecutar `python evaluate_local.py` para generar plots y métricas.
9. Añadir entrada en `docs/EXPERIMENTS.md`.
10. Actualizar `docs/STATE.md` con el nuevo estado.
11. Commit: `docs: actualizar estado post-iterN` y push.

---

## Prioridad actual (abril 2026)

**Mejorar el modelo**. El TFM se redacta después, hay margen hasta septiembre 2026 para entrega. No empezar redacción extensa del TFM hasta que el modelo esté cerrado, para evitar reescribir.

Siguiente paso concreto: pendiente de definir (ver `docs/STATE.md` sección "Siguientes pasos").

---

## Cosas que NO hacer en este proyecto

- No proponer cambiar a LSTM, Transformer puro o arquitecturas radicalmente distintas sin evidencia de que la TCN es insuficiente. La TCN está justificada por la memoria corta del sistema (lag óptimo 10 min).
- No usar gating multiplicativo (probabilidad × magnitud). Comprime picos. Usar switch duro (if/else).
- No usar sampling estratificado agresivo. Causa drift train/val. Preferir distribución natural + loss ponderada.
- No usar Cosine Annealing. Probado iter 14, empeora. Usar ReduceLROnPlateau.
- No incluir `hour_sin`/`hour_cos` si hay diagnóstico que lo desaconseje. PI negativo consistente en iters anteriores.
- No quitar todos los `rain_sum_*` largos a la vez. Iter 12 demostró que son necesarios para calibrar magnitud.
- No prometer mejoras en extremos sin datos que lo soporten. El error en pico del ~83% es sistemático, no aleatorio. 15 de 59 eventos extremos no tienen lluvia en la ventana de entrada (físicamente impredecibles con los datos actuales).