# Analisis de la iteracion 6

## 1. Objetivo de esta iteracion

Esta iteracion se enfoca en dos acciones concretas y pequenas:

1. Crear un modulo de diagnostico integral para observar claramente donde falla el modelo.
2. Desactivar por defecto `log1p` en el target para reducir compresion de cola alta en la escala normalizada.

La idea es mejorar observabilidad y calibracion del espacio objetivo sin redisenar arquitectura ni trainer.

## 2. Cambios realizados

### 2.1 Nuevo modulo `src/evaluation/diagnostics.py`

Se implemento `run_full_diagnostics(...)` con las entradas solicitadas:

- `y_pred_norm`, `y_real_norm`
- `norm_params`
- `df_train_norm`, `df_val_norm`, `df_test_norm`
- `target_col`

La funcion ahora genera y devuelve un diccionario JSON-serializable con cuatro bloques:

1. `normalized_target_distribution`
- min, max, mean, median, P95, P99, P99.9
- porcentaje `== 0`, `< 0.01`, `< 0.05`, `< 0.10`
- para train, val y test

2. `prediction_diagnostics`
- NSE, RMSE, MAE globales (en MGD reales)
- bias por buckets de severidad: base, pequeno, moderado, grande, extremo
- diagnostico de pico global (real, pred, error MGD, error %)
- ratio pred/real para top-10 picos reales
- maximo predicho vs maximo real

3. `residual_diagnostics`
- media, mediana, std
- % sobreestimacion > 2 MGD
- % infraestimacion > 2 MGD
- correlacion entre magnitud real y residuo

4. `normalization_diagnostics`
- rango efectivo del target normalizado por split
- compresion P95-P99 (span normalizado vs span real)
- flags de concentracion `>80%` bajo `0.05`

Ademas, la funcion imprime un resumen corto en consola para comparacion rapida entre iteraciones.

### 2.2 Ajuste en `src/pipeline/normalize.py`

Se cambio solo una linea:

- `apply_log1p_to_target: bool = True` -> `apply_log1p_to_target: bool = False`

No se modifico ninguna otra logica del modulo. La funcion sigue aceptando el parametro para reactivarlo manualmente cuando se desee.

## 3. Justificacion tecnica

La compresion excesiva del target en escala normalizada puede ocultar diferencias de magnitud en cola alta, justo donde el proyecto necesita sensibilidad operacional.

Cambiar el default a `False` permite que el flujo por defecto sea mas directo para distinguir eventos moderados, grandes y extremos. Al mismo tiempo, mantener el parametro configurable deja abierta la comparacion A/B futura con `log1p` sin tocar codigo adicional.

## 4. Riesgos y control experimental

- Riesgo principal: sin `log1p`, la cola alta puede volver mas inestable el entrenamiento.
- Mitigacion: el nuevo modulo de diagnostico permite monitorear inmediatamente compresion, sesgo por severidad y comportamiento de residuos para decidir el siguiente ajuste pequeno.

## 5. Siguiente validacion recomendada

En la siguiente corrida, comparar contra la iteracion previa:

1. `pct_lt_0_05` en train/val/test.
2. `norm_real_span_ratio` de P95-P99.
3. `bias` en buckets base y extremo.
4. ratio pred/real en top-10 picos.

Eso dira si el cambio de escala del target mejora separacion de magnitudes sin empeorar demasiado la estabilidad general.
