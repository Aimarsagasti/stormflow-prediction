# Analisis de la iteracion 9

## 1. Contexto observado

Tras agregar `api_dynamic` y `temp_daily_f` en la iteracion 8, el comportamiento global fue mixto:

- `NSE` mejoro de `-2.58` a `-1.94`.
- El sesgo en baseflow empeoro (`+2.81` a `+3.27` MGD).
- El error de pico global empeoro fuerte (`-30%` a `-63%`).
- El mejor epoch aparecio muy temprano (epoch 2), con oscilacion alta en validacion y corte temprano posterior.

Esto sugiere que el modelo encontro rapido una solucion local que mejora algo el promedio, pero no estabiliza el aprendizaje de la cola extrema (justo donde importa para CSO).

## 2. Cambio implementado en diagnosticos

Se agrego `permutation importance` en `src/evaluation/diagnostics.py` mediante la nueva funcion publica:

- `run_permutation_importance(model, dataloader, device, feature_columns, norm_params)`

### 2.1 Como funciona

1. Calcula baseline con el dataloader original y obtiene RMSE en MGD reales (desnormalizado).
2. Para cada feature:
   - baraja esa columna en todos los batches (manteniendo forma del tensor)
   - vuelve a inferir
   - recalcula RMSE
   - mide `delta_rmse` respecto al baseline
3. Ordena features por `delta_rmse` descendente y devuelve ranking serializable.
4. Imprime ranking en consola para inspeccion rapida.

### 2.2 Salida esperada

Cada fila del ranking incluye:

- `feature`
- `baseline_rmse_mgd`
- `permuted_rmse_mgd`
- `delta_rmse_mgd`
- `relative_increase_pct`

Esto permite verificar si `api_dynamic` y `temp_daily_f` realmente aportan senal o si estan metiendo ruido temprano en entrenamiento.

## 3. Analisis de convergencia rapida

Hipotesis mas probable: **paso de optimizacion demasiado agresivo al inicio** para el nuevo espacio de entrada.

Razonamiento:

- Con features nuevas cambia la geometria del problema (escala, varianza y correlaciones cruzadas).
- El modelo ahora puede dar saltos grandes al principio: baja rapido algo de loss media, pero no construye representacion estable para extremos.
- Ese patron es consistente con mejor epoch muy temprano + validacion oscilante + degradacion fuerte en picos grandes.

No hay evidencia suficiente aun para quitar o debilitar las features nuevas; primero conviene estabilizar el arranque y medir importancia real con el diagnostico agregado.

## 4. Cambio pequeno aplicado para estabilizacion

Se aplico **un solo ajuste** en entrenamiento:

- `src/training/trainer.py`
- `learning_rate` por defecto: `1e-3` -> `5e-4`

Justificacion:

- Es el cambio mas pequeno y de menor riesgo para reducir oscilacion de validacion en etapas iniciales.
- Mantiene intacta arquitectura, loss y features.
- Si el problema era un inicio demasiado brusco, este ajuste deberia retrasar convergencia prematura y dar mas epocas utiles antes de early stopping.

## 5. Archivos modificados

- `src/evaluation/diagnostics.py`
- `src/training/trainer.py`

## 6. Recomendacion de validacion en la proxima corrida

1. Correr entrenamiento con el nuevo `learning_rate` por defecto.
2. Ejecutar `run_permutation_importance(...)` sobre test o val.
3. Revisar tres señales juntas:
   - si el mejor epoch deja de ser tan temprano
   - si mejora error de pico y top-10 ratios
   - si `api_dynamic` y `temp_daily_f` aparecen con `delta_rmse` positivo claro

Si la curva sigue oscilando fuerte aun con `5e-4`, el siguiente ajuste minimo recomendado seria mantener LR y suavizar solo una componente de la loss (no varias a la vez) para aislar causa.
