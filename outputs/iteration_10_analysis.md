# Iteration 10 Analysis

## Que se cambio y por que
- Se reemplazo el escalado Min-Max por z-score en la normalizacion para evitar la compresion extrema causada por outliers en el maximo de train.
- Se mantuvo log1p en las mismas columnas (lluvia y opcionalmente el target) para estabilizar la cola.
- Se actualizo `norm_params` para almacenar `mean` y `std`, y se ajustaron normalizacion y desnormalizacion del target para usar esas estadisticas.

## Formulas antes vs despues
Antes (log1p + Min-Max):
- Normalizacion: `x_norm = (log1p(x) - min_train) / range_train`
- Desnormalizacion: `x_real = expm1(x_norm * range_train + min_train)`

Despues (log1p + z-score):
- Normalizacion: `x_norm = (log1p(x) - mean_train) / std_train`
- Desnormalizacion: `x_real = expm1(x_norm * std_train + mean_train)` y luego `clip(x_real, min=0)` para mantener stormflow no negativo.

## Archivos verificados sin cambios necesarios
- `src/models/loss.py`
- `src/evaluation/metrics.py`
- `src/evaluation/diagnostics.py`
