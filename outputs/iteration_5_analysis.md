# Analisis de la iteracion 5

## 1. Resumen ejecutivo

La regresion directa de la iteracion 4 elimino el cuello de botella del gating, pero tambien elimino su efecto de supresion fuera de evento.

Al revisar codigo y metricas, el problema principal no parece ser solo la arquitectura TCN, sino la **funcion objetivo** actual:

- Penaliza muy fuerte infraestimacion en cola alta.
- Penaliza poco y de forma indirecta la sobreestimacion cuando el target real esta cerca de cero.
- Con target en escala `log1p + minmax`, errores moderados en espacio normalizado se traducen en varios MGD al desnormalizar.

Esto explica el patron observado: columna vertical alta en `x~0`, sesgo positivo fuerte en base y, al mismo tiempo, compresion de picos severos.

## 2. Evidencia en archivos revisados

### 2.1 `src/models/tcn.py`

- La salida es regresion directa sin gating y sin restriccion explicita de magnitud.
- Esto es correcto para evitar compresion por probabilidad, pero deja toda la calibracion a la loss.

### 2.2 `src/models/loss.py`

La `CompositeLoss` actual tiene 3 terminos:

1. `weighted_huber`
2. penalizacion asimetrica de infraestimacion (`y_true >= p95`)
3. `peak_mse` (`y_true >= p95`)

No existe un termino dedicado a castigar sobreestimaciones cuando `y_true` esta en regimen base (cerca de 0 MGD).

### 2.3 `src/pipeline/sequences.py`

- Los pesos por muestra priorizan fuerte cola alta (`5, 12, 24`).
- En baseflow la mayoria queda con peso `1.0`.

Esto es coherente con el objetivo operativo, pero sin un termino explicito anti-falsos-positivos puede empujar al modelo a "encenderse" de mas en base para reducir riesgo de infraestimar picos.

### 2.4 `src/pipeline/normalize.py`

- El target se transforma con `log1p` y luego `minmax`.
- En esa escala, un error no muy grande puede crecer bastante al volver a MGD.

Concretamente, esto agrava visualmente el sesgo en base cuando el modelo sobrepredice en escala normalizada.

### 2.5 `outputs/data_analysis/test_metrics.json`

- `severity.base.bias = +3.615` MGD (sobreestimacion sistematica importante).
- `severity.extremo.bias = -47.93` MGD (infraestimacion extrema aun alta).
- `peak_error_pct = -42.55%`.

El modelo esta fallando en ambos extremos de calibracion: se mantiene alto en base y sigue corto en eventos severos.

## 3. Diagnostico

El fallo principal es de **alineacion de loss** tras quitar gating, no necesariamente de capacidad de backbone TCN.

La loss actual transmite con mucha claridad "no subestimes picos", pero transmite debilmente "no sobreestimes cuando no hay stormflow".

## 4. Cambio propuesto para iteracion 5 (pequeno y aislado)

Aplicar un cambio unico en `src/models/loss.py`:

1. Agregar un nuevo termino de perdida: **penalizacion de sobreestimacion en baseflow**.
2. Activarlo solo cuando `y_true <= umbral_base` (por defecto 0.5 MGD, convertido a escala normalizada igual que P95/P99/P99.9).
3. Mantener arquitectura y pipeline sin cambios.
4. Mantener los terminos de cola alta existentes para no perder enfoque en picos.

Justificacion:

- Es un ajuste pequeno y controlado (una sola pieza del sistema).
- Ataca directamente la patologia principal observada en scatter/residuos.
- Permite medir causalidad con una sola variable nueva antes de tocar arquitectura o features.

## 5. Hipotesis esperada de esta iteracion

Si la hipotesis es correcta, deberiamos ver:

1. Menor densidad vertical en `x~0`.
2. Menor bias en bucket `base`.
3. Menor pico en el histograma de residuos alrededor de `-10` MGD.
4. Sin degradar de forma fuerte la captura de extremos (o con degradacion menor que la mejora en base).

Si aun persiste infraestimacion severa de picos despues de este cambio, el siguiente ajuste incremental recomendado seria reintroducir una senal auxiliar de evento **solo en la loss** (sin volver al gating multiplicativo).
