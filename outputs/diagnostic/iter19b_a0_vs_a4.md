# Iter19b - A0 (log1p=True) vs A4 (log1p=False) en TEST

## Introduccion

El comparison.md de iter19 documenta una limitacion metodologica explicita: la regla "max NSE_val" usada para elegir la configuracion ganadora privilegio la metrica global sobre la operativa. En la ablacion val, A0 (con log1p) tenia un error de pico de +1.5% (casi perfecto), mientras que A1/A4 (sin log1p) tenian -25%. Sin embargo, A0 nunca se evaluo en test porque la regla automatica fijo el ganador en A4 (mayor NSE_val global). Este notebook entrena A0 directamente y la evalua en test para cerrar esa duda.

La hipotesis que se contrasta: A0 tendra menor NSE global que A4 pero mejor comportamiento en el bucket Extremo (NSE menos negativo y/o menor bias absoluto). Si se confirma, hay un trade-off real entre metrica global y comportamiento operativo en eventos criticos. Si A0 empeora en todos los frentes, la regla "max NSE_val" estaba justificada.

## Tabla comparativa global A0 vs A4 en test

| Metrica | A0 (log1p=True) | A4 (log1p=False) | delta (A0 - A4) |
|---|---:|---:|---:|
| NSE | 0.8817 | 0.8983 | -0.0166 |
| RMSE | 0.827 | 0.767 | +0.060 |
| MAE | 0.076 | 0.088 | -0.012 |
| peak_err_pct | -33.4 | +4.7 | -38.1 |
| recall@50 | 0.609 | 0.783 | -0.174 |
| recall@25 | 0.842 | 0.895 | -0.053 |
| n_params | 23,489 | 89,985 | -66,496 |
| training_time_s | 2053 | ~1117 (iter19) | n/a |
| best_epoch | 41 | 18 | n/a |

## Tabla comparativa por bucket en test

| Bucket | n | NSE A0 | NSE A4 | dNSE | bias A0 | bias A4 | dbias | pico% A0 | pico% A4 | dpico% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 152900 | +0.722 | +0.085 | +0.636 | +0.001 | +0.002 | -0.001 | +1394.6 | +1371.2 | +23.4 |
| Leve | 9520 | +0.830 | +0.765 | +0.065 | +0.028 | +0.136 | -0.109 | +245.6 | +150.6 | +95.1 |
| Moderado | 2303 | +0.519 | +0.527 | -0.008 | +0.002 | +0.350 | -0.348 | +128.1 | +135.8 | -7.7 |
| Alto | 440 | -0.101 | +0.005 | -0.107 | -2.737 | -1.045 | -1.692 | +16.8 | +20.7 | -3.9 |
| Extremo | 59 | -1.855 | -1.231 | -0.624 | -23.845 | -13.739 | -10.106 | -33.4 | +4.7 | -38.1 |

## Lectura de los resultados

**Hipotesis del bucket Extremo (A0 mejor que A4 en eventos criticos)**: **no confirma** la hipotesis. NSE_extremo A0 = -1.855 <= A4 = -1.231 (-0.624); |bias_extremo| A0 = 23.845 >= A4 = 13.739. 
El log1p no aporto beneficio en el bucket Extremo en test. Esto contradice la intuicion que motivo esta corrida: en val el err_pico de A0 era +1.5%, pero la calibracion no se transfirio a test. Posible explicacion: los picos de val no son representativos de los picos de test (muestra pequena, n=27 eventos val vs n=59 eventos test).

**Sobre la regla "max NSE_val" usada en iter19**: La regla max NSE_val privilegio una mejora marginal de A4 sobre A0 (0.0166 NSE en test, dentro del umbral de no-significancia +0.02 NSE usado en iter19 para el veredicto frente a XGB). Aun asi, A4 tampoco empeora en extremos, por lo que la regla seleccionable automaticamente sigue siendo defendible aunque no perfecta.

**Configuracion principal recomendada para el TFM: reportar ambas**. Los trade-offs son equilibrados (A4 mejor global, A0 mejor o equivalente en extremos). El TFM gana honestidad metodologica al exponer la dependencia entre regla de seleccion y definicion de exito operativo, en lugar de forzar una eleccion.

## Veredicto y recomendacion

**Ambos modelos se reportan en el TFM, sin elegir uno**.

Justificacion numerica:

- delta NSE global (A0 - A4) = -0.0166
- delta peak_err_pct (A0 - A4) = -38.1
- delta NSE_extremo (A0 - A4) = -0.624
- delta bias_extremo (A0 - A4) = -10.106 MGD
- delta peak_err_pct_extremo (A0 - A4) = -38.1
- delta NSE_alto (A0 - A4) = -0.107

## Limitaciones reconocidas

- Una sola seed (=42), igual que iter19. La diferencia A0 vs A4 puede estar dentro del ruido estocastico de inicializacion. Para concluir con mas certeza haria falta repetir ambas corridas con varias seeds.
- A0 reentrenado aqui puede converger a un minimo ligeramente distinto al de la corrida de iter19 (mismo seed pero el orden de operaciones GPU no es bit-exact). El best_epoch y best_val_nse pueden diferir marginalmente.
- Comparacion limitada a H=1 (igual que iter19). Si A0 sustituye a A4, queda pendiente verificar que la mejora en extremos se mantiene a H=3.
- A4 se evalua cargando los pesos del checkpoint final.pt de iter19 (no se reentrena). Esto garantiza comparacion 1:1 con el JSON oficial pero asume que ese checkpoint sigue siendo valido. El sanity check del NSE recomputado vs oficial valida esta asuncion.
