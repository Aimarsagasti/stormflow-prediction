# Informe de Analisis - Modelos Stormflow TCN
## Fecha de generacion: 2026-04-14

## 1. Resumen ejecutivo
Los modelos son utiles para prediccion muy corta, pero no son robustos a horizontes largos ni a eventos extremos. En global, `H1_sinSF` es el mejor modelo por NSE=0.861 y RMSE=0.894 MGD, mientras que `H1_conSF` mejora MAE (0.242 MGD) pero introduce sobreestimacion severa del pico maximo (+58.8%). A 15 minutos (`H3`) el sistema sigue siendo util para descripcion general de eventos, pero ya pierde mucha fidelidad en magnitud; a 30 minutos (`H6`) el modelo `sinSF` colapsa (NSE=-1.212) y `conSF` solo conserva utilidad limitada (NSE=0.255) con errores enormes en extremos. Para el TFM, el material actual es suficiente para documentar resultados y discutir limitaciones con honestidad, pero no para afirmar una solucion operativa robusta de captura de picos extremos.

## 2. Metricas globales

| Modelo | NSE | RMSE (MGD) | MAE (MGD) | Error pico (%) | Bias base (MGD) | Bias extremo (MGD) |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 0.861 | 0.894 | 0.293 | -21.0 | +0.21 | -12.71 |
| H1_conSF | 0.854 | 0.916 | 0.242 | +58.8 | +0.18 | -8.86 |
| H3_sinSF | 0.471 | 1.745 | 0.379 | -52.1 | +0.25 | -15.89 |
| H3_conSF | 0.488 | 1.718 | 0.354 | -48.6 | +0.24 | -20.38 |
| H6_sinSF | -1.212 | 3.570 | 1.211 | -78.2 | +0.94 | -50.46 |
| H6_conSF | 0.255 | 2.072 | 0.493 | -88.2 | +0.28 | -57.36 |

Comentario general:
- El mejor modelo global es `H1_sinSF`. Tiene el mayor NSE, el menor RMSE y un error de pico maximo importante pero todavia interpretable (-21.0%).
- `H1_conSF` parece competitivo solo si se mira MAE, pero el error del pico maximo (+58.8%) es demasiado agresivo para un problema donde la calibracion del pico importa mucho.
- La degradacion por horizonte no es lineal. La caida de `H1 -> H3` es moderada; la de `H3 -> H6` es brusca, especialmente en `sinSF`.
- `H6_conSF` supera claramente a `H6_sinSF`, pero sigue estando lejos de un nivel robusto para uso serio en picos.

## 3. Metricas por rango de magnitud

### Base (<0.5 MGD)

| Modelo | N | NSE | RMSE | MAE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 152904 | -14.513 | 0.344 | 0.212 | 73605.7 | +0.206 |
| H1_conSF | 152904 | -9.734 | 0.286 | 0.186 | 71289.1 | +0.181 |
| H3_sinSF | 152902 | -99.064 | 0.875 | 0.252 | 84822.8 | +0.245 |
| H3_conSF | 152902 | -125.394 | 0.983 | 0.244 | 86207.4 | +0.239 |
| H6_sinSF | 152899 | -1056.103 | 2.843 | 0.945 | 315069.1 | +0.942 |
| H6_conSF | 152899 | -115.940 | 0.946 | 0.282 | 81224.1 | +0.277 |

Lectura:
- Ningun modelo funciona bien en NSE para baseflow. Todos tienen sesgo positivo y todos sobrepredicen el nivel base.
- `H1_conSF` es el menos malo en base por RMSE, MAE y bias.
- El NSE base extremadamente negativo no significa que el modelo "explote" visualmente todo el tiempo; significa que pequenas desviaciones positivas son muy penalizadas cuando la varianza real es casi cero.
- `H6_sinSF` es claramente inutil en baseflow.

### Leve (0.5-5 MGD)

| Modelo | N | NSE | RMSE | MAE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 9518 | -0.256 | 1.222 | 0.710 | 59.8 | +0.554 |
| H1_conSF | 9518 | 0.410 | 0.838 | 0.399 | 32.4 | +0.354 |
| H3_sinSF | 9518 | -3.002 | 2.182 | 0.974 | 79.5 | +0.693 |
| H3_conSF | 9518 | -2.578 | 2.063 | 0.807 | 65.3 | +0.713 |
| H6_sinSF | 9518 | -23.616 | 5.411 | 3.312 | 272.4 | +3.209 |
| H6_conSF | 9518 | -9.083 | 3.463 | 1.925 | 150.1 | +1.780 |

Lectura:
- `H1_conSF` es el unico modelo claramente util en flujo leve.
- `H1_sinSF` ya pierde capacidad en este rango, aunque sigue siendo razonable visualmente en algunos eventos.
- `H3` empeora mucho y `H6` deja de ser defendible.
- Si el objetivo fuera solo caudales leves o pequenos eventos, `H1_conSF` seria la mejor opcion.

### Moderado (5-20 MGD)

| Modelo | N | NSE | RMSE | MAE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 2305 | 0.191 | 3.639 | 2.424 | 25.8 | +1.465 |
| H1_conSF | 2305 | 0.518 | 2.809 | 1.659 | 17.1 | +0.435 |
| H3_sinSF | 2305 | -1.368 | 6.228 | 3.865 | 40.4 | +2.395 |
| H3_conSF | 2305 | -1.073 | 5.827 | 3.287 | 34.4 | +1.693 |
| H6_sinSF | 2305 | -5.197 | 10.076 | 6.605 | 74.3 | +4.481 |
| H6_conSF | 2305 | -1.558 | 6.474 | 4.576 | 50.2 | +1.262 |

Lectura:
- `H1_conSF` es el mejor modelo en eventos moderados. Es la unica configuracion con NSE claramente positivo y margen amplio sobre el resto.
- `H1_sinSF` funciona, pero con bastante mas dispersioin.
- `H3_conSF` supera a `H3_sinSF`, aunque ambos ya son negativos.
- En moderado, el stormflow como feature si ayuda.

### Alto (20-50 MGD)

| Modelo | N | NSE | RMSE | MAE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 437 | 0.021 | 7.314 | 5.537 | 19.9 | +0.050 |
| H1_conSF | 437 | 0.021 | 7.314 | 5.309 | 18.6 | -2.476 |
| H3_sinSF | 437 | -3.046 | 14.866 | 8.697 | 30.1 | -0.266 |
| H3_conSF | 437 | -4.186 | 16.831 | 9.157 | 31.7 | -1.355 |
| H6_sinSF | 437 | -5.834 | 19.322 | 13.290 | 46.7 | -5.330 |
| H6_conSF | 437 | -4.749 | 17.722 | 13.906 | 47.5 | -12.190 |

Lectura:
- El rendimiento alto es mediocre incluso en `H1`. El NSE apenas supera cero.
- `H1_sinSF` y `H1_conSF` quedan practicamente empatados en NSE, con ligera ventaja de `conSF` en MAE y MAPE pero peor bias.
- `H3` y `H6` ya no sirven bien para magnitud alta.
- No hay un ganador claro en este rango: solo dos modelos "aceptables", ambos en `H1`.

### Extremo (>50 MGD)

| Modelo | N | NSE | RMSE | MAE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF | 59 | -0.986 | 27.819 | 21.733 | 30.3 | -12.706 |
| H1_conSF | 59 | -2.389 | 36.345 | 27.289 | 39.8 | -8.865 |
| H3_sinSF | 59 | -5.727 | 51.206 | 36.730 | 50.3 | -15.886 |
| H3_conSF | 59 | -3.220 | 40.559 | 31.933 | 44.3 | -20.377 |
| H6_sinSF | 59 | -7.405 | 57.237 | 52.609 | 75.6 | -50.464 |
| H6_conSF | 59 | -8.768 | 61.703 | 57.358 | 81.9 | -57.358 |

Lectura:
- Ningun modelo es bueno en extremos. Esto es el principal limite tecnico del sistema actual.
- `H1_sinSF` es el menos malo en extremos por NSE, RMSE y MAE.
- `H1_conSF` reduce algo el bias medio extremo frente a `H1_sinSF` (-8.865 vs -12.706), pero la dispersion es mucho peor y el NSE cae mas.
- `H6` es claramente inutil para extremos.

## 4. Analisis de la curva de predicibilidad
En `predictability_curve.png` se ven dos curvas simples, una azul (`sinSF`) y una naranja (`conSF`). Las dos arrancan muy altas en `H=1`, casi superpuestas: `H1_sinSF=0.861` y `H1_conSF=0.854`. En `H=3` ambas caen alrededor de 0.48, todavia utiles pero ya lejos del nivel de corto plazo. El quiebre fuerte aparece entre `H=3` y `H=6`: la curva azul cae de 0.471 a -1.212, cruza la linea `NSE=0` y termina muy abajo; la curva naranja solo baja a 0.255 y se mantiene positiva.

Conclusiones:
- La caida no es lineal. Hay un punto de quiebre claro entre 15 y 30 minutos.
- El stormflow como feature no ayuda realmente en `H1`; de hecho, el mejor NSE es `H1_sinSF`.
- El stormflow si ayuda mucho a 30 minutos, porque evita el colapso total que sufre `H6_sinSF`.
- `H6_sinSF` deja de ser util porque tiene `NSE < 0`. `H6_conSF` todavia es mejor que la media, pero solo marginalmente.

## 5. Analisis de los hidrogramas

### Evento moderado 1 - 2024-12-15 03:30 - pico real 10.1 MGD
Visualmente, el hidrograma real es negro y las seis predicciones lo rodean con bastante fidelidad en timing general. `H1_sinSF` y `H1_conSF` capturan bien la secuencia de subpicos entre las horas 5 y 15, aunque ambos tienden a ir algo por encima del negro en las recesiones. `H3_sinSF` y `H3_conSF` siguen la forma, pero mas suavizados. `H6_sinSF` y `H6_conSF` aparecen en rojo, con nivel base demasiado alto casi todo el tiempo y picos espurios antes y despues del evento; son claramente mas ruidosos. La forma general del evento si esta capturada, pero el principal error en `H6` es un offset positivo persistente mas que un fallo completo de timing.

### Evento moderado 2 - 2025-01-18 03:15 - pico real 10.3 MGD
Este evento es mas limpio: hay un pico principal alrededor de la hora 4 y luego una cola larga. `H1_sinSF` y `H1_conSF` lo siguen bien y llegan al pico casi en el mismo momento; `H1_sinSF` incluso toca ligeramente por encima del real. `H3_sinSF` y `H3_conSF` siguen la estructura pero infraestiman la cola final y suavizan la recesion. `H6_sinSF` y `H6_conSF` vuelven a mostrar sobreprediccion base y algun spike aislado antes del evento. Aqui la diferencia entre horizontes se ve con claridad: `H1` es util, `H3` es aceptable, `H6` introduce sesgo y ruido.

### Evento alto 1 - 2024-09-30 06:35 - pico real 33.5 MGD
La figura muestra un gran pico centrado cerca de la hora 10. El hidrograma real sube casi verticalmente y cae rapido. `H1_sinSF` y `H1_conSF` lo siguen muy bien en timing y magnitud, con maximos apenas por encima de 50 MGD en azul, lo que implica una sobreestimacion importante del pico. `H3_sinSF` y `H3_conSF` tambien reaccionan fuerte y generan un pico muy estrecho, algo mas alto que el real. `H6_sinSF` y `H6_conSF` fallan mas: mantienen una base elevada antes del pico y tienen actividad espuria adicional despues. El timing no parece malo en `H1/H3`; el problema aqui es mas de amplitud y falsas oscilaciones.

### Evento alto 2 - 2024-12-09 00:30 - pico real 33.6 MGD
Este hidrograma es mas ancho y multietapa. `H1_sinSF` y `H1_conSF` reproducen razonablemente bien el ascenso inicial hasta el maximo y la posterior meseta entre 12 y 20 MGD. `H3_sinSF` y `H3_conSF` quedan algo mas bajos y mas suaves. `H6_sinSF` y `H6_conSF` van altos casi todo el tiempo, con una recesion mucho mas lenta que la real. En este caso el modelo no falla por delay severo: el pico aparece casi a tiempo. El fallo esta en la cola y en la sobreestimacion persistente posterior.

### Evento extremo con lluvia 1 - 2025-04-29 18:20 - pico real 70.3 MGD
Aqui aparece una subida explosiva cerca de la hora 5.8-6.0. `H1_sinSF`, `H1_conSF`, `H3_sinSF` y `H3_conSF` reaccionan todos en el momento correcto, pero ninguno reproduce bien la altura exacta. El mejor visualmente es `H3_conSF`, que sobrepasa el negro y llega alrededor de 75 MGD; `H1` se queda mas cerca de 50-55 MGD y por tanto infraestima el pico. `H6_sinSF` tiene un overshoot rojo posterior muy tardio, y `H6_conSF` queda bajo y con mucho offset. Este grafico sugiere que en extremos el sistema "detecta" el evento, pero no calibra bien la magnitud.

### Evento extremo con lluvia 2 - 2024-09-27 16:40 - pico real 71.3 MGD
Es el hidrograma mas largo y complejo. El evento principal ocurre alrededor de las horas 19-22, con un cluster de picos. Todos los modelos reaccionan en esa zona, pero hay mucho ruido antes y despues. `H1_sinSF` y `H1_conSF` generan un gran pico cercano a 70-75 MGD, bastante cerca del real. `H3` tambien llega alto. `H6_sinSF` y `H6_conSF` levantan una base roja persistentemente positiva durante casi todo el evento y disparan spikes adicionales en varias zonas. La forma gruesa esta capturada; la estabilidad no.

Resumen general de hidrogramas:
- `H1` captura bien la forma del hidrograma en moderados y altos.
- `H3` conserva timing razonable, pero ya suaviza o descalibra la magnitud.
- `H6` es el mas ruidoso y el que mas offset positivo muestra.
- Las predicciones no son lineas vacias: todos los paneles muestran cobertura completa. El problema no es falta de salida, sino calidad de la salida.

## 6. Comportamiento general (temporal_range)
`temporal_range_1.png` muestra la ventana 2024-11-14 a 2024-11-28. La serie negra tiene varios episodios de lluvia y algunos picos estrechos. En calma, ambos modelos `H1` tienden a quedar algo por encima del cero real, pero no generan falsas alarmas grandes; el error base es mas bien un "colchon" positivo pequeno. Cuando llega la tormenta grande alrededor del 19 de noviembre, ambos reaccionan exactamente en el momento del ascenso, lo cual es buena noticia para el clasificador. El pico mas alto se sobreestima: `H1_sinSF` llega alrededor de 44-45 MGD y `H1_conSF` alrededor de 35 MGD, mientras el real negro queda algo por debajo. En los eventos menores del 21, 25 y 28 de noviembre ambos vuelven a reaccionar a tiempo y siguen razonablemente bien la forma.

Comparacion visual entre `H1_sinSF` y `H1_conSF`:
- `H1_sinSF` parece mas agresivo y produce picos algo mas altos.
- `H1_conSF` parece un poco mas contenido y suave.
- No se observan falsas alarmas largas sin lluvia.
- En timing los dos son buenos; la diferencia principal es de amplitud.

## 7. Comparacion sin/con stormflow
Las tres figuras `comparison_sf_vs_nosf_*.png` son muy utiles porque comparan exactamente los mismos eventos.

### Comparacion 1 - 2025-10-07 02:50 - pico 28.1 MGD
En el panel `sin stormflow`, `H1_sinSF` sigue bastante bien el pico principal y los subpicos posteriores. `H3_sinSF` tiene varios spikes espurios altos antes del evento y vuelve a exagerar despues. En `con stormflow`, `H1_conSF` mantiene forma similar a `H1_sinSF`, pero `H3_conSF` muestra aun mas spikes espurios al principio y en el tramo medio. Aqui anadir stormflow no mejora visualmente: el panel derecho se ve mas inestable.

### Comparacion 2 - 2024-07-24 16:25 - pico 28.2 MGD
Ambos paneles capturan el pico principal cerca de la hora 4. `H1_sinSF` y `H1_conSF` hacen un trabajo parecido. `H3_sinSF` y sobre todo `H3_conSF` muestran overshoot claro: en `conSF` aparece un pico naranja punteado por encima de 80 MGD, muy lejos del real. Este es el ejemplo mas claro de sobreestimacion grave asociada a incluir stormflow en horizontes mayores.

### Comparacion 3 - 2025-07-16 12:55 - pico 28.5 MGD
`H1_sinSF` y `H1_conSF` vuelven a seguir bastante bien los dos picos negros principales. `H3_sinSF` sobreestima moderadamente. `H3_conSF` vuelve a disparar picos punteados muy altos, uno de ellos cerca de 47-48 MGD cuando el real esta en 28.5 MGD. La inclusion de stormflow no arruina `H1`, pero si vuelve mas inestable a `H3`.

Conclusion de esta comparacion:
- Anadir stormflow como feature ayuda en rangos leves y moderados segun metrica, pero visualmente no mejora de forma consistente.
- En `H1`, la diferencia entre sinSF y conSF es pequena y depende del evento.
- En `H3`, `conSF` tiende a sobreestimar mas y a introducir spikes mas agresivos.
- Si la prioridad es robustez de picos y narrativa segura para el TFM, no conviene vender `conSF` como modelo principal.

## 8. Analisis de eventos extremos

### 8.1 Con lluvia vs sin lluvia
El archivo `extreme_events_no_rain.json` reporta:
- Total de extremos: 59
- Con lluvia: 59
- Sin lluvia: 0

Metricas separadas del grupo con lluvia (`H1_sinSF`):
- NSE = -0.986
- RMSE = 27.819 MGD
- MAE = 21.733 MGD
- Error medio = -17.87%
- Error mediano = -15.19%
- Error minimo = -78.14%
- Error maximo = +81.25%

Lo que se ve en `extreme_rain_vs_norain.png` coincide con eso: el panel izquierdo contiene las 59 muestras, muy dispersas alrededor de la diagonal, y el panel derecho esta vacio con el texto `Sin muestras`.

Observacion critica:
- Este resultado contradice el contexto de `project_status.md`, que afirmaba aproximadamente 15 extremos sin lluvia.
- Por tanto, con los outputs actuales no hay evidencia reproducida de extremos sin lluvia.
- Esto puede deberse a un cambio en el criterio de lluvia, a una alineacion distinta respecto al input historico o a que la nueva version del script esta clasificando de otra forma.

### 8.2 Eventos sin lluvia - hipotesis de deshielo
El archivo `outputs/data_analysis/extreme_no_rain_detailed.json` no existe en el workspace al momento de este analisis. Eso es coherente con el hecho de que el analisis anterior no encontro ningun extremo sin lluvia y por tanto la funcion no genero detalle.

Consecuencias:
- No se puede sostener ni refutar con datos actuales la hipotesis de deshielo.
- No se puede responder empiricamente en que meses ocurren ni que temperaturas tienen esos supuestos eventos.
- Para el mentor, esto debe leerse como una limitacion del output actual, no como evidencia de ausencia fisica del fenomeno.

Si no son deshielo, otras causas plausibles mencionables en discusion metodologica son:
- retraso hidrologico largo entre lluvia acumulada y pico de stormflow,
- lluvia fuera de la ventana de entrada seleccionada,
- artefactos de agregacion de lluvia,
- desalineacion temporal entre sensores de lluvia y caudal.

## 9. Scatter plots
`scatter_all_models.png` resume muy bien el estado global.

Lo que se observa:
- En `H1_sinSF`, la nube de puntos base y moderados se agrupa relativamente cerca de la diagonal 1:1. Los puntos rojos de extremos estan dispersos, varios por debajo de la diagonal y algunos por encima.
- En `H1_conSF`, la nube central tambien es compacta, pero aparecen mas outliers muy por encima de la diagonal. Esto coincide con el gran error de pico maximo positivo.
- En `H3_sinSF` y `H3_conSF`, la dispersion aumenta mucho. Los puntos moderados y altos se abren y los extremos dejan de seguir la diagonal.
- En `H6_sinSF`, la mayor parte de los puntos extremos quedan muy por debajo de la diagonal. El modelo casi siempre infraestima picos grandes.
- En `H6_conSF`, el patron es parecido: muchisimos puntos extremos muy bajos respecto al real, aunque el global mejora algo frente a `H6_sinSF`.

Conclusiones del scatter:
- La dispersion crece claramente con el horizonte.
- El patron de infraestimacion en picos si existe, sobre todo en `H3` y `H6`.
- Los puntos rojos extremos estan muy lejos de la diagonal en todos los modelos.
- El unico panel que transmite control visual razonable es `H1`.

## 10. Diagnostico: donde esta el cuello de botella
La evidencia conjunta sugiere que el cuello de botella principal no es el clasificador en `H1/H3`, sino la calibracion de magnitud del regresor. En `temporal_range_1.png` y en varios hidrogramas, los modelos suelen reaccionar cuando empieza la tormenta y el pico aparece aproximadamente en el momento correcto. Eso indica que la deteccion de evento existe. Lo que falla es cuanto sube la prediccion y cuanta cola deja despues.

Diagnostico puntual:
- En `H1` el clasificador parece suficientemente bueno para el TFM. El problema dominante es de magnitud en extremos.
- En `H3` el sistema aun detecta, pero la regresion ya suaviza o distorsiona bastante la amplitud.
- En `H6` probablemente fallan las dos cosas a la vez: el horizonte es demasiado largo para la senal disponible y se acumulan sesgo, smoothing y offset.
- Hay evidencia fuerte de problema de datos y de objetivo: solo hay 59 extremos, y ademas el analisis de extremos sin lluvia no es consistente entre contexto y outputs.
- No hay evidencia fuerte de que una arquitectura completamente nueva sea necesaria para cerrar el TFM. La arquitectura actual si aprende formas razonables a corto plazo.
- Si hubiera que apostar por una mejora tecnica concreta, la evidencia apunta mas a calibracion de loss/objetivo/threshold que a rediseino total del backbone.

Sobre la loss function:
- Si. Hay evidencia moderada de que la loss actual no penaliza suficiente el error de magnitud extrema, porque el timing esta razonablemente bien y aun asi el error extremo sigue siendo enorme.
- No hay evidencia de que cambiar la loss arregle por si solo `H6`.

Sobre la arquitectura:
- No hay evidencia fuerte de que el TCN sea incapaz de modelar la dinamica. En `H1` y algunos `H3` la forma del hidrograma si aparece.
- El problema parece mas de horizonte efectivo, desbalance y calibracion que de incapacidad estructural.

Sobre el threshold 0.3:
- No hay evidencia directa de que sea optimo.
- Tampoco hay evidencia directa de que sea el problema principal, porque los eventos grandes suelen ser detectados visualmente.
- Sin una curva precision-recall o una barrida de threshold, no se puede defender un cambio con seguridad.

## 11. Recomendaciones concretas

1. Usar `H1_sinSF` como modelo principal del TFM para resultados operativos y narrativa central.
Esfuerzo: bajo.
Impacto esperado: alto.
Motivo: es el mejor NSE global, el mejor en extremos y el mas estable visualmente.

2. Usar `H1_conSF` solo como comparacion secundaria, destacando que mejora leves/moderados pero empeora robustez del pico maximo.
Esfuerzo: bajo.
Impacto esperado: medio.
Motivo: aporta discusion metodologica interesante, pero no deberia ser el headline model.

3. No invertir tiempo en reentrenar todos los horizontes solo para "salvar" `H6`.
Esfuerzo evitado: alto.
Impacto de insistir en H6: bajo a medio.
Motivo: la evidencia actual muestra deterioro demasiado fuerte a 30 min.

4. Si queda tiempo para una unica mejora tecnica, hacer una calibracion ligera del threshold del clasificador para `H1` y `H3`, sin cambiar arquitectura.
Esfuerzo: bajo.
Impacto esperado: medio.
Motivo: es la modificacion mas barata con posibilidad real de mejorar onset/falsas alarmas, y no exige reentrenar todo si se puede evaluar post hoc.

5. Si queda tiempo para una segunda mejora tecnica, probar un ajuste focalizado de loss para penalizar mas extremos en `H1`, no en toda la familia de modelos.
Esfuerzo: medio.
Impacto esperado: medio.
Motivo: los errores extremos siguen siendo el principal dolor, pero no hay garantia de mejora grande.

6. Documentar explicitamente la discrepancia sobre "extremos sin lluvia" como limitacion experimental, no esconderla.
Esfuerzo: bajo.
Impacto esperado: alto.
Motivo: es una observacion metodologica importante y el mentor la va a detectar.

## 12. Material listo para el TFM
Directamente usables:
- `predictability_curve.png`
- `scatter_all_models.png`
- `nse_por_rango.png`
- `temporal_range_1.png`
- `comparison_sf_vs_nosf_*.png`
- `hydrograph_moderado_*.png`
- `hydrograph_alto_*.png`
- `hydrograph_extremo_lluvia_*.png`

Utiles pero con mas valor diagnostico que estetico:
- `extreme_rain_vs_norain.png`
- `extremos_H*_*.png`

Que necesitan mejoras esteticas si fueran a tesis final:
- Reducir saturacion visual en hidrogramas de 6 lineas.
- Mover leyendas a un formato mas compacto.
- Homogeneizar escalas y margenes en comparaciones para lectura mas limpia.
- En `scatter_all_models.png`, fijar limites comparables por fila o anotar mejor el rango visible de cada panel.

Analisis adicionales que harian falta para cerrar resultados:
- Barrida de threshold del clasificador para `H1` y `H3`.
- Metricas por evento, no solo por timestep, especialmente error en magnitud y delay del pico.
- Aclarar por que el nuevo output no reproduce los supuestos extremos sin lluvia.
- Una tabla final de "mejor modelo por criterio" para que el lector vea rapido el trade-off entre NSE global, error de pico y comportamiento extremo.

## Conclusiones finales
El proyecto ya tiene una historia tecnica clara para el TFM: a 5 minutos el sistema es util y aprende dinamica razonable; a 15 minutos sigue siendo interpretable pero ya no confiable para magnitud precisa; a 30 minutos la degradacion es demasiado fuerte. El principal cuello de botella no parece ser "no detectar tormentas", sino calibrar la magnitud de los picos, sobre todo extremos. `H1_sinSF` es la mejor opcion para presentar como modelo principal. La principal alerta metodologica no esta en las figuras, sino en la inconsistencia entre el contexto del proyecto y los outputs actuales sobre extremos sin lluvia.
