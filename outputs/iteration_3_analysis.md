# Analisis de la iteracion 3

## Resumen ejecutivo

La iteracion 2 mejoro respecto a la iteracion 1, pero el problema central no quedo resuelto:

1. El gating si redujo falsas alarmas, pero lo hizo comprimiendo demasiado la magnitud de eventos severos.
2. La cabeza de evento aprendio una tarea demasiado amplia: "estar dentro de un intervalo de evento", no "estar en una fase de stormflow materialmente alta".
3. La rama de magnitud no estaba siendo supervisada de forma directa; solo recibia gradiente a traves de la salida ya gateada.
4. El entrenamiento se detuvo demasiado pronto para un problema raro, ruidoso y con validacion de alta varianza.

El patron observado en test encaja exactamente con esa combinacion:

- sobreestimacion sistematica cerca de cero y en eventos pequenos/moderados
- discriminacion insuficiente entre niveles intermedios
- subestimacion brutal en eventos extremos reales
- una metrica global de pico que puede parecer buena aunque el modelo falle el pico correcto o falle su timing

## Que salio mal y por que

### 1. El gating multiplico la salida final y aplasto picos reales

El modelo actual produce:

`stormflow_prediction = magnitude_prediction * event_probability`

Ese diseno tiene una ventaja clara: cuando `event_probability` baja, las falsas alarmas se reducen. El problema es que tambien introduce un cuello de botella fuerte para los picos reales.

Si la rama de magnitud estima un evento severo, pero la rama de evento queda en una probabilidad intermedia como 0.20-0.40, la salida final cae inmediatamente al 20-40% de la magnitud potencial. Eso explica muy bien casos como:

- pico real de `135.2 MGD`
- forma temporal razonablemente capturada
- magnitud final predicha de solo `29.4 MGD`

El modelo no necesariamente "ignoro" el evento. Es mas probable que haya aprendido parte de la forma, pero la probabilidad de evento no se saturara y la multiplicacion final destruyera la amplitud.

### 2. La cabeza de evento estaba mal alineada con la decision operativa real

Actualmente la supervision de evento viene de `is_event`, que marca si el timestamp cae dentro de un intervalo de evento.

Eso no equivale a:

- "hay stormflow relevante ahora"
- "estamos cerca del pico"
- "la magnitud futura amerita activacion operativa"

Dentro de un mismo evento hay muchas muestras de cola o transicion donde el `stormflow` real es bajo. Al tratarlas igual que el nucleo del evento:

- la cabeza de evento aprende una nocion demasiado amplia de positivo
- el gating puede quedar encendido cuando el target real es pequeno
- aparecen sesgos positivos en `base`, `pequeno` y `moderado`
- la separacion entre moderado, grande y extremo se vuelve borrosa

Eso coincide con tus observaciones:

- columna vertical en `x ~= 0`
- bias positivo creciente hasta `moderado`
- inversion brusca a infraestimacion en `extremo`

### 3. La rama de magnitud no estaba siendo supervisada directamente

La loss actual compara el target continuo contra la salida final gateada, no contra `magnitude_prediction`.

Consecuencia:

- si el gate se queda corto, la magnitud tambien recibe gradiente a traves de una salida ya comprimida
- la rama de regresion no aprende de manera limpia "cuanto deberia valer el pico si el evento si ocurre"
- el modelo mezcla dos errores distintos dentro de un solo canal: activacion y amplitud

En otras palabras:

- la cabeza de evento responde a "hay algo parecido a evento"
- la cabeza de magnitud intenta corregir una salida que ya fue recortada

Eso hace mucho mas dificil aprender extremos.

### 4. El entrenamiento se corto demasiado pronto

Tus observaciones sobre las curvas son importantes:

- solo `11` epochs
- mejor epoch `6`
- gap `train/val` moderado
- sin evidencia fuerte de overfitting severo

Eso sugiere que el early stopping actual esta siendo demasiado agresivo para una validacion ruidosa. En problemas muy desbalanceados, la `val_loss` puede oscilar varias epochs aunque el modelo todavia este mejorando en representaciones utiles.

El trainer actual corta con:

- paciencia relativamente corta
- `min_delta` estricto
- sin un `min_epochs` que fuerce una fase minima de aprendizaje

Resultado:

- el modelo apenas entra en regimen util
- no tiene tiempo suficiente para afinar cola alta y calibracion del gate

### 5. Los pesos por muestra siguen elevando demasiado "evento" generico

El pipeline actual da un aumento de peso a cualquier muestra con `event_array=True`, incluso si la magnitud futura sigue siendo baja.

Eso no es un error conceptual grave, pero en esta tarea si introduce un sesgo:

- el modelo ve demasiada presion para "encenderse" en intervalos de evento
- esa presion ayuda a recall, pero empeora calibracion fina en baseflow y severidad intermedia

Para esta iteracion conviene que la prioridad la marque mas la magnitud futura y menos la pertenencia bruta al intervalo.

### 6. Las features quedaron algo cortas para diferenciar intensificacion rapida

La simplificacion de features en la iteracion 2 redujo dependencia excesiva del target, lo cual fue una buena correccion. Pero el set actual quedo sin algunas senales causales que ayudan justamente a distinguir:

- lluvia sostenida vs lluvia que acelera
- respuesta hidrologica estable vs ascenso rapido

En un sistema con lag optimo de `10 min`, pequenas diferencias en la intensificacion reciente de lluvia pueden cambiar mucho la magnitud del pico futuro. Hoy el modelo ve acumulados y maximos, pero no ve tan explicitamente el cambio reciente de esa lluvia.

## Por que la metrica global de pico puede estar enganando

`test_metrics.json` reporta un pico global casi perfecto, pero tus graficos muestran un error severo en un pico extremo real. Eso indica que el maximo predicho global probablemente ocurrio:

- en otro instante
- en otro evento
- o como falsa alarma desplazada en el tiempo

Conclusiones:

- el pico global por si solo no basta
- el modelo puede "acertar el maximo numerico" y aun asi fallar el evento importante
- el problema real sigue siendo de magnitud y timing por evento

No voy a cambiar ahora toda la capa de evaluacion porque primero hace falta corregir el aprendizaje, pero esta observacion queda documentada: el criterio de exito no puede depender solo del pico maximo global.

## Cambios que voy a implementar

### A. Hacer el early stopping menos agresivo

Voy a introducir una fase minima de entrenamiento antes de permitir early stopping y voy a relajar los defaults para problemas de alta varianza.

Justificacion:

- el modelo actual se corta antes de tener tiempo real para aprender cola alta
- no hay evidencia de overfitting severo que justifique cortar tan pronto

### B. Supervisar la rama de magnitud de forma directa

Voy a agregar una componente de loss que compare `magnitude_prediction` contra el target continuo en muestras donde realmente hay respuesta de stormflow relevante.

Justificacion:

- separa mejor "cuanto vale el evento" de "cuan seguro estoy de que hay evento"
- evita que la rama de magnitud aprenda solo a traves de una salida ya comprimida por el gate

### C. Redefinir dentro de la loss el concepto efectivo de evento

No voy a confiar ciegamente en `is_event` como supervision positiva total. Voy a construir una mascara efectiva de evento que combine:

- pertenencia al intervalo de evento
- magnitud real por encima de un umbral base
- prioridad adicional para cola alta

Justificacion:

- reduce positivos "demasiado faciles" en colas de evento con stormflow casi nulo
- alinea mejor la cabeza de evento con la necesidad operativa real

### D. Hacer el gate menos propenso a aplastar extremos

Voy a suavizar la conversion de logit a probabilidad usando una temperatura del gate.

Justificacion:

- mantiene el beneficio del gating para falsas alarmas
- reduce la compresion excesiva cuando el clasificador esta inseguro pero el evento si es real

### E. Recalibrar pesos de muestra

Voy a bajar ligeramente el premio por "evento generico" y a concentrar mas el enfasis en cuantiles altos.

Justificacion:

- el sesgo positivo en pequenos/moderados sugiere que el modelo sigue demasiado predispuesto a activarse
- los pesos deben priorizar severidad real, no solo pertenencia a una ventana amplia de evento

### F. Agregar features causales de intensificacion de lluvia

Voy a sumar unas pocas features nuevas, fisicamente razonables y sin leakage:

- acumulado intermedio adicional para humedad reciente
- cambios de lluvia a muy corto plazo

Justificacion:

- los eventos moderados ya se predicen relativamente bien
- lo que falta es distinguir mejor cuando la lluvia reciente esta escalando hacia un pico fuerte

## Hipotesis operativa para la iteracion 3

Si estos cambios son correctos, deberian verse senales muy concretas:

1. entrenamiento mas largo que `11` epochs antes de detenerse
2. menor compresion de picos severos
3. menos bias positivo en `base`, `pequeno` y `moderado`
4. transicion de severidad mas suave entre `moderado`, `grande` y `extremo`
5. menor dependencia de una probabilidad de evento excesivamente amplia

No espero perfeccion inmediata en extremos, pero si espero que el error deje de tener esta forma tan especifica de "sobreestimar lo medio y aplastar lo extremo". Ese patron apunta a un problema estructural del acoplamiento entre gate, etiquetas de evento y supervision de magnitud, y esa es exactamente la parte que voy a corregir.
