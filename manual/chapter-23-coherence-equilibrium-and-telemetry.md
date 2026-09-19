# Capítulo 23

## Coherencia, equilibrio y telemetría

> «Una señal resume algo del sistema; no contiene todo lo que el sistema puede hacer.»

El tablero de un vehículo muestra unas pocas señales. Son útiles porque responden
a preguntas concretas. Ninguna lectura aislada permite conocer todos los fallos
posibles ni predecir cualquier viaje.

En TNFR ocurre algo parecido. **Telemetría** es la información que registramos
para examinar un estado o una ejecución. Medirla no es lo mismo que producir la
dinámica que medimos.

## El indicador C(t)

Para una red no vacía, el motor define su coherencia global como

$$C(t)=\frac{1}{1+\overline{|\Delta\mathrm{NFR}|}+\overline{|d\mathrm{EPI}|}}.$$

Las barras superiores indican promedios sobre nodos. En este indicador, presión
y cambio registrados de menor magnitud dan una lectura mayor. No es, en general,
el promedio de aplicar la misma fórmula por separado a cada nodo.

Es una definición diagnóstica con escalas numéricas elegidas, no una fórmula
derivada de manera única de la ecuación nodal. Comparar redes medidas en escalas
distintas requiere justificar la normalización.

Además, el lector del motor consulta presión y velocidad almacenadas. No las
refresca automáticamente ni prueba que procedan del mismo instante. Sin esa
procedencia, un número correcto sobre los datos guardados puede describir mal
el estado actual.

## Estar quieto ahora no garantiza seguir igual

Presión y velocidad nulas dan C=1 en ese registro. Eso no demuestra que la fase,
las capacidades o los enlaces permanezcan fijos, ni que una perturbación futura
no cambie la forma.

Un C pequeño indica magnitudes grandes según esta escala, no una fragmentación
inevitable. Un C alto tampoco prueba identidad, autosostenimiento o validez de
una interpretación de laboratorio.

## El índice de sentido y la tétrada

**Si**, el índice de sentido, combina señales normalizadas de capacidad, fase y
presión con pesos configurados. Puede resultar útil para comparar observaciones
dentro de un protocolo. No es por sí solo un pronóstico validado de bifurcaciones
ni una ley que determine qué operador debe aparecer.

Algunos controladores lo utilizan para elegir acciones. En tal caso hay una
decisión programada que debe declararse; calcular el indicador no convierte esa
decisión en una propiedad emergente demostrada.

La **tétrada** aporta otras cuatro lecturas: potencial estructural, gradiente de
fase, curvatura de fase y longitud de coherencia. Conserva información distinta,
pero tampoco constituye una descripción completa del estado. Una medida
indefinida o una estimación no disponible debe señalarse, no sustituirse por un
cero favorable.

## ¿Y ahora qué?

**¿Qué acciones implementa el motor, y qué diferencia hay entre observarlas,
invocarlas y demostrar que emergen por sí solas?**

## Ideas clave

- C y Si son diagnósticos con definiciones y escalas declaradas.
- Un registro almacenado no certifica que sus entradas estén actualizadas.
- Ni un score ni la tétrada garantizan toda la evolución futura.
- Un controlador que usa telemetría añade una política explícita.

## La idea que nos llevamos

Leer bien una señal incluye saber qué pregunta no puede responder.

## Experimento de observación

Recupera una observación cotidiana del libro. Enumera señales que podrías medir
y explica qué información perdería cada resumen. No asignes un C o Si numérico
a una impresión cualitativa si no has medido sus entradas.

1. ¿Qué datos necesitarías para calcular C?
2. ¿Podrían dos estados distintos tener el mismo resumen?
3. Completa: «Esta señal me permite afirmar... pero no...».

El [alcance de los diagnósticos](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md),
[su kernel](../src/tnfr/metrics/common.py) y
[la tétrada](../docs/STRUCTURAL_FIELDS_TETRAD.md) contienen las definiciones.
