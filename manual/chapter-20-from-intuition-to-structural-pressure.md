# Capítulo 20

## De la intuición a la presión estructural

> «Una diferencia puede orientar un cambio; falta precisar qué diferencia y con qué ley.»

En los vasos comunicantes, comparar niveles ayuda a pensar en una respuesta con
dirección. En TNFR, la **presión estructural**, ΔNFR, también tiene dirección en
el espacio donde representamos la forma. En una coordenada escalar puede tener
signo positivo o negativo.

No es necesariamente una presión mecánica ni una cantidad que podamos identificar
con el combustible, la fuerza o la tensión social sin construir antes un modelo.

## Un ejemplo calculable

Tomemos solo el canal de EPI. En un nodo conectado, este canal compara el promedio
ponderado de EPI en sus vecinos con el EPI propio:

$$\Delta\mathrm{NFR}_{\mathrm{EPI},i}
=\overline{\mathrm{EPI}}_{\mathcal N(i)}-\mathrm{EPI}_i.$$

Si el promedio de sus vecinos es 3 y el nodo está en 1, la contribución es +2.
Si está en 5, la contribución es −2. Los dos casos tienen igual magnitud y
sentidos contrarios. Son números de un ejemplo matemático, no mediciones de agua.

El motor tiene además canales asociados a fase, capacidad y topología. La presión
total combina sus contribuciones con coeficientes declarados. Igualar EPI con
los vecinos puede anular el canal de EPI sin anular la presión total.

No todas las contribuciones usan los pesos de la misma manera. Por eso debemos
reutilizar la definición del canal que estudiamos, sin sustituirla por una imagen
vaga de «diferencia con el entorno».

## Presión y capacidad no son intercambiables

Con una presión conocida, νf indica cuánto cambio de EPI corresponde por unidad
de tiempo. Con la misma capacidad, cambiar el signo de la presión cambia el
sentido de la respuesta escalar.

Estrechar el tubo de nuestra analogía altera una interacción. Eso no demuestra
que hayamos modificado precisamente νf: también podría corresponder a un peso de
enlace en un modelo elegido. Para distinguir ambas posibilidades hacen falta
definiciones y medidas independientes.

La presión debe calcularse desde los inputs declarados antes de comprobar una
predicción. Si primero observamos cómo cambió EPI y luego definimos la presión
para que encaje, hemos reconstruido una igualdad; no hemos probado una ley.

## ¿Y ahora qué?

**¿Qué relación propone TNFR entre forma, capacidad y presión?**

Tenemos el vocabulario para escribirla. Aún faltarán leyes para especificar todos
sus términos y, cuando proceda, su evolución.

## Ideas clave

- ΔNFR es una respuesta orientada; su signo importa.
- «Promedio de vecinos menos valor propio» define un canal concreto, no toda presión posible.
- Los demás canales y sus coeficientes deben declararse.
- Una predicción requiere calcular la presión sin ajustarla al resultado reservado.

## La idea que nos llevamos

Poner nombre al empuje no basta: necesitamos calcularlo de forma independiente.

## Experimento de observación

En papel, dibuja tres nodos en fila y asigna EPI 0, 1 y 4. Usa pesos iguales y
solo el canal de EPI. Predice el signo de la presión en cada nodo antes de
calcular los promedios.

Después cambia únicamente el EPI central y repite.

1. ¿Qué presiones cambiaron?
2. ¿Qué información no aparece en este ejemplo, como fase o fuentes?
3. Completa: «Mi cálculo predice este canal bajo la condición de que...».

Las [bases nodales](../theory/NODAL_PARAMETER_FOUNDATIONS.md) y
[el cálculo de presión](../src/tnfr/dynamics/dnfr.py) especifican los canales.
