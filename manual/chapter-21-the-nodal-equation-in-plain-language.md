# Capítulo 21

## La ecuación nodal, en palabras sencillas

> «Una ecuación precisa una relación; también permite ver qué falta por conocer.»

El capítulo anterior separó capacidad y presión. TNFR las relaciona con la
velocidad de cambio de la forma mediante su ecuación nodal:

$$\frac{\partial\mathrm{EPI}}{\partial t}
=\nu_f\,\Delta\mathrm{NFR}.$$

**En palabras:** velocidad de cambio de la forma igual a capacidad de
reorganización por presión estructural.

El símbolo de la izquierda indica una velocidad instantánea, no la forma misma.
En una coordenada escalar, una presión negativa hace disminuir EPI cuando la
capacidad es positiva.

## Lo que dice el producto

En esta ley sin término aditivo, si νf es cero, la velocidad de EPI es cero.
Si ΔNFR es cero, también lo es. Si ambos son distintos de cero, EPI cambia.

Esta propiedad ayuda a leer el producto, pero no demuestra que la multiplicación
sea la única relación posible. Las funciones matemáticas xy y x²y, por ejemplo,
se anulan cuando cualquiera de sus variables se anula. La proporcionalidad
concreta forma parte de la relación declarada por TNFR; las analogías anteriores
no bastan para derivar su unicidad.

Si EPI se mide en una unidad de forma X y el tiempo en T, νf tiene unidades
T⁻¹ y la presión tiene unidades X. Así el producto tiene unidades X/T. Una
presión tangente no puede sustituirse sin más por una distancia sin signo.

## Lo que todavía no dice

Conocer el producto no determina por separado los dos factores. Para predecir
necesitamos especificar de qué depende la presión y cómo evolucionan capacidad,
fase y conexiones, además del estado inicial.

Tampoco confunde un instante quieto con un equilibrio duradero. Una fase puede
seguir cambiando y generar presión después, aunque ahora EPI tenga velocidad
cero. Una ley con una fuerza aditiva sería otro modelo declarado: en él, νf=0
no bastaría para impedir cualquier cambio de EPI.

Los operadores que veremos más adelante pueden representar eventos que cambian
el estado. Sus saltos no son automáticamente soluciones de una evolución
continua durante un intervalo.

## ¿Y ahora qué?

Podemos intentar construir interpretaciones de fenómenos conocidos, pero aún no
hemos probado la ecuación en una llama, un columpio o una bandada.

**¿Cómo distinguimos una historia compatible con la fórmula de una predicción
que realmente la pone a prueba?**

## Ideas clave

- La ecuación relaciona velocidad de EPI, capacidad y presión.
- Anular un factor anula este producto; no congela necesariamente el sistema completo.
- Las condiciones de anulación no seleccionan una única ley matemática.
- Para predecir faltan las dependencias y evoluciones de los términos.

## La idea que nos llevamos

La ecuación nos da una relación precisa y una lista de preguntas que no podemos saltarnos.

## Experimento de observación

En el ejemplo matemático del capítulo anterior, elige una capacidad positiva y
calcula la velocidad inicial de cada nodo. Después duplica todas las capacidades,
manteniendo las presiones: ¿qué debería pasar con esas velocidades?

Compara esto con el caso de presión cero. Explica por qué esas cuentas describen
un instante y no garantizan por sí solas el comportamiento futuro.

1. ¿Qué inputs conocías antes de calcular?
2. ¿Qué ley necesitarías para dar un paso más?
3. Completa: «Una velocidad inicial no demuestra todavía...».

Los [fundamentos](../theory/FUNDAMENTAL_THEORY.md) desarrollan la representación
y las condiciones necesarias para cerrar un modelo.
