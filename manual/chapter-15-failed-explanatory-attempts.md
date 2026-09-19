# Capítulo 15

## Lenguajes especializados

Imagina dos habitaciones conectadas por una puerta. Cada una tiene un termostato que regula su calefacción. Cuando la puerta está cerrada, podemos estudiar cada habitación casi por separado. Al abrirla, el intercambio de calor hace que lo que ocurre en una influya en la otra.

¿Nos basta con saber qué temperatura quiere mantener cada termostato? Nos falta cómo se transmite el calor. ¿Nos basta con dibujar una línea entre las habitaciones? Nos falta cuánto intercambian y cómo responden sus controles.

Necesitamos describir los estados, las conexiones y las reglas de cambio conjuntamente. Esto puede hacerse con herramientas de control y modelos dinámicos en red. Combinarlas no está prohibido ni requiere que hayan fallado antes.

Quédate un momento con la diferencia entre dibujar una conexión y explicar su efecto.

Una línea en un mapa dice que dos lugares están relacionados. Una regla de evolución dice qué cambia debido a esa relación. El mapa y la regla cumplen funciones distintas que pueden trabajar juntas.

## Elegir la pregunta

Las herramientas se eligen según lo que queremos averiguar. Para estudiar si la temperatura vuelve a un intervalo deseado, necesitamos analizar la respuesta del sistema. Para saber qué habitaciones intercambian calor, necesitamos su red de conexiones. Para investigar oscilaciones, también importan los tiempos de respuesta y las condiciones del acoplamiento.

Una misma investigación puede necesitar las tres cosas. Ninguno de esos ejemplos demuestra que exista una única manera de describirlas ni que todas las redes tengan el mismo comportamiento.

La pregunta que guía este libro es cómo describir patrones que conservan rasgos reconocibles mientras cambian e interactúan. Los modelos que ya existen pueden responder partes de esa pregunta, e incluso combinar esas partes. TNFR propone una organización concreta de variables y transformaciones que tendremos que evaluar por lo que consiga calcular y predecir.

## Un dibujo que empieza a cambiar

Haz un dibujo de las dos habitaciones. Anota una temperatura inicial en cada una y marca si la puerta está abierta. Ahora imagina dos modelos: uno supone que el intercambio es rápido; el otro, que es lento.

El dibujo de conexiones puede ser idéntico, pero la evolución será diferente. La estructura de la red aporta información; las reglas y sus parámetros aportan otra. Confundir ambas cosas nos impediría distinguir los modelos.

Este ejemplo es una comparación pedagógica. Todavía no hemos definido una medición de EPI, capacidad o presión para esas habitaciones. Para probar una explicación TNFR necesitaríamos hacerlo antes de comparar una predicción con datos reservados.

## ¿Y ahora qué?

Ya podemos formular una exigencia precisa para el marco que vamos a estudiar: sus variables, conexiones y leyes deben describirse de forma compatible. Después tendremos que comprobar qué consecuencias se deducen de ellas y cuáles dependen de decisiones adicionales.

La siguiente pregunta es qué podemos pedirle a esa propuesta sin dar su éxito por supuesto.

## Ideas clave

- Un mapa de conexiones y una ley de evolución contienen información diferente.
- Los modelos de control y de redes pueden combinarse.
- La misma red admite dinámicas distintas.
- Una analogía ayuda a formular preguntas, pero no valida una teoría.
- TNFR debe mostrar qué calcula, qué supone y qué puede contrastarse.

## La idea que nos llevamos

Comprender un patrón exige saber cómo está organizado y cómo cambia.

## Experimento de observación

En el dibujo de las dos habitaciones, predice qué cambiaría al abrir la puerta y qué información te falta para calcular cuánto tardaría. No hace falta manipular aparatos: compara las predicciones de tus dos modelos imaginarios.

Después distingue qué has supuesto, qué has deducido y qué tendrías que medir. ¿Podrían dos reglas diferentes coincidir al principio y separarse más tarde? ¿Qué observación permitiría distinguirlas?

Para el protocolo técnico de contraste, véase el [plan de investigación](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md).
