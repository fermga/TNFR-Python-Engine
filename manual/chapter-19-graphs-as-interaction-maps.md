# Capítulo 19

## Grafos como mapas de interacción

> «El mapa indica por dónde puede pasar una interacción; la ley indica qué pasa.»

Imagina varios recipientes unidos por tubos. Un dibujo de círculos y líneas
permite señalar qué recipientes están conectados directamente. Ese dibujo es
un ejemplo de **grafo**.

En una analogía de vasos comunicantes podemos observar cómo se redistribuye el
agua. Pero el dibujo no contiene toda la explicación: también importan las
condiciones del experimento y la respuesta de los tubos y recipientes.

La misma distinción sirve para TNFR. Saber quién está conectado no basta para
decidir cómo cambiará la red.

## Cuatro nombres nuevos

**Nodo.** Una parte representada con sus atributos.

**Enlace.** Una conexión entre dos nodos. Puede llevar datos como un peso.

**Vecindario.** Los nodos conectados directamente a uno dado.

**Grafo.** El conjunto de nodos y enlaces, con la información que declaramos.

Tres círculos en fila tienen dos enlaces. Los extremos no son vecinos directos,
aunque existe un camino entre ellos. Eso distingue una interacción local de un
efecto que llega a través de varios pasos.

Un peso de transporte y una longitud geométrica responden a preguntas distintas.
No debemos tratarlos como si fueran la misma cantidad sin declarar esa elección.

## Una red no obliga a igualarse

Hay un modelo concreto del motor en el que EPI cambia hacia el promedio ponderado
de sus vecinos. Es el canal difusivo de EPI. En ese modelo, la diferencia local
determina una parte de la presión.

Con soporte fijo, conectado por enlaces de peso positivo, pesos recíprocos no
negativos y capacidades positivas constantes, ese flujo puro de EPI se relaja
hacia una forma uniforme.
Si hay componentes desconectadas, cada una puede terminar con su propio valor.

La conclusión necesita esas condiciones. Un grafo por sí solo también puede
acompañar otras leyes, fuentes o cambios de conexión. La fase, la capacidad y
los demás canales pueden producir respuestas que no consisten en igualar EPI.

Por eso un rumor o un atasco son imágenes para pensar en conexiones, no pruebas
de que toda red obedezca la misma difusión. Tampoco basta contar vecinos para
ordenar siempre la rapidez de respuesta.

## ¿Y ahora qué?

**En el modelo elegido, ¿cómo calculamos la presión antes de observar el cambio?**

## Ideas clave

- El grafo representa conexiones, no una ley de evolución completa.
- Vecindad directa y distancia a través de caminos son conceptos distintos.
- La relajación a un campo uniforme es un resultado condicionado del canal puro de EPI.
- Cambiar una ley o un canal puede cambiar el resultado sobre el mismo grafo.

## La idea que nos llevamos

Para explicar una red necesitamos tanto el mapa como la regla de cambio.

## Experimento de observación

Dibuja tres o cuatro recipientes conectados, o una red cotidiana que conozcas.
Antes de observarla, escribe qué relaciones son directas.

Después distingue lo que permite afirmar el dibujo de lo que requeriría medir
una interacción. Si observas vasos comunicantes, registra niveles y condiciones;
no identifiques automáticamente el nivel con EPI ni el tubo con νf.

1. ¿Qué conexiones quedan fuera de tu dibujo?
2. ¿Qué cantidad necesitas medir para predecir una respuesta?
3. Completa: «Conocer los enlaces todavía no me dice...».

El [resultado de difusión](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md) conserva
las hipótesis de la afirmación matemática.
