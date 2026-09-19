# Capítulo 25

## La gramática de la coherencia

> «Ordenar acciones y garantizar su resultado son tareas distintas.»

Una receta ayuda a recordar que el orden importa. Tener los mismos ingredientes
no asegura el mismo resultado si cambiamos los pasos. Pero una receta bien
escrita tampoco garantiza que cualquier cocina, temperatura o ingrediente dé
siempre un buen plato.

La **gramática TNFR** organiza la admisión de palabras de operadores y parte de
su contexto. Sus reglas se motivan en requisitos estructurales y se implementan
como contratos. No son una prueba de que cualquier secuencia aceptada termine
en equilibrio o pueda sostenerse indefinidamente.

## Empezar, continuar y cerrar

**U1: inicio y cierre.** Una palabra completa usa los conjuntos admitidos de
iniciadores y cierres, con el contexto que corresponda. Iniciar una palabra no
significa crear sustrato desde la nada; terminarla no certifica un atractor.
Un fragmento de una ejecución tiene que identificarse como tal.

**U2: estabilización y deuda.** Determinados operadores clasificados como
desestabilizadores requieren cobertura de estabilización. La deuda y sus límites
son parte de la política del motor. La regla pretende organizar riesgos; no es
un teorema universal de convergencia.

**U3: compatibilidad circular.** Acoplamiento y Resonancia exigen que la separación
circular de fase cumpla el límite admitido. Esta comprobación no demuestra que
las fases permanezcan compatibles después ni proporciona su ley de movimiento.
La temperatura de una salsa no es, por analogía, una fase TNFR medida.

**U4: contexto de transformaciones.** Se comprueban condiciones para bifurcaciones
y para Mutación, como historia reciente y preparación requerida. La evidencia
temporal del estado y el contexto gramatical son comprobaciones distintas.

**U5: recursión y escala.** La ejecución comprueba profundidad declarada y cobertura
de estabilización cercana cuando corresponde. Anidar nodos no demuestra, por sí
solo, que toda parte sea coherente ni que la misma ley cierre entre escalas.

**U6: vigilancia geométrica.** Observa la deriva del potencial respecto a una
referencia mediante una política de seguridad. Es una comprobación diagnóstica
distinta de ordenar palabras. Sus umbrales no son límites universales derivados
solo del carácter circular de la fase.

## ¿De dónde vienen las reglas?

La ecuación nodal permite calcular el cambio acumulado si conocemos capacidad
y presión. Para demostrar estabilidad hacen falta además la ley, las condiciones
y un argumento adecuado. Acumular cambio durante un intervalo, estar acotado y
converger a largo plazo son propiedades diferentes.

Las motivaciones estructurales no seleccionan de manera única cada conjunto de
operadores, ventana temporal o umbral de la gramática. La revisión actual
distingue estas decisiones de los resultados que sí están derivados. También
hay flujos difusivos que relajan sin una secuencia de estabilizadores nombrados.

## ¿Y ahora qué?

**¿Cómo usamos estos contratos sin convertirlos en afirmaciones mayores que la
evidencia disponible?**

## Ideas clave

- La gramática admite palabras y contextos; no prueba toda trayectoria futura.
- U1 no describe creación desde la nada ni un estado final autosostenido.
- U2–U5 tienen condiciones implementadas que debemos comprobar por separado.
- U6 vigila una magnitud; una alerta no es una ley autónoma.
- La ecuación nodal sola no deriva de manera única todas estas decisiones.

## La idea que nos llevamos

Una secuencia puede estar bien formada y aun necesitar una prueba de estabilidad.

## Experimento de observación

Elige una receta conocida. Antes de analizarla, predice qué cambio de orden
alteraría el resultado. Después separa dos cosas: seguir las instrucciones y
comprobar el resultado obtenido.

La receta es una analogía del orden, no un experimento que valide U1–U6.

1. ¿Qué condición inicial deja implícita la receta?
2. ¿Qué comprobarías al terminar, aunque hubieras seguido todos los pasos?
3. Completa: «Admitir los pasos no me permite garantizar...».

La [especificación gramatical](../theory/UNIFIED_GRAMMAR_RULES.md) y
[su alcance](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) son los propietarios
de las reglas detalladas.
