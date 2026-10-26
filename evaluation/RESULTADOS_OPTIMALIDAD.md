# Evaluacion del esfuerzo de prueba de optimalidad

Estado comprobado: 2026-10-26. Las cuatro rondas han terminado. El objetivo del 90 % fuera del entrenamiento no se ha alcanzado. Se conserva el perfil recomendado anterior; los modelos experimentales se guardan por separado.

## Resultados comparables

Las nuevas rondas usan las mismas 272 instancias y cinco particiones externas estratificadas (semilla 20261024), con tres particiones internas para seleccionar configuraciones. No son un test externo: esta cohorte ya se ha examinado en numerosas busquedas. El 84,19 % del perfil anterior procede de una ablacion exploratoria seleccionada posteriormente y tampoco constituye validacion externa.

| Ronda | Acierto | Macro-F1 | Easy correctas |
|---|---:|---:|---:|
| Sondas de prueba de optimalidad | 82.72 % | 0.576 | 1/15 |
| Umbrales ordinales | 83.82 % | 0.603 | 1/15 |
| Relajaciones por maquina | 83.82 % | 0.616 | 1/15 |
| Sondas y relajaciones combinadas | 83.09 % | 0.577 | 1/15 |

## Que se ha probado

La primera ronda incorpora esfuerzo determinista hasta primera solucion y ultima mejora, integral primal, progreso del incumbente, cota, gap, conflictos, propagaciones, ramas y prueba de optimalidad. Compara una busqueda CP-SAT de presupuesto 10 con una de propagacion disyuntiva fuerte de presupuesto 1. Son unidades deterministas del solver, no segundos de pared. Se utilizaron hasta 12 procesos entre las ejecuciones; cada solver y cada ajuste interno usa un solo hilo.

La segunda ronda aprende cuatro decisiones acumuladas entre clases y proyecta sus probabilidades para mantener el orden. No asigna un tiempo numerico inventado a open. La tercera agrega cotas de subproblemas no preemptivos por maquina, con presupuesto determinista 0,02 por maquina. Usa cotas inferiores demostradas, nunca incumbentes de una relajacion como si fueran cotas inferiores. La cuarta combina ambos bloques de variables.

## Por que no basta con el tiempo de una sonda

JSPLIB etiqueta usando el tiempo necesario para probar optimalidad con alguno de sus motores de referencia. Varias etiquetas easy del snapshot se apoyan en OptalCP. Las sondas locales de CP-SAT no son ese mismo protocolo ni comparten necesariamente hardware, configuracion o motor. El tiempo historico conocido no entra como caracteristica: no esta disponible para una instancia nueva.

En estas sondas se probaron 71 de 154 toy y ninguna de las otras clases. Los gaps medianos crecen: toy 0,0240; easy 0,1113; medium 0,1287; hard 0,1662; open 0,2017. Es una asociacion descriptiva, no una regla suficiente para separar clases. El propio desajuste en toy muestra que un timeout local no permite cambiar la etiqueta a open.

Solo hay 15 easy. Las nuevas rondas aciertan uno de ellos; sus matrices de confusion muestran dispersion hacia varias clases. No se demuestra que el 90 % sea imposible: estas cuatro alternativas no lo alcanzan. Para una siguiente evaluacion independiente hace falta ampliar las etiquetas verificadas, especialmente las clases intermedias, y contrastar motores bajo un protocolo comparable. Repetir ajustes sobre esta cohorte no proporciona una validacion nueva.

## Artefactos y comprobaciones

Cada carpeta conserva plan, procedencia de variables, particiones, predicciones fuera del entrenamiento, metricas y referencia reajustada. Los modelos nuevos se exportan para la API ordinaria de CEJSP; el manifiesto permite repetir el ajuste. Las exportaciones comprueban equivalencia de votos con el experimento y persistencia. Las pruebas de cotas usan enumeraciones independientes en problemas pequenos.

La bateria completa actual se registra en [proof-validation-final.log](proof-validation-final.log). Los modelos de sondas y de relajaciones pasan la equivalencia JSON/DZN/Taillard y la invariancia por lotes; sus comprobaciones estan en format-check.json dentro de cada carpeta. No se ha publicado ni hecho push.

## Reproducir una ronda

Los scripts requieren el codigo de pypi/src en PYTHONPATH y las dependencias del manifiesto del modelo. Para repetir sin sobrescribir referencias, indicar otra carpeta de salida:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'pypi/src'
.\.venv\Scripts\python.exe evaluation/study_cumulative_effort.py --features models/jsplib-bottleneck-v11/features.json --workers 12 --out models/nueva-evaluacion
```
