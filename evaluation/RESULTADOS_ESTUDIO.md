# CEJSP: evaluación científica y de rendimiento

Fuente comprobada: `evaluation/resultados-20261022-184844/`. Ejecución completada sin errores, con las 33 pruebas del paquete y cinco pruebas del estudio aprobadas.

## Conclusión

Se confirma el ahorro de entrenar una sola vez por pliegue para calcular métricas y predicciones. No se confirma superioridad predictiva de CEJSP inductivo: en esta cohorte su MAE es mayor que el de un Random Forest que recibe únicamente jobs, máquinas y operaciones. Las características de grafo tampoco muestran una mejora clara de MAE frente a ese control.

## Datos y protocolo

Se comprobaron los 2401 pares DZN/JSON originales, incluidas factibilidad y coherencia del objetivo de los schedules. Hay 707 estados OPTIMAL_SOLUTION y 1694 SATISFIED; la factibilidad no demuestra optimalidad. Los baselines de tamaño sobre los 2401 casos se conservan en `size_baseline_summary.json`.

El estudio con grafos utiliza 196 instancias, jobs y máquinas entre 2 y 15, con tres semillas (11, 29, 47), cinco pliegues por semilla y 100 árboles. La identidad base se verifica por contenido y no cruza los pliegues. Se obtienen tres predicciones fuera de muestra por instancia: 588 predicciones no son 588 instancias independientes. Los intervalos pareados promedian primero el error de cada instancia entre repeticiones.

El objetivo principal es `clip(solveTime/maxTime,0,1)`: tiempo observado como fracción del presupuesto, no tiempo latente hasta demostrar optimalidad. La variable secundaria combina señales del solver sin tamaño de modelo, calibradas dentro de entrenamiento; es un proxy construido.

## Precisión fuera de muestra

Métricas agregadas de las predicciones de los 15 pliegues, objetivo principal:

| Método | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mediana de entrenamiento | 0.061491 | 0.175199 | -0.136027 |
| RF de tamaño | 0.059919 | 0.139561 | 0.279134 |
| RF de grafo | 0.060409 | 0.132872 | 0.346582 |
| CEJSP inductivo | 0.097217 | 0.163889 | 0.005910 |

La menor RMSE y mayor R² del RF de grafo coexisten con un MAE ligeramente mayor que el RF de tamaño. No hay un ganador uniforme entre métricas.

El contraste predefinido es MAE del RF de tamaño menos MAE del modelo; valores positivos favorecen al modelo. Bootstrap pareado de 2000 remuestreos, por instancia base:

| Objetivo | Modelo | Diferencia de MAE | IC 95 % |
|---|---|---:|---|
| Fracción de tiempo | RF de grafo | -0.000491 | [-0.004546, 0.003607] |
| Fracción de tiempo | CEJSP inductivo | -0.037298 | [-0.049282, -0.025899] |
| Proxy ponderado | RF de grafo | 0.001087 | [-0.001817, 0.004041] |
| Proxy ponderado | CEJSP inductivo | -0.031921 | [-0.046345, -0.019159] |

Los intervalos del RF de grafo incluyen cero; no establecen una mejora clara de MAE. Los de CEJSP inductivo están por debajo de cero en ambos objetivos. No se han ajustado pesos ni hiperparámetros después de observar estos resultados. Estos intervalos son condicionales al conjunto y a las particiones utilizados; no cubren variabilidad entre generadores, hardware, protocolos de solver o nuevos entrenamientos.

## Extrapolación por tamaño

Se entrenó con hasta 117 operaciones y se reservaron 39 instancias con más de 117 operaciones, todas dentro del límite 15×15.

| Método | MAE de fracción de tiempo | R² |
|---|---:|---:|
| Mediana | 0.162603 | -0.433208 |
| RF de tamaño | 0.145585 | 0.102444 |
| RF de grafo | 0.182285 | 0.085089 |
| CEJSP inductivo | 0.179903 | 0.160516 |

CEJSP logra mayor R² en este único corte, pero mantiene mayor MAE que el RF de tamaño. No permite afirmar una mejora general de extrapolación.

## Rendimiento medido

Comparación controlada del patrón anterior de entrenamiento duplicado (`cross_validate` más `cross_val_predict`) frente a ajustar una vez cada pliegue y reutilizar predicciones. Datos, árboles, particiones y un hilo iguales; calentamiento previo y orden alternado durante cinco repeticiones.

- Tiempo mediano con entrenamiento duplicado: 0.899188 s.
- Tiempo mediano reutilizando ajustes: 0.462531 s.
- Mediana de los cocientes pareados de aceleración: **1.972270×**; rango observado 1.813283× a 1.998282×.
- Diferencia máxima entre predicciones: **0.0** en las cinco repeticiones.
- Extracción de grafos de las 196 instancias: 9.719212 s, medida por separado una vez.

La mediana de cocientes no tiene por qué coincidir con el cociente entre medianas. El speedup corresponde exclusivamente al componente de validación con ajustes duplicados. No mide aceleración global del paquete ni del solver y no incluye extracción de grafos. No se presenta la medida única de extracción como benchmark comparativo.

## Implicaciones

La validación de software se mantiene: el paquete funciona, conserva persistencia e inferencia y evita fugas en el protocolo evaluado. La utilidad de la puntuación inductiva como predictor de dificultad todavía no está respaldada por esta comparación. La mezcla de atipicidad estructural y regresión es una posible causa que requiere ablación; estos resultados no la identifican causalmente.

Antes de presentar una superioridad científica, conviene separar las cuatro señales estructurales de la regresión, analizar redundancia y rango de la referencia, y evaluar una cohorte más amplia con suficientes instancias difíciles. Cualquier selección de pesos debe hacerse mediante validación interna, reservando un test externo intacto. No se cambian automáticamente la metodología ni los pesos para hacer positivos estos resultados.
