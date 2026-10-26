# Variables para mejorar la identificación de easy

Las etiquetas son dificultad empírica histórica de JSPLIB. La cohorte contiene 272 instancias y solo 15 easy. El objetivo del estudio es encontrar señales calculables en una instancia nueva, sin consultar su clase ni su mejor solución publicada. Una asociación no demuestra que una variable determine causalmente la dificultad.

| Grupo | Variables | Hipótesis para easy | Coste y restricciones |
|---|---|---|---|
| Tamaño y tiempos | Dimensiones, cuantiles, dispersión, asimetría | Control de tamaño y distribución frente a clases próximas | Estadística de la instancia |
| Rutas y cargas | Posiciones de máquina, coincidencia de orden, cargas por tramos | Distinguir estructuras con el mismo tamaño y diferente congestión | JSP de rutas de permutación |
| Heads y tails | Trabajo antes y después de una máquina; dispersión entre trabajos | Detectar conflictos tempranos o tardíos que los promedios ocultan | Cálculo a partir de las rutas |
| Cotas | Relajación preemptiva de una máquina con releases y tails de cada operación | Medir cuánto refuerza la estructura la cota de carga elemental | Sin resolver el JSP completo; cota verificada |
| Reglas constructivas | SPT, LPT, mayor y menor trabajo restante; gaps contra la nueva cota | Valorar si decisiones de despacho simples consiguen buenas soluciones | Cuatro construcciones factibles, sin óptimo conocido |
| Búsqueda breve | Prueba de optimalidad, gap, mejora del incumbente, conflictos, ramas y propagaciones de CP-SAT | Aproximar la respuesta inicial de un motor, más próxima al origen de las etiquetas | Modo probe explícito, presupuesto determinista 0,2 y un worker por sonda |

Las sondas no utilizan tiempos de pared, porque dependen de la carga y del paralelismo de la máquina. El modo probe admite tiempos enteros y rechaza fracciones en vez de redondearlas silenciosamente. Las fechas y alternativas de máquina fuera del contrato del backend estructural se rechazan. No se identifica un timeout con la etiqueta open.

## Estudio específico de easy

Se ajusta una regresión logística univariante, con imputación y escalado dentro de cada pliegue, para cada característica. La predicción se obtiene fuera del entrenamiento en cinco pliegues. Se compara easy frente al resto. El ranking resultante es exploratorio: hay variables correlacionadas, solo 15 positivos y posible efecto de familia. No equivale a una validación independiente de la selección de variables.

En esta cohorte destacan ramas exploradas por CP-SAT (AUC 0,724), variación en coincidencia de rutas (0,696), dispersión de tails (0,676) y gaps de despacho LWKR (0,673) y SPT (0,668). No se observa un separador casi perfecto con este análisis lineal; esto no descarta relaciones no lineales. Los valores, medianas por clase y pliegues están en `models/jsplib-easy-v5/easy-feature-associations.csv` y `feature-study.json`.

La cota preemptiva se comprobó por enumeración en problemas pequeños. También se verificó que no excedía la mejor solución publicada en 172 benchmarks con filas comparables. Esa mejor solución se usa solamente para comprobar la cota, nunca como característica.

## Reentrenamiento completo y alternativas

Se comparan Extra Trees, Random Forest, SVM, HistGradientBoosting y CatBoost, con variación de profundidad, regularización, número de árboles, tamaño de hoja, variables por división y balanceo. Se ajustan multiplicadores de decisión de easy, medium y hard dentro de la validación interna. Son votos ajustados, sin calibración probabilística.

La primera ronda permite variables estructurales, cotas y reglas constructivas. Si no bastan, la segunda añade las sondas CP-SAT, manteniendo los candidatos anteriores como posible fallback. La selección utiliza 0,5 de acierto, 0,4 de macro-F1 y 0,1 de recall de easy; se publica cada métrica por separado. El 95% se comprueba sobre predicciones fuera del entrenamiento, nunca sobre el ajuste final a todas las etiquetas.

La validación externa es de cinco pliegues, con tres internos para elegir configuración y pesos de decisión. Todo el conjunto ya se ha explorado en estudios anteriores: sigue siendo validación interna del proyecto, no un test externo nuevo. El ajuste final utiliza las 272 etiquetas reales y conserva las referencias anteriores.

Los procesos paralelizan configuraciones o instancias; cada ajuste interno y sonda usa un hilo. La primera cuadrícula con boosting de profundidad siete resultó demasiado costosa y se interrumpió sin resultado de precisión. Se conserva su plan y registro; se pasó a profundidades tres y cinco y 160 iteraciones de boosting, reutilizando las características verificadas. No se presenta ese intento interrumpido como un experimento aprobado.

Fuentes técnicas: [modelo JSP de OR-Tools](https://developers.google.com/optimization/scheduling/job_shop), [parámetros CP-SAT](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto) y [parámetros de CatBoost](https://catboost.ai/docs/en/references/training-parameters/common).


## Extensión y cierre

Se añadieron geometría de precedencias, entropía de transiciones, solapamiento de prefijos, bloques críticos, holguras y progreso de la sonda entre 0,2 y 3 unidades deterministas, hasta 189 variables. El estudio extendido también usa árboles univariantes para detectar bandas no lineales: propagaciones de la sonda larga AUC 0,791; coincidencia de rutas 0,772; carga del último tramo 0,761. No son porcentajes de acierto. La quinta ronda selecciona 16/32/64 variables dentro de cada entrenamiento mediante información mutua de easy y/o multiclase. Resultados completos: RESULTADOS_EASY.md.
