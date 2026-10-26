# Mejora del clasificador y experimento IGJSP

Referencia recomendada: `models/jsplib-8a6e36ef3126-v2/reference.joblib`, CEJSP 0.3.0. Se conservan los modelos anteriores y los experimentales. No se ha publicado ni hecho push.

## Comparación sobre las mismas instancias

Los 272 benchmarks originales se evaluaron con cinco pliegues estratificados, semilla 20261023. Todas las predicciones de estas tablas se hicieron sin usar la instancia evaluada para ajustar el modelo. Los datos ya se habían explorado antes: esta es evaluación interna, no un nuevo test externo.

| Procedimiento | Acierto | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Control original, grafo + Random Forest | 0.7610 | 0.5364 | 0.5277 |
| Variante fija estructural + Extra Trees | 0.8125 | 0.5713 | 0.5846 |
| Selección anidada entre once variantes | 0.7868 | 0.5165 | 0.5234 |
| Variante fija con pseudoetiquetas IGJSP | 0.7978 | 0.5442 | 0.5502 |
| Variante fija con 30 benchmarks reales adicionales | 0.7904 | 0.5216 | 0.5386 |

La variante fija gana 5.15 puntos de acierto y 0.057 de macro-F1. Se eligió tras comparar candidatos: su 81,25% es una estimación exploratoria de la variante, no una precisión externa certificada. La evaluación anidada estima el procedimiento de selección completo y arroja 78,68%, con macro-F1 ligeramente inferior al control original. No se ha demostrado que la búsqueda mejore todas las clases.

El diagnóstico dejando fuera familias completas es más exigente: control macro-F1 0.4245 y selección anidada 0.3867. El acierto pasa de 0.6618 a 0.6765, pero empeora macro-F1. No hay evidencia suficiente de mejora equilibrada entre familias desconocidas.

## Errores y cambios de entrada

| Clase | Casos reales | Aciertos del control | Aciertos de Extra Trees estructural |
|---|---:|---:|---:|
| toy | 154 | 137 | 147 |
| easy | 15 | 3 | 2 |
| medium | 22 | 9 | 9 |
| hard | 21 | 7 | 10 |
| open | 60 | 51 | 53 |

`easy` continúa siendo escasa y difícil. Comparar su recall sobre estos 15 casos con el antiguo 0/3 del test de 55 casos sería comparar conjuntos distintos. Se conservan las predicciones completas en `error-analysis.csv` junto al modelo.

Hay 11 configuraciones de dimensiones con más de una etiqueta: el tamaño por sí solo no determina la dificultad. Las características nuevas incluyen cuantiles y asimetría de duraciones, concentración de cargas por trabajo y máquina, posición de las máquinas en las rutas, coincidencia de precedencias relativas, cargas por tramo y concentración espectral de la matriz de tiempos. Son señales calculadas de la instancia; no entran nombre, familia, etiqueta, estado de solución ni mejor cota conocida. La asociación con el error no prueba causalidad de una característica individual.

Se añadió el backend `structural` y el híbrido con el grafo, sin sustituir el estimador científico anterior. Extra Trees se presenta como una variante supervisada explícita, elegida por su evaluación interna; no como equivalente numérico del estimador original. `structural` exige rutas JSP de permutación, una velocidad o proyección explícita y ausencia de fechas. Rechaza FJSP y restricciones fuera de ese contrato.

## IGJSP: datos generados y etiquetas débiles

Se utilizó IGJSP 2.0.1 sin modificarlo para generar 816 instancias: tres semillas por benchmark, dimensiones coincidentes y distribución uniforme. Un reescalado externo, registrado en cada JSON, aproxima la mediana de duración del benchmark de referencia. No se ha afirmado reproducir toda su estructura. Datos y hashes: `data/igjsp-augmentation-v1/manifest.json`.

No se copió la clase del benchmark de referencia. El profesor se ajusta únicamente con el entrenamiento de cada pliegue. Sus pseudoetiquetas requieren voto mínimo 0,7, tienen peso 0,15 y cupo de 30 por clase. El conjunto de variantes asociado a un benchmark se excluye cuando ese benchmark se evalúa. La selección de la ampliación se hace dentro de los pliegues internos.

En el ajuste final experimental se admitieron 31 pseudoetiquetas: toy 30 y hard 1; ninguna easy. La ampliación fija empeora acierto y macro-F1. Se conserva `v3-igjsp/experimental-reference.joblib` para reproducibilidad y no se recomienda como modelo principal. Las pseudoetiquetas no son verdad de terreno ni pruebas de optimalidad.

También se prepararon 30 benchmarks originales con etiquetas reales que antes se excluyeron por tamaño del grafo, ampliando la cohorte a 302. La variante estructural permite procesarlos. La ablación fija no mejora sobre los mismos 272 casos; el resultado y modelo experimental se conservan en `v4-expanded/`. No se añaden al modelo recomendado por el mero hecho de disponer de más datos.

## Uso y reentrenamiento

```powershell
.\.venv\Scripts\python.exe -m cejsp classify instancia.json models\jsplib-8a6e36ef3126-v2\reference.joblib --trust-model
```

También acepta DZN y Taillard dentro del contrato anterior. El modelo usa 272 etiquetas originales, el protocolo fijado de JSPLIB y votos sin calibrar. `open` es una predicción de la clase histórica del protocolo.

Para repetir, conservar las carpetas de resultados o elegir rutas nuevas en las constantes de los scripts: `evaluation/improve_jsplib.py`, `evaluation/augment_igjsp.py` y `evaluation/expand_real_jsplib.py`. Ninguno sustituye una referencia existente. El manifiesto v2 permite reentrenar mediante `cejsp retrain`, usando otra versión y salida. Los scripts de evaluación conservan selección y particiones; `cejsp train` solo ajusta el modelo y no ejecuta automáticamente esa evaluación.

La mejora pendiente exige etiquetas independientes, especialmente easy, medium y hard. Las instancias generadas pueden entrar en una campaña de resolución con motores y presupuesto declarados; sus tiempos deberán calibrarse con benchmarks reales antes de mezclarlos con etiquetas históricas. Un timeout aislado no etiqueta open.

## Verificación

50 pruebas aprobadas sin omisiones y 17 subtests. Avisos externos de Torch y de SVC en scikit-learn, conservados. Wheel y sdist CEJSP 0.3.0 construidos y comprobados con twine check --strict. Persistencia e invariancia individual/lote verificadas en los modelos evaluados. No se presenta una aceleración sin medirla ni se afirma optimalidad global de la búsqueda de clasificadores.


## Ablación y comprobación final

En las mismas particiones de cinco pliegues, Extra Trees con cinco señales de dimensiones obtiene 62,50% de acierto y macro-F1 0,450; con 62 señales de dimensiones, duraciones y cargas, 72,06% y 0,482; con las 132 señales estructurales completas, 81,25% y 0,585. Se conservan los resultados en `feature-ablation.json`. Es una ablación exploratoria, no una explicación causal individual de cada error.

El wheel y el sdist finales se instalaron en dos entornos limpios con las dependencias científicas necesarias para la referencia estructural. Ambos cargaron el modelo persistido y produjeron votos idénticos para ft06 en JSON, DZN y Taillard, con invariancia individual/lote. No se instalaron los extras de grafos en esos dos entornos; las pruebas completas de grafos se ejecutaron en el entorno de trabajo. Evidencia: `clean-wheel-check.json` y `clean-sdist-check.json`.

La cola `data/igjsp-augmentation-v1/labeling-queue.csv` deja 100 casos para etiquetado independiente. Para repetir el experimento generativo se requiere `python -m pip install IGJSP==2.0.1`; el paquete CEJSP de inferencia no necesita IGJSP ni modifica su código.
