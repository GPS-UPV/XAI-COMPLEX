# CEJSP: primera referencia supervisada JSPLIB

Modelo: `models/jsplib-8a6e36ef3126-v1/reference.joblib`. Versión: `jsplib-8a6e36ef3126-v1`.
Fuente: [ScheduleOpt JSPLIB](https://scheduleopt.github.io/benchmarks/jsplib), commit `8a6e36ef31263d36975e56fb481e0ba04e188d6f`. Datos CC-BY-SA-4.0; atribución y licencia preservadas en `data/jsplib/`.

## Datos y método

Se descargaron y verificaron 334 JSON; 42 archivos comprimidos no se descargaron. De las 376 etiquetas originales, 272 casos entran en el entrenamiento: toy 154, easy 15, medium 22, hard 21, open 60. Se excluyeron 62 JSON de más de 2500 operaciones y 42 casos sin JSON descomprimido. Las exclusiones individuales están en `dataset-report.json`.

Bosque aleatorio balanceado de 200 árboles, semilla 42, con características del grafo original y la variante explícita `graph_approx`: intermediación con 100 nodos muestreados, semilla 42. No se usan nombres, familias, etiquetas ni información de optimalidad como características. La energía se excluye para no distinguir formatos por datos ausentes. El estimador original sigue disponible y no se declara equivalencia numérica entre métodos.

El test estratificado de 55 instancias se fijó antes del ajuste, con semilla 20261022. Se comprobaron duplicados por contenido y disjunción de identificadores base. La imputación se ajustó dentro del conjunto de entrenamiento. No se seleccionaron hiperparámetros con el test. El diagnóstico por familia entrena de nuevo dejando fuera una de las ocho familias cada vez.

## Resultados reservados

| Método | Acierto | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CEJSP con grafos | 0.7455 | 0.4904 | 0.4970 |
| Bosque con dimensiones | 0.6000 | 0.4515 | 0.3922 |
| Clase mayoritaria | 0.5636 | 0.2000 | 0.1442 |

Macro-F1 con familia completa fuera: 0.4245; balanced accuracy: 0.4319; acierto: 0.6618. Todos los pliegues de familia resultaron evaluables.

El intervalo bootstrap del 95% de macro-F1 es [0.3827, 0.6093]. La diferencia frente al bosque de tamaño tiene intervalo [-0.0142, 0.2182], que incluye cero. No se afirma una ventaja estadísticamente concluyente. Son intervalos condicionados a estos modelos y este test pequeño.

Recall por clase del test: toy 29/31, easy 0/3, medium 3/5, hard 1/4 y open 8/12. La clase easy es una limitación observada, no un resultado satisfactorio oculto por el acierto global. El modelo es una primera referencia experimental; requiere más etiquetas minoritarias y evaluación externa IGJSP. Los votos no están calibrados. `open` predicho no certifica un problema abierto.

La extracción de características tardó 156.95 segundos en esta ejecución. No es una comparación de rendimiento ni justifica una aceleración respecto a otro backend.

## Modelo y trazabilidad

`evaluation-reference.joblib` conserva el modelo que produjo las métricas de test. `reference.joblib` se reajustó después con las 272 instancias; no se presenta su acierto de entrenamiento como validación. Se comprobaron persistencia y predicción individual frente a lote. `input-formats-check.json` registra la clasificación de una misma instancia existente ft06 en JSON, DZN y Taillard.

El informe completo, partición, manifiesto, predicciones y versiones están junto al modelo. `requirements-model.txt` fija las dependencias de persistencia. El modelo se entrenó con Python 3.12.14; no se ha comprobado la carga entre versiones menores distintas de Python.

## Uso desde la raíz

```powershell
.\.venv\Scripts\python.exe -m cejsp classify instancia.json models\jsplib-8a6e36ef3126-v1\reference.joblib --trust-model
```

Se admiten también `instancia.dzn` y un archivo Taillard existente. Instalar CEJSP 0.2.1 para cargar el esquema de grafos aproximados. Si hay varias velocidades, debe seleccionarse una proyección explícita para la referencia; no se descartan velocidades automáticamente.

Para repetir el protocolo completo, elegir otra carpeta de salida para preservar los modelos:

```powershell
.\.venv\Scripts\python.exe evaluation\train_jsplib.py data\jsplib\8a6e36ef31263d36975e56fb481e0ba04e188d6f --out models\jsplib-repeticion
```

Para reentrenar con nuevos datos etiquetados, crear un manifiesto completo que incluya los casos anteriores que se deseen conservar y una nueva versión:

```powershell
.\.venv\Scripts\python.exe -m cejsp retrain dataset\manifest-v2.json models\reference-v2.joblib --previous models\jsplib-8a6e36ef3126-v1\reference.joblib --trust-model
```

## Comprobaciones de esta entrega

46 pruebas aprobadas sin omisiones, más 12 subtests; un aviso de deprecación de Torch externo. Wheel y sdist CEJSP 0.2.1 construidos y comprobados con `twine check --strict`. No se ha publicado ni hecho push. Las pruebas completas de dependencias se ejecutaron en el entorno de trabajo; las instalaciones limpias adicionales del wheel y sdist verifican solamente importaciones y lectura ligera sin dependencias científicas.

Verificación adicional del wheel instalado con Python aislado: 45 pruebas unittest aprobadas sin omisiones y flujo original completo aprobado sobre 20 instancias reales. La misma ft06 produjo votos idénticos en JSON IGJSP, DZN y Taillard. Los logs se conservan junto al modelo.
