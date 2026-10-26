# Evaluación científica y de rendimiento de CEJSP

El primer estudio completo terminó en `resultados-20261022-184844/`. Consultar [RESULTADOS_ESTUDIO.md](RESULTADOS_ESTUDIO.md): se confirma ahorro en el componente de ajustes duplicados, pero no superioridad predictiva del estimador inductivo frente a un control de tamaño. El protocolo y las limitaciones siguientes se mantienen; los apartados de preparación describen su diseño previo.

Este estudio es independiente de la batería de distribución. No modifica IGJSP, los datos originales ni la metodología del paquete. No se afirman resultados de CEJSP antes de ejecutar el estudio completo.

## Correspondencia y auditoría

`run_study.py` lee DZN ya existentes de tipo 0 y velocidad 1 y sus JSON de solver con el mismo nombre completo. No interpreta el tipo como semilla ni utiliza los CSV ya fusionados o SHAP como entradas del predictor. Comprueba dimensiones, precedencias, solapamientos en máquinas y que el objetivo guardado equivale a makespan más energía. El orden de operaciones se obtiene ordenando las máquinas por su rango de precedencia, conforme a `Minizinc/Models/RD/JSP0.mzn`. Los hashes de cada par de archivos se guardan en `manifest.csv`.

Las unidades son milisegundos según las conversiones explícitas de `getSolutions.py`. La fuente no registra hardware, versión de solver ni todas las condiciones de ejecución; no se supone comparabilidad con ejecuciones futuras. Las soluciones sin schedule se identifican expresamente. Un certificado de factibilidad no certifica optimalidad.

## Protocolo fijado antes de la ejecución completa

- Objetivo principal: fracción del presupuesto consumido, `clip(solveTime/maxTime,0,1)`. Se informa cuántas observaciones superan el límite. Es tiempo observado condicionado al presupuesto, no tiempo latente hasta demostrar optimalidad.
- Objetivo secundario: calibrador ponderado de solver sin componente de tamaño del modelo. Los cuantiles se ajustan solo con entrenamiento. Es una variable construida, no ground truth independiente.
- Baselines iniciales sobre todos los datos válidos: mediana y Ridge con logaritmos de jobs, máquinas y operaciones. Normalización ajustada en cada pliegue; regularización fija 1. No se busca el hiperparámetro con el test.
- Cohorte de grafos predefinida: jobs y máquinas hasta 15 por defecto, para limitar el coste de centralidad exacta. Se compara RF de tamaño, RF de características del grafo original, mediana y CEJSP inductivo con supervisión. No se proyecta toda la población a partir de esta subcohorte.
- Tres semillas fijadas: 11, 29 y 47; cinco pliegues por semilla, agrupados por hash de tiempos y rutas base. No se separan duplicados entre entrenamiento y test.
- Extrapolación adicional: instancias por encima del percentil 80 del número de operaciones se reservan como test. No representa transferencia a otra familia de generación.
- MAE, RMSE, R² y Spearman por pliegue, sin interpretar cada repetición como una observación independiente. Diferencias pareadas de MAE frente al RF de tamaño con bootstrap de 2000 remuestreos a nivel de instancia base, después de promediar repeticiones. Se publican todos los contrastes, incluidos negativos.

La clasificación en tres niveles no se utiliza como ground truth: las categorías del paquete son relativas a la referencia. El estudio evalúa puntuaciones continuas. Las advertencias de covarianza se conservan. Una asociación con un objetivo construido no demuestra complejidad computacional intrínseca.

## Rendimiento

Se registran por separado extracción de grafos, ajuste e inferencia. La comparación controlada reproduce el patrón anterior `cross_validate` seguido de `cross_val_predict` frente a entrenar cada pliegue una vez y reutilizar predicciones para métricas. Se mantienen árboles, datos, splits y un solo hilo; hay calentamiento y cinco repeticiones con orden alternado. Se exige igualdad de predicciones a `1e-12`.

El speedup resultante mide solo la eliminación de ajustes duplicados. No es una comparación completa de las versiones antigua y nueva, ni incluye extracción. El preprocesamiento común de ese microbenchmark está fuera de la medición; sus resultados no se usan como evidencia científica de precisión.

## Ejecución

Desde la raíz del repositorio, aprovechando el entorno que ya funcionó:

```powershell
.\evaluation\EVALUAR_WINDOWS.ps1 -Python .\pypi\validacion-20261022-175907\entorno\Scripts\python.exe
```

Los resultados se guardan en una carpeta nueva `evaluation/resultados-FECHA/`. El script instala la distribución renombrada, comprueba sus pruebas y ejecuta auditoría, baselines, precisión y rendimiento. No publica nada.

Salidas: `audit.json`, `manifest.csv`, `size_baseline_summary.json`, `predictions.csv`, `metrics.csv`, `paired_intervals.csv`, `splits.json`, `timings.csv`, `controlled_performance.csv`, `study.json` y transcripción. Ante un fallo se detiene; los archivos parciales no equivalen a un estudio completo.

La auditoría local inicial comprobó 2401 pares DZN/solver y sus schedules sin rechazos. Los resultados de CEJSP y el speedup siguen pendientes hasta que la ejecución completa produzca `study.json`.
