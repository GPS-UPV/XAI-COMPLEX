# CEJSP: resultados iniciales

Se han comprobado 2401 instancias DZN existentes frente a sus soluciones JSON de cp-sat. Todas pasan la verificación independiente de schedules: precedencias, exclusión de máquinas y objetivo. El hash del contenido base es distinto en los 2401 casos. No se ha modificado ni generado ninguna instancia.

El solver acredita optimalidad en 707 registros; otros 1694 constan como SATISFIED. Una solución factible no demuestra optimalidad. La fracción de tiempo observada tiene mediana 0.400563 y percentil 75 de 0.948865; ninguna supera el presupuesto guardado. El objetivo principal de este piloto es esa fracción observada, no el tiempo latente para resolver óptimamente.

Con tres semillas prefijadas y cinco pliegues sin contaminación de instancias base:

| Método de referencia | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mediana ajustada en entrenamiento | 0.392283 | 0.421809 | -0.021870 |
| Ridge de log(jobs), log(máquinas), log(operaciones) | 0.196368 | 0.238380 | 0.673635 |

Las métricas agregan predicciones fuera de muestra de las tres repeticiones, no 7203 observaciones independientes. Los archivos `results/size_baseline_predictions.csv` y `results/size_baseline_summary.json` permiten comprobar los valores exactos.

El tamaño explica una parte considerable de la variación del objetivo. Una evaluación de CEJSP debe superar controles de tamaño y examinar la extrapolación, además de mostrar correlaciones globales.

La precisión de CEJSP y el speedup siguen pendientes: está preparado el protocolo de `README.md`, pero este entorno no puede ejecutar el Python 3.11 con las dependencias. No se atribuyen al estimador los resultados de estos dos baselines.
