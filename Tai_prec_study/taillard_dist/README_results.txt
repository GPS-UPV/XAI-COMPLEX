Resultados generados para: .

Instancias leídas correctamente: 162
Ficheros no parseados: 1
Familias detectadas: abz, ft, la, orb, swv, ta, yn

Interpretación rápida:
- Variable principal analizada: tiempos de proceso p_ij.
- Los tiempos son enteros, por lo que la comparación principal está en la familia discreta.
- Se generan resultados por fichero, por familia de benchmark, por tamaño y globales.
- La distribución global discreta más plausible según AIC es: discrete_uniform_0_100.
- Media global: 49.6063
- Desviación típica global: 28.3158
- p-value global frente a U{1,...,99} con chi-cuadrado agrupado: nan

Archivos clave:
- summary_by_family.csv: estadísticos y mejor distribución para ta, swv, yn, la, etc.
- fit_candidates_by_family.csv: ranking de distribuciones candidatas por familia.
- summary_by_size.csv: estadísticos por tamaño de instancia, por ejemplo 10x10, 20x15.
- global_fit.csv: fitting de todos los tiempos juntos.
- machine_position_summary.csv: frecuencias de máquinas por posición de operación.

Notas:
- Los ajustes continuos son exploratorios y no deben compararse directamente con los AIC discretos.
- Los órdenes de máquina son permutaciones, no variables continuas. Para ellos no tiene sentido ajustar normal/gamma/weibull.
- Si una instancia pequeña da discrete_uniform_minmax en vez de discrete_uniform_1_99, puede deberse simplemente a que no aparecen los extremos 1 o 99 en esa muestra.
