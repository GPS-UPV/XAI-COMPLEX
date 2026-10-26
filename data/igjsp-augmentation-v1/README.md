# Aumento experimental mediante IGJSP

Generador: IGJSP 2.0.1, instalado desde su wheel publicado en PyPI, SHA256 `3728651c8ca52d13167b8b2fbab8fc6a0139c60dfd479abf0c09713b571a8175`. No se modificó el paquete IGJSP. Su documentación y licencia están en https://pypi.org/project/IGJSP/.

Hay 816 instancias: tres semillas por cada uno de los 272 benchmarks elegibles. La configuración conserva dimensiones y usa JSP, una velocidad, distribución uniforme y ausencia de fechas. Las rutas son generadas de forma independiente por IGJSP. Un reescalado externo aproxima la mediana de duración del benchmark de referencia; el factor se conserva en metadata. No se afirma reproducir la distribución conjunta ni la dificultad del benchmark.

La referencia de configuraciones procede de ScheduleOpt JSPLIB, commit `8a6e36ef31263d36975e56fb481e0ba04e188d6f`, CC-BY-SA-4.0. Véase `../jsplib/README.md` y la licencia de la instantánea para la atribución de los datos fuente.

`manifest.json` conserva semillas, hashes y grupo de referencia. `ground_truth_label` es null: no se hereda la etiqueta del benchmark. El entrenamiento de cada pliegue usa exclusivamente los descendientes de sus grupos de entrenamiento.

Las pseudoetiquetas finales, con voto y peso, están en `models/jsplib-8a6e36ef3126-v3-igjsp/final-weak-labels.json`. Son etiquetas débiles. Su ampliación empeoró la evaluación controlada y no entra en la referencia recomendada.

`labeling-queue.csv` prioriza 100 casos por voto no calibrado de easy y entropía, para una campaña posterior de etiquetado independiente. La columna de etiqueta real está vacía deliberadamente. Un timeout de una ejecución corta no convierte una instancia en open.
