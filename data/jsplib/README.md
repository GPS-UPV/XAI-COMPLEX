# Datos JSPLIB para CEJSP

Fuente: [ScheduleOpt JSPLIB](https://scheduleopt.github.io/benchmarks/jsplib), repositorio [ScheduleOpt/benchmarks](https://github.com/ScheduleOpt/benchmarks).
Instantánea: `8a6e36ef31263d36975e56fb481e0ba04e188d6f` (22 de octubre de 2026).
Los datos y las etiquetas se conservan bajo la licencia de origen CC-BY-SA-4.0; véase `LICENSE.upstream` dentro de la instantánea. Esta licencia de datos se distingue de la licencia del código CEJSP.

Se descargaron los 334 JSON sin comprimir del índice fijado y se verificó cada archivo contra su hash Git. Los otros 42 registros del índice son archivos comprimidos que no se descargaron. El README original contiene 376 etiquetas; la cifra de instancias de su texto introductorio no se usa para contar el conjunto.

El entrenamiento admite 272 instancias JSP de rutas de permutación y hasta 2500 operaciones: toy 154, easy 15, medium 22, hard 21, open 60. Se excluyen 62 JSON por tamaño y 42 registros sin JSON descomprimido. Los motivos por instancia están en el informe del modelo. No se modifica ninguna etiqueta ni se convierte `closed` en otra clase; esta instantánea no contiene filas etiquetadas `closed`.

Para repetir la descarga verificada desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe evaluation\download_jsplib.py data\jsplib\8a6e36ef31263d36975e56fb481e0ba04e188d6f
```

Los ficheros `snapshot.json`, los índices originales y `download-report.json` permiten comprobar procedencia e integridad. Los JSON originales no se incorporan al wheel de CEJSP.
