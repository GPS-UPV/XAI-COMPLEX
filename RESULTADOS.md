# Resultado de la revisión, 22/10/2026

Los cambios están aplicados en `D:\CODE\XAI-COMPLEX`. No se ha publicado, hecho push ni modificado IGJSP. La validación funcional y de distribución de xai-complex terminó correctamente en `pypi/validacion-20261022-175907/`, según los registros comprobados. El nuevo nombre público es **CEJSP: JSP Complexity Estimator**, distribución e import `cejsp`. Se conservan `xai_complex` y la identidad de las clases para compatibilidad con referencias persistidas. El paquete renombrado está construido y comprobado parcialmente; requiere repetir la validación con dependencias antes de publicar. Los bloqueos de instalación descritos más abajo son históricos para la versión anterior, aunque el agente sigue sin poder ejecutar el intérprete Microsoft Store del usuario.

## Evaluación científica iniciada

**Nuevo flujo 0.2:** se añaden lectores públicos JSON/DZN/Taillard, las clases toy/easy/medium/hard/open y `JobShopComplexityClassifier` supervisado con protocolo explícito, artefactos versionados y reentrenamiento completo sin sobrescribir referencias. Se conserva el estimador anterior. Pasan los lectores sobre 2401 DZN y la equivalencia de rutas y duraciones de 162 pares JSON/Taillard, incluido cero en orb07. La batería actual descubre 43 pruebas: 29 aprobadas y 14 omitidas por dependencias ausentes en la sesión del agente. Tres omisiones nuevas cubren el clasificador supervisado de cinco clases, su persistencia y clasificación equivalente de archivos en tres formatos; no se consideran aprobadas. No existe todavía un modelo real de cinco clases ni precisión validada para él: faltan el protocolo de etiquetado y las etiquetas de referencia. La validación completa anterior y sus métricas no se trasladan a esta nueva variante. Consultar `pypi/docs/FIVE_CLASSES.md`.

**Actualización final del piloto:** `evaluation/resultados-20261022-184844` completó el estudio. El informe científico es `evaluation/RESULTADOS_ESTUDIO.md`. En la cohorte de 196 instancias hasta 15×15, CEJSP inductivo obtiene MAE 0.097217 frente a 0.059919 del RF de tamaño para fracción de tiempo. El intervalo pareado de la diferencia de MAE favorece al control de tamaño; no se confirma superioridad predictiva de CEJSP. El RF de grafo no muestra mejora clara de MAE. La reutilización de ajustes acelera el componente controlado de validación 1.972270× de mediana, con diferencia de predicciones cero; no es una aceleración global. Las menciones a métricas pendientes que siguen en este apartado describen el historial anterior, no el estado actual del piloto.

La ejecución `evaluation/resultados-20261022-183913` superó las 33 pruebas del paquete CEJSP sin omisiones y las cuatro pruebas iniciales de semántica del estudio. La evaluación se detuvo al crear registros de características con nombres enteros en un DataFrame temporal. Se corrigió el script para conservar los nombres originales después de seleccionar columnas de entrenamiento y reutilizarlos en test. Las cinco pruebas actuales del estudio pasan localmente, incluida la regresión de esquema. No se ha cambiado el estimador ni se han producido todavía métricas completas de CEJSP.

Se ha ejecutado la auditoría independiente de 2401 DZN y sus JSON de solver: los 2401 schedules satisfacen precedencias, exclusión de máquinas y objetivo makespan más energía según el modelo MiniZinc guardado. Esto no vuelve a demostrar optimalidad. Los identificadores completos se emparejan por archivo y quedan registrados con hashes en `evaluation/results/manifest.csv`; no se reutiliza el remapeo antiguo de tipo a semilla.

Baselines ejecutados en esos 2401 casos con 3 semillas y 5 pliegues agrupados por contenido base, usando como objetivo la fracción de presupuesto consumida:

| Baseline | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mediana de entrenamiento | 0.392283 | 0.421809 | -0.021870 |
| Ridge sobre logaritmos de tamaños | 0.196368 | 0.238380 | 0.673635 |

Hay 707 registros OPTIMAL_SOLUTION y 1694 SATISFIED. La etiqueta temporal mide ejecución observada, no tiempo necesario para demostrar optimalidad. Estos resultados de baselines no son precisión de CEJSP. La evaluación CEJSP frente a RF de tamaño y RF de grafo, la extrapolación por tamaño y el microbenchmark controlado de reutilización de pliegues están preparados en `evaluation/run_study.py` y pendientes de ejecución con dependencias. No se afirma ninguna aceleración ni superioridad de CEJSP.

La ejecución completa se realiza con `evaluation/EVALUAR_WINDOWS.ps1`, descrita en `evaluation/README.md`; usa por defecto una cohorte de grafos hasta 15×15, por lo que sus conclusiones no cubren todas las instancias grandes. Los avisos de covarianza no se ocultan.

## Estado final verificado

- 32 pruebas aprobadas, sin omisiones, en el entorno inicial y en cada instalación limpia de wheel y sdist: tres ejecuciones satisfactorias de la misma batería.
- `pip check` correcto en los tres entornos.
- 162 JSON reales leídos, normalizados y descritos sin fallos.
- Flujo completo con 20 instancias reales correcto en los tres entornos: grafos, características, estimación, invariancia numérica individual/lotes y persistencia.
- Diferencia máxima individual/lotes: `4.631124372878048e-10`, dentro de la tolerancia absoluta `1e-9`; categorías y señales coinciden. Diferencia tras persistencia: cero.
- Wheel y sdist construidos; ambos superan `twine check --strict` y se instalan con sus dependencias en entornos separados.
- Los artefactos de esa ejecución son los validados. Se copian a `pypi/dist/` y sus hashes se registran en `pypi/validation/artifacts-sha256.json`.

Los avisos de covarianza siguen visibles; esta validación de software no demuestra precisión predictiva, representatividad de la referencia ni mejoras de rendimiento. No se ha realizado publicación en TestPyPI/PyPI ni se ha comprobado disponibilidad del nombre. La auditoría de comentarios anterior conserva su evidencia de AST sin cambios; los cambios posteriores del verificador mantienen comentarios breves y no alteran el estimador.

## Historial de preparación

## Aplicación y cambios

Git estaba limpio. No se encontró AGENTS.md en el repositorio. Los cinco hashes de la entrega anterior coincidían con los originales; se revisaron sus diferencias y se aplicaron con copia de seguridad en `backup-xai-20261022-164519/`. Después se corrigieron problemas adicionales detectados durante esta revisión.

- INFEASIBLE se distingue de FEASIBLE. Los estados y pesos de las etiquetas se validan; los componentes ausentes renormalizan los pesos disponibles.
- Vecinos y componentes se acotan al tamaño de muestra. Los objetivos sin señales y los lotes totalmente constantes se rechazan.
- Objetivo y preprocesamiento se ajustan dentro de cada pliegue. Cada modelo de validación se ajusta una vez para métricas y predicciones. Se devuelve la partición para auditarla.
- `base_instance_id`, cuando está presente, agrupa automáticamente las variantes en evaluación. Se rechazan grupos ausentes. Sin grupos, se emite un aviso explícito; el usuario debe aportar el identificador de la instancia base.
- No se truncan identificadores para inferir correspondencias. Se exigen identificadores completos y agregación explícita de repeticiones del solver. Los CSV reales de soluciones tienen otro esquema de ID que los grafos: necesitan un mapeo verificado antes de calibrar. No se ha inventado ese mapeo.
- Se conserva la última fila y se permite exportar sin etiquetas; SHAP es opcional en el script experimental. No hay supresión global de avisos en los módulos del paquete.
- Fechas por trabajo se expanden por operaciones. La selección de velocidad es explícita; también se rechazan energías con varias velocidades sin proyectar. Se reconoce la semilla antes de sufijos de configuración.
- El grafo de rutas fijas rechaza disponibilidad flexible. El adaptador conserva la información de entrada; no se afirma compatibilidad científica con FJSP. Un fallo comunicado por el extractor no se elimina silenciosamente al filtrar características numéricas.
- `pypi/` contiene fuente, pruebas, CLI, persistencia, documentación, verificadores y distribuciones. `xai_complex/` replica la fuente para los scripts de investigación de la raíz.

## Metodología

La API por lotes conserva las cuatro señales del estimador ponderado: Isolation Forest, LOF, distancia robusta de Mahalanobis y vecinos, con su prior y mezcla supervisada opcionales. Normalizaciones y categorías dependen del lote.

`ComplexityEstimator` es una variante inductiva: aprende escalas, vecindario y umbrales de referencia y los congela para consultas posteriores. No utiliza el prior del análisis original; LOF se usa en modo novelty y la combinación conserva calibración de referencia. Es necesaria para ofrecer inferencia persistente individual, pero no equivale numéricamente al estimador por lotes. No se distribuye modelo entrenado ni se demuestra validez predictiva mediante las pruebas de software.

`estimate_complexity.py` sigue siendo experimental. La entrega anterior sustituyó etiquetas arbitrarias de clúster por distancia al centroide; esta revisión limita los centroides a `min(2401, floor(sqrt(n)))` para evitar la degeneración de un centroide por muestra. Es una heurística explícita, cambia sus puntuaciones y requiere validación científica. No es el estimador publicado en el paquete ni se afirma que mejore resultados científicos. Su calibración supervisada anterior tenía fugas y remapeaba semillas a cero: se sustituye por el protocolo ponderado compartido y se documenta aquí ese cambio. Los resultados antiguos deben recalcularse y contrastarse antes de reutilizarse en publicaciones.

## Validación realmente ejecutada

| Comprobación | Resultado |
|---|---|
| Lectura y escritura del repositorio | Correctas; cambios y copia de seguridad creados |
| Entorno virtual local `.venv` | Creado con Python 3.12.14; pip funciona |
| Instalación de dependencias y extras | Bloqueada: DNS/getaddrinfo al descargar de PyPI |
| Pruebas sobre wheel instalado, Python `-I` | 28 descubiertas: 17 aprobadas, 11 omitidas, sin fallos entre las ejecutadas |
| Verificador estricto de pruebas | Devuelve error por las 11 omisiones, como debe |
| JSON reales de TaillardInstancesJSON | 162/162 pasan normalización y descripción, incluido orb07 con duración cero |
| Flujo JSON, grafo, características y estimación | Intentado y bloqueado por dependencias ausentes |
| Persistencia, inferencia individual/lotes, agrupación | Pruebas preparadas; ejecución bloqueada, no aprobadas |
| Wheel y sdist | Construidos con backend setuptools local, sin aislamiento de construcción |
| Wheel en entorno limpio fuera del árbol fuente | Instalación sin dependencias y pruebas ligeras; no valida el estimador |
| Sdist en entorno limpio | Instalación bloqueada por dependencias de construcción ausentes |
| twine check --strict | Intentado; twine no está instalado y su descarga está bloqueada |
| Antislop de comentarios | AST idéntico antes/después en los 16 archivos revisados |

Las 11 omisiones son las 7 de estimador/grafos de la entrega anterior más 4 regresiones nuevas de evaluación, IDs, grupos y lote constante. No representan un empeoramiento del resultado de pruebas ejecutadas ni se cuentan como aprobadas. La nueva prueba de pesos pasa.

No se han medido aceleraciones. La reutilización de predicciones reduce trabajo duplicado por diseño, pero no se atribuye ningún factor de mejora de rendimiento. Las dependencias de los scripts ajenos al flujo del paquete (por ejemplo, solvers comerciales y notebooks de investigación) no se han instalado ni se afirman validadas.

Los registros actuales están en `pypi/validation/`. Los registros heredados se conservan en `pypi/validation/previous/`. El informe de la pasada de comentarios es `pypi/validation/comment-audit.json`; su comprobación AST se refiere exclusivamente a esa pasada, no a las correcciones funcionales anteriores.

## Continuación exacta

Segundo diagnóstico de invariancia: se inspeccionaron las 20 parejas de predicciones guardadas en `validacion-20261022-174519/pipeline.json`. La máxima diferencia absoluta es `4.631124372878048e-10`; no cambian categorías ni señales. La tolerancia relativa anterior era excesivamente estricta para estos datos. El verificador utiliza ahora tolerancia absoluta `1e-9` en la escala [0,1], sin tolerancia relativa, y mantiene igualdad de categorías y señales. Las 20 parejas guardadas pasan y las cuatro pruebas del comparador pasan, incluida la regresión de la discrepancia observada y el rechazo de `2e-9`. Esto valida la comparación de esas predicciones, no sustituye la repetición del flujo completo ni la comprobación de persistencia. Los avisos de covarianza sin rango completo permanecen visibles: hay características dependientes y más características que referencias en esta muestra; esta prueba no demuestra calidad de calibración científica. No se ha cambiado el estimador ni su covarianza para ocultar esos avisos.

Actualización tras la ejecución del usuario `pypi/validacion-20261022-174519`: las dependencias se instalaron correctamente en Python 3.11, `pip check` pasó, las 28 pruebas pasaron sin omisiones y los 162 JSON pasaron. El flujo real llegó a la comparación de predicciones individuales/por lote y falló en la igualdad exacta de diccionarios. El registro no mostraba la magnitud de la diferencia, por lo que todavía no puede atribuirse con certeza al redondeo. Se corrigió el verificador para admitir únicamente diferencias mínimas (`rtol=1e-10`, `atol=1e-12`), exigir igualdad de categorías y señales, y registrar las diferencias. Si la discrepancia supera la tolerancia, sigue fallando y guarda las predicciones en el JSON de diagnóstico. Tres pruebas nuevas del comparador pasan localmente. No se ha modificado el estimador por este fallo. Este entorno de agente no puede iniciar el intérprete Microsoft Store 3.11 usado en esa ejecución; el flujo completo corregido necesita repetirse allí. El estado anterior de 17 aprobadas/11 omitidas corresponde a la sesión restringida del agente, no a la ejecución posterior del usuario. La construcción, twine y las instalaciones limpias posteriores al flujo real no llegaron a ejecutarse en ese registro.

Reintento del 22/10/2026: el entorno virtual sigue ejecutándose, pero la instalación con el índice público explícito vuelve a fallar al obtener setuptools. La consulta DNS de pypi.org devuelve acceso denegado en esta sesión. Se intentó además descargar numpy directamente para aislar el problema de la construcción del paquete. Registros: `pypi/validation/retry-install.log` y `pypi/validation/retry-network.log`. No se han ejecutado de nuevo las pruebas que requieren esas dependencias ni se consideran resueltos los bloqueos.

Desde una sesión con acceso a las dependencias:

```powershell
Set-Location 'D:\CODE\XAI-COMPLEX'
& .\pypi\VALIDAR_WINDOWS.ps1 -Python python
```

Solo tras terminar sin errores ni omisiones, seguir `pypi/docs/PUBLICAR_PYPI.md`. Ese documento contiene los comandos de TestPyPI y PyPI; aquí no se ha ejecutado ninguna subida. Falta además comprobar disponibilidad del nombre y validar científicamente la referencia con el protocolo de solver y los grupos reales del experimento.


## Referencia JSPLIB entrenada (22 de octubre de 2026)

Primera referencia supervisada de cinco clases entrenada con 272 instancias y etiquetas originales de JSPLIB. Modelo: `models/jsplib-8a6e36ef3126-v1/reference.joblib`. Datos, partición, métricas por clase, limitaciones y comandos de uso: [RESULTADOS_JSPLIB.md](evaluation/RESULTADOS_JSPLIB.md). No hay validación externa con etiquetas IGJSP.


## Mejora supervisada CEJSP 0.3.0

Referencia recomendada: `models/jsplib-8a6e36ef3126-v2/reference.joblib`, características estructurales y Extra Trees. Se conservan las referencias anteriores. Resultados, limitaciones y experimentos IGJSP: `evaluation/MEJORA_CLASIFICADOR.md`. El aumento con pseudoetiquetas no se incorpora por empeorar el resultado controlado. No hay validación externa de dificultad IGJSP.


## Estudio easy y referencias 0.4.0

Se completaron cinco rondas de nuevas variables, parámetros y selección dentro de los pliegues. No se alcanzó el 95% fuera del entrenamiento. El perfil de acierto global obtiene 84,19% exploratorio; el orientado a easy reconoce 7/15 casos con 79,41% global y precisión easy de 21,21%. Son perfiles experimentales con distintos compromisos; los modelos anteriores se conservan. Informe y comandos: `evaluation/RESULTADOS_EASY.md`; perfiles: `models/PROFILES.json`. Las sondas añaden resolución limitada durante inferencia.
