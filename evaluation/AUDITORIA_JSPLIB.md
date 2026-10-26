# Auditoría posterior JSPLIB

Modo antislop: after, por instrucción de sesión. Alcance: adaptación JSPLIB, entrenamiento supervisado y documentación de esta entrega. Las reglas visuales no aplican: no se ha creado una interfaz gráfica.

1. C-5 / R-36, HIGH, corregido: el acierto global puede ocultar clases deficientes. El informe presenta macro-F1, balanced accuracy, recall por clase y el resultado easy 0/3; no atribuye generalización validada a IGJSP.
2. C-5 / R-36, HIGH, corregido: una medición de extracción no prueba aceleración. Se registra el tiempo observado y se evita afirmar una mejora de rendimiento o equivalencia entre centralidad exacta y muestreada.
3. R-31, MEDIUM, corregido: la documentación que indicaba ausencia de una referencia real quedó obsoleta. Se sustituye por procedencia, protocolo, rutas del modelo y limitaciones verificables.

Delivery Gate:

- PASS C-2 / R-35: entrenamiento terminado, artefacto persistido y recargado; verificaciones de persistencia, lote e inferencia desde archivo registradas en training-report.json.
- PASS C-5 / R-17 / R-36 / R-38: métricas proceden de training-report.json; partición y predicciones se conservan. No se inventan etiquetas ni se equipara open predicho con un problema abierto certificado.
- PASS R-31, comentarios: el comentario añadido sobre label de JSPLIB explica una restricción de formato real; se conserva. La auditoría de comentarios no modifica lógica ejecutable.
- PASS C-3 / R-16: la documentación describe descarga, exclusiones, uso y reentrenamiento, sin reclamos comerciales.
- PASS R-35, pruebas: 46 pruebas aprobadas sin omisiones y 12 subtests; wheel y sdist 0.2.1 superan twine check --strict. Las instalaciones limpias ligeras no se presentan como pruebas científicas completas.
- No aplicable: diseño, navegación, tipografía, animación y accesibilidad visual, al no existir una interfaz gráfica en este cambio.
