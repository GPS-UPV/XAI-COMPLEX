# Auditoría posterior de la mejora CEJSP

Antislop after, por instrucción de sesión. Alcance: nuevos descriptores, candidatos supervisados, evaluación y datos IGJSP. No hay interfaz gráfica; las reglas visuales no aplican.

1. R-36 / C-5, HIGH, resuelto: una cifra de la variante elegida puede sobreestimar la selección. El informe diferencia 81,25% de validación cruzada exploratoria de la variante fija y 78,68% de selección anidada, y muestra la ausencia de mejora de macro-F1 de esta última.
2. R-36 / C-5, HIGH, resuelto: los datos sintéticos similares no tienen necesariamente la etiqueta del benchmark. Las 816 instancias generadas conservan origen y semillas; las etiquetas débiles se registran aparte y no se incorporan al modelo recomendado.
3. R-17 / R-36, HIGH, resuelto: se documentan los experimentos negativos con pseudoetiquetas y con 30 benchmarks adicionales. No se presenta un aumento del conjunto como mejora automática.
4. R-31, MEDIUM, revisado: los comentarios nuevos explican restricciones de etiquetas, dependencia entre pliegues y reescalado externo. No incluyen narración decorativa. La revisión de comentarios conserva la lógica ejecutable.

Delivery Gate:

- PASS C-2 / R-35: los estudios terminaron, conservaron modelos y predicciones, y comprobaron persistencia y predicción individual/lote.
- PASS R-17 / R-36 / R-38: tablas extraídas de JSON y CSV de evaluación, sin etiquetas inventadas ni afirmaciones de precisión externa.
- PASS C-5: se declara el fallo persistente de easy y el empeoramiento de macro-F1 en la prueba de selección entre familias.
- PASS R-31: comentarios breves en español sobre restricciones reales; no se modificó lógica durante su limpieza.
- PASS R-35: 50 pruebas y 17 subtests aprobados sin omisiones; 49 pruebas unittest del wheel instalado aprobadas; wheel y sdist 0.3.0 superan twine check --strict.
- No aplicable: elementos visuales, tipografía, navegación, animación y accesibilidad gráfica.
