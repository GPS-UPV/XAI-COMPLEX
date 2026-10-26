"""Write the study report from recorded predictions and metrics."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];models=root/'models'
    a=json.loads((models/'jsplib-easy-v5/evaluation.json').read_text())
    c=json.loads((models/'jsplib-easy-v6-specialists/evaluation.json').read_text())
    d=json.loads((models/'jsplib-easy-v7-trajectory/evaluation.json').read_text())
    e=json.loads((models/'jsplib-easy-v8-selection/evaluation.json').read_text())
    ablation=json.loads((models/'jsplib-easy-v7-trajectory/feature-group-ablation.json').read_text())
    accuracy=next(r['metrics'] for r in ablation['results'] if r['group']=='long_probe_trajectory')
    baseline=a['baseline'];easy=e['round_e']
    rows=[('Control anterior',baseline),('Ronda 1: variables y parámetros',a['round_a']),('Ronda 2: sonda corta',a['round_b']),('Ronda 3: ordinal y especialista',c['round_c']),('Ronda 4: geometría y trayectoria',d['round_d']),('Ronda 5: selección para easy',easy)]
    table='\n'.join(f"| {name} | {r['accuracy']*100:.2f}% | {r['macro_f1']:.3f} | {round(r['per_class']['easy']['recall']*15)}/15 |" for name,r in rows)
    ablation_table='\n'.join(f"| {r['group']} | {r['features']} | {r['metrics']['accuracy']*100:.2f}% | {r['metrics']['macro_f1']:.3f} | {round(r['metrics']['per_class']['easy']['recall']*15)}/15 |" for r in ablation['results'])
    selected=json.loads((models/'jsplib-easy-v8-selection/selected-features.json').read_text())
    feature_names=', '.join('`'+r['feature']+'`' for r in selected['features'][:12])
    text=f'''# Estudio de variables y reentrenamiento completo

## Resultado y objetivo del 95%

No se ha alcanzado ni demostrado un 95% de acierto fuera del entrenamiento. Se completaron cinco rondas, con hasta 12 procesos paralelos, 189 variables candidatas y ajuste completo de referencias nuevas. No se alteraron etiquetas ni se usaron nombres de instancia, estados publicados de solución o mejores soluciones conocidas como características.

Se entregan dos perfiles experimentales:

| Perfil | Acierto global | Macro-F1 | Recall easy | Precisión easy |
|---|---:|---:|---:|---:|
| Priorizar acierto global | {accuracy['accuracy']*100:.2f}% | {accuracy['macro_f1']:.3f} | 1/15 ({accuracy['per_class']['easy']['recall']*100:.1f}%) | {accuracy['per_class']['easy']['precision']*100:.1f}% |
| Priorizar identificación de easy | {easy['accuracy']*100:.2f}% | {easy['macro_f1']:.3f} | 7/15 ({easy['per_class']['easy']['recall']*100:.1f}%) | {easy['per_class']['easy']['precision']*100:.1f}% |

El primero procede de una ablación con configuración fija, elegida tras estudiar los resultados: es exploratorio y puede tener optimismo de selección. El segundo refleja la validación anidada de su procedimiento de selección. Son las mismas 272 instancias, pero no se debe interpretar la diferencia entre procedimientos como una prueba independiente de superioridad.

El control obtiene {baseline['accuracy']*100:.2f}% y 1/15 easy en estas particiones. El perfil global gana {(accuracy['accuracy']-baseline['accuracy'])*100:.2f} puntos de acierto; no mejora su recall de easy. El perfil orientado a easy gana seis aciertos de esa clase, pero reduce el acierto global y produce muchas falsas predicciones easy. Su precisión de {easy['per_class']['easy']['precision']*100:.1f}% impide presentarlo como clasificación fiable de esa clase.

## Rondas repetidas

| Procedimiento | Acierto | Macro-F1 | Easy correctas |
|---|---:|---:|---:|
{table}

La validación usa cinco pliegues externos estratificados, semilla 20261024, y tres internos para parámetros y multiplicadores de decisión. El objetivo de selección es 0,5 acierto + 0,4 macro-F1 + 0,1 recall easy. Los pesos de decisión, el escalado, la imputación y la selección de variables se ajustan sin consultar los ejemplos del pliegue externo. La quinta ronda incluye ranking de información mutua multiclase/easy y 16, 32 o 64 variables.

La cohorte ya había sido examinada en estudios anteriores y se realizaron varias búsquedas. Las métricas son evaluación interna; no constituyen un nuevo test externo ni una demostración de transferencia a IGJSP o familias desconocidas. No se midió el 95% con el ajuste a todos los ejemplos: la referencia v5 alcanza 97,06% en entrenamiento y 15/15 easy, pero solo 80,51% y 4/15 fuera del ajuste. Esa diferencia es sobreajuste.

La primera cuadrícula de CatBoost profundo se interrumpió por coste y no se cuenta como validación completada. Se conservan su plan y registro. Se redujeron profundidad e iteraciones y se reutilizaron las características verificadas. La extracción larga usa checkpoints y procedencia para no reutilizar silenciosamente una caché de otra configuración.

## Qué aportan las variables

La comparación siguiente mantiene fijo Extra Trees con 200 árboles, max_features 0,8, balanceo y sin multiplicadores de decisión:

| Grupo de entrada | Variables | Acierto | Macro-F1 | Easy correctas |
|---|---:|---:|---:|---:|
{ablation_table}

Las nuevas variables incluyen heads/tails, una cota preemptiva por máquina, gaps de cuatro reglas de despacho, entropía y geometría de las rutas, bloques críticos y progreso de CP-SAT entre presupuestos deterministas de 0,2 y 3. Los tiempos de pared y el número de workers no entran como características. `n_jobs` en la matriz significa número de trabajos de la instancia, no procesos de entrenamiento.

El análisis univariante incorpora relaciones no lineales, porque easy es una clase intermedia. Las propagaciones de la sonda larga alcanzan AUC 0,791, el cuartil de coincidencia de rutas 0,772 y la dispersión de carga del último tramo 0,761. Son asociaciones exploratorias easy frente al resto, no porcentajes de acierto ni determinantes causales. Hay correlaciones entre variables y efectos de familia.

El perfil easy final selecciona {selected['count']} variables, entre ellas {feature_names}. Ese ranking procede del ajuste final y no es evidencia independiente de significación. Los CSV completos contienen medianas por clase y AUC lineal/no lineal. Las cotas se validaron por enumeración en problemas pequeños y contra 172 mejores soluciones publicadas, empleadas solo para comprobación.

## Referencias entrenadas y uso

Todos los modelos se reajustaron con las 272 etiquetas originales tras evaluar, manteniendo las referencias anteriores. No se incorporaron pseudoetiquetas IGJSP: su experimento anterior fue negativo.

Perfil que prioriza acierto global:

```powershell
.\\.venv\\Scripts\\python.exe -m cejsp classify instancia.json models\\jsplib-easy-v7-trajectory\\reference-accuracy.joblib --trust-model
```

Perfil que prioriza easy:

```powershell
.\\.venv\\Scripts\\python.exe -m cejsp classify instancia.json models\\jsplib-easy-v8-selection\\reference.joblib --trust-model
```

También se admiten DZN y Taillard dentro del contrato JSP de permutación. Estos perfiles usan sondas: la inferencia ejecuta una búsqueda limitada y no es instantánea ni tiene garantía de segundos de pared. Los tiempos deben ser enteros; no se redondean fracciones silenciosamente. Se rechazan fechas y alternativas de máquina no representadas. Los votos están sin calibrar y open sigue siendo una predicción histórica, no una certificación.

La referencia v5, sin CP-SAT durante inferencia, queda disponible como alternativa de menor esfuerzo computacional: `models/jsplib-easy-v5/reference.joblib`. No se afirma una aceleración medida entre perfiles. La versión de paquete necesaria es CEJSP 0.4.0; los requisitos exactos acompañan cada modelo.

Para reentrenar, copiar el manifiesto correspondiente, incluir explícitamente todos los ejemplos que deban conservarse, fijar una nueva model_version y escribir otro archivo:

```powershell
cejsp retrain dataset/manifest-v2.json models/reference-v2.joblib --previous models/jsplib-easy-v8-selection/reference.joblib --trust-model
```

Las reglas de etiquetado nuevas requieren otro identificador de protocolo. El comando retrain ajusta el modelo; los scripts de evaluation realizan la evaluación anidada. Los planes, parámetros, particiones, predicciones, variables seleccionadas y fuentes están en las carpetas de cada estudio.

## Limitaciones pendientes

Solo hay 15 easy. Corregir los 14 errores easy del control, dejando iguales las otras clases, llevaría el acierto global a 85,29%, no al 95%: también deben mejorar medium, hard y open. Aumentar variables y capacidad permite memorizar el entrenamiento, pero no garantiza esas mejoras fuera de él.

Hace falta un conjunto externo, con más easy, medium y hard y etiquetas verificadas bajo un protocolo comparable. Las 816 instancias IGJSP y su cola de etiquetado permanecen disponibles; la similitud de características y un timeout corto no proporcionan por sí solos esas etiquetas. Estos resultados no prueban que 95% sea imposible; muestran que las búsquedas realizadas no lo alcanzaron.

## Validación técnica

59 pruebas pytest y 22 subtests aprobados sin omisiones; 58 pruebas unittest del wheel instalado aprobadas. El flujo original con grafo, estimación, persistencia e invariancia se mantiene aprobado sobre 20 instancias reales, incluida orb07. Wheel y sdist 0.4.0 superan twine check --strict. En cada instalación limpia pasaron las nueve pruebas nuevas de investigación sin omisiones, además de la inferencia equivalente JSON/DZN/Taillard de los dos perfiles. Sus registros están junto a las referencias. La batería completa, incluidos los grafos, se ejecutó en el entorno de trabajo; no se ha repetido toda la batería con extras de grafos en los dos entornos limpios.

La auditoría final añadió una regresión de carga total cero: sus gaps son cero. La fuente usada para extraer la cohorte se conserva en extraction-source, con hashes comprobados. La corrección posterior no modifica las 145 variables sin CP de ninguna de las 272 instancias de entrenamiento; los resultados y hashes están en zero-fix-equivalence.json.

No se publicó en PyPI/TestPyPI ni se hizo push. Las instrucciones de publicación permanecen en pypi/docs/PUBLICAR_PYPI.md. El 95% sigue pendiente.
'''
    (root/'evaluation/RESULTADOS_EASY.md').write_text(text,encoding='utf-8')
    (models/'jsplib-easy-v8-selection/README.md').write_text(text,encoding='utf-8')
    profiles={'model':'jsplib-easy-v7-trajectory/reference-accuracy.joblib','package_version':'0.4.0','goal_95_achieved':False,
              'profiles':{'accuracy':{'model':'jsplib-easy-v7-trajectory/reference-accuracy.joblib','exploratory_accuracy':accuracy['accuracy'],'easy_recall':accuracy['per_class']['easy']['recall'],'evaluation':'Fixed-model exploratory OOF ablation, post-selection'},
                          'easy':{'model':'jsplib-easy-v8-selection/reference.joblib','nested_accuracy':easy['accuracy'],'easy_recall':easy['per_class']['easy']['recall'],'easy_precision':easy['per_class']['easy']['precision']},
                          'no_solver_probe':{'model':'jsplib-easy-v5/reference.joblib','nested_accuracy':a['round_b']['accuracy'],'evaluation_warning':'Nested estimate of the selection procedure; final selected model is search-only'}},
              'previous_model':'jsplib-8a6e36ef3126-v2/reference.joblib','report':'../evaluation/RESULTADOS_EASY.md','warning':'Experimental internal estimates; solver probes add inference effort; no external IGJSP difficulty validation'}
    (models/'RECOMMENDED.json').write_text(json.dumps(profiles,indent=2));(models/'PROFILES.json').write_text(json.dumps(profiles,indent=2))


if __name__=='__main__':main()
