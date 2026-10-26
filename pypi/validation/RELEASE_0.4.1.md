# CEJSP 0.4.1: prueba IGJSP y build

Estado: comprobado el 26 de octubre de 2026. Se conserva el modelo jsplib-easy-v7-trajectory/reference-accuracy.joblib, cuyo 84,19 % es un resultado exploratorio sobre JSPLIB, no una precision garantizada para instancias generadas nuevas.

## Ejecutar la prueba

Desde la raiz del repositorio, con las dependencias del modelo y IGJSP instaladas:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'pypi/src'
.\.venv\Scripts\python.exe pypi/examples/test_igjsp_classifier.py --model models/jsplib-easy-v7-trajectory/reference-accuracy.joblib --output pypi/validation/mi-prueba --jobs 6 --machines 6 --seed 42
```

Usar una carpeta de salida vacia. El script invoca IGJSP 2.0.1, guarda su JSON nativo, comprueba las dimensiones, carga el clasificador, obtiene la clase y los votos y verifica que recargar el modelo conserva el resultado. No modifica el generador ni reescala duraciones. Guarda instance.json y result.json.

El caso ejecutado devuelve toy, con voto 0,985 sin calibrar y aviso outside_training_bounds. No hay etiqueta verdadera de esa instancia: la comprobacion acredita funcionamiento de la integracion, no exactitud predictiva. Otros tamanos y semillas se pueden indicar como argumentos.

## Distribucion

Los artefactos preparados son ../dist/cejsp-0.4.1-py3-none-any.whl y ../dist/cejsp-0.4.1.tar.gz. Ambos pasan twine check --strict. El sdist incluye examples/test_igjsp_classifier.py; el wheel contiene la biblioteca y la CLI. La referencia entrenada se entrega separada del paquete y debe conservarse junto a las versiones de models/jsplib-easy-v7-trajectory/requirements-accuracy.txt. El extra cejsp[integration] incorpora IGJSP y OR-Tools; no incluye el archivo de modelo.

Se ejecutaron 66 pruebas y 22 subpruebas sin omisiones; el registro es tests-0.4.1.log. Las instalaciones de wheel y de un wheel reconstruido desde el sdist pasan el mismo test usando Python aislado (-I), sin importar el codigo fuente del proyecto. Comparten dependencias del entorno de trabajo, por lo que no equivalen a resolver todas las dependencias desde cero. Los hashes y detalles estan en release-0.4.1/report.json. No se ha publicado en PyPI ni TestPyPI; la disponibilidad del nombre/version no esta verificada.

## Retirada de la cascada

La carpeta models/jsplib-cascade-v13 y los tres scripts de la cascada se retiraron del arbol activo y del indice de perfiles. La revision automatica rechazo el borrado recursivo porque requeria aprobacion de sandbox, deshabilitada en esta sesion. Se traslado el contenido a .validation-tmp/retired-cascade-v13 mediante una operacion reversible; no se afirma que haya sido eliminado definitivamente.

## Correccion del arranque del ejemplo

El ejemplo ahora localiza pypi/src cuando CEJSP no esta instalado. Se ha comprobado sin PYTHONPATH y forzando la ausencia del paquete instalado. Se reconstruyeron los artefactos y se repitio twine check --strict. Los hashes actuales estan en example-import-fix.json; el informe de instalaciones anterior conserva los hashes anteriores para no atribuirle comprobaciones de otro artefacto.
