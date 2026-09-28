# Ejecución del entregable OSHA

Usar el entorno existente `ml_env`; no instalar automáticamente `requirements.txt` sobre otro entorno. El archivo enumera versiones verificadas, no autoriza cambios de paquetes.

1. Abrir `_build/html/index.html` para leer el libro sin ejecutar Python ni disponer del CSV.
2. Para reproducir, colocar el CSV original junto al notebook o establecer `OSHA_DATA_DIR` con su carpeta. Se espera `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv`, obtenido de [OSHA](https://www.osha.gov/itadata). La ejecución verifica una huella SHA-256 y no modifica el archivo.
3. Ejecutar `python -u execute_notebook.py` con el Python de `ml_env`. El kernel `python3` debe apuntar a ese mismo Python; el programa lo verifica sin modificarlo. Se ejecutan las 65 celdas y se reconstruye Jupyter Book.
4. Examinar `osha_group_time_results/delivery_verification.json` y `book_build.log`. La verificación acredita reproducibilidad e integridad computacional, no una calificación docente.

## Diseño y cachés

`osha_group_time_study.py` fija la asignación aleatoria de establecimientos (60 % a prueba, semilla 42) sin utilizar resultados del modelo. Esta proporción equilibra el soporte raro con el tamaño de entrenamiento: el preanálisis dejó 39 fallecimientos en prueba y 269.516 casos de desarrollo; no es una proporción exigida por el curso. Un cálculo de planificación con Wilson da cerca de ±15 puntos porcentuales si la sensibilidad verdadera ronda 50 %, pero no garantiza una conclusión precisa. El desarrollo usa enero–septiembre hasta el 23 de septiembre en el 40 % restante. La prueba usa octubre–diciembre exclusivamente en establecimientos reservados. Se purgan siete días antes de la prueba y de cada ventana de validación. Las validaciones son futuras y separadas por empresa. Filas fuera de esas celdas se excluyen y su total se informa.

Los resultados y modelos se guardan en `osha_group_time_results/`; cada caché se asocia al hash SHA-256 del CSV, código, utilidades, parámetros, versiones y diseño. Los modelos de una identidad distinta no se reutilizan. La regresión logística y la referencia Dummy mantienen los parámetros base: no hay búsqueda de hiperparámetros ni ajuste por resultados de prueba.

Los directorios `osha_temporal_results/` y otros artefactos anteriores son evidencia histórica del trabajo previo; no son resultados del protocolo grupo-tiempo actual. El cuarto trimestre ya había sido examinado, por lo que esta evaluación no es completamente inédita.

## Auditoría y procedencia

`osha_screening_results.json` conserva el inventario global y una primera evaluación agrupada histórica. `provenance/osha_screening.py` es su generador original sin modificar. `python reproduce_screening.py` reproduce únicamente el inventario y no reajusta el modelo principal.

Los diagnósticos univariables se guardan aparte en `single_feature_checks.json`; no determinan los predictores. Los archivos de predicción y partición contienen identificadores para auditoría y no son datos anonimizados. El paquete omite el CSV original y las narrativas; limitar su distribución a la revisión académica.

La instalación anterior cambió inesperadamente tres dependencias auxiliares, restauradas después: zipp 3.23.0, typing-extensions 4.14.1 e importlib-metadata 8.7.0. Se conserva la evidencia del inventario corregido y su verificación. Durante esta revisión no se modificaron paquetes.

La pregunta es clasificación retrospectiva de cuatro resultados en incidentes reportados, no predicción de ocurrencia ni demostración de alerta temprana. Los intervalos por establecimiento no miden incertidumbre de entrenamiento ni choques comunes de calendario.
