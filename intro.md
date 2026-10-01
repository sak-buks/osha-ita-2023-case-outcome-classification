# Clasificación de resultados de incidentes laborales

Este primer entregable estudia cuatro resultados registrados en casos de lesión o enfermedad reportados mediante OSHA ITA. El análisis prioriza calidad de datos, exploración interpretada y evaluación de una regresión logística frente a una referencia trivial.

La evaluación primaria separa establecimientos y periodos: el modelo se desarrolla con enero–septiembre en establecimientos asignados al desarrollo y se prueba en octubre–diciembre en establecimientos reservados. Se purgan siete días antes de los cortes temporales de evaluación. La clase de fallecimiento es muy infrecuente y se presenta con sus métricas e incertidumbre.

**Autores:** Jaime Andres Besada, Ivan Prada.

## Datos y ejecución

- [Fuente oficial](https://www.osha.gov/itadata).
- Base de datos: [ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip](https://www.osha.gov/sites/default/largefiles/ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip) (~655 MB descomprimido; no se incluye en el repositorio).
- Archivo local: `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv` (sin modificaciones).
- Notebook: `osha_2023_eda_revisado.ipynb`.
- Dependencias: [`requirements.txt`](https://github.com/sak-buks/osha-ita-2023-case-outcome-classification/blob/main/requirements.txt) (Python 3.9; versiones usadas y semillas aleatorias fijadas). Las semillas también se listan en la sección 1.1 del notebook.
- Implementación del modelo: `osha_group_time_study.py`, con utilidades comunes en `osha_deliverable_utils.py`.
- Artefactos actuales: `osha_group_time_results/`; las evaluaciones anteriores se conservan separadamente como evidencia histórica.

El libro presenta los resultados guardados. Para reproducir, abrir el notebook en un entorno con `requirements.txt` y, si se desea reconstruir el libro, instalar además `requirements-book.txt` y ejecutar `jupyter-book build .`. La prueba por empresas y tiempo está fijada por semilla, soportes y huellas de datos y código.

## Límites

La publicación contiene casos reportados bajo criterios administrativos; no representa a todos los trabajadores ni permite inferir tasas de lesión sin denominadores apropiados. El proyecto no identifica causas, intervenciones ni una alerta temprana.
