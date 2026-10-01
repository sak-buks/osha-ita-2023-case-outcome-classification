# OSHA ITA 2023: primer entregable

- Notebook: `osha_2023_eda_revisado.ipynb` (salidas ya ejecutadas).
- Módulos: `eda_utils.py`, `osha_deliverable_utils.py`, `osha_group_time_study.py`, `osha_review_checks.py`; inventario global en `osha_screening_results.json`.
- Python 3.9 (el del curso). Dependencias: `pip install -r requirements.txt` (análisis) y `requirements-book.txt` (solo para construir Jupyter Book con `jupyter-book build .`).
- Datos (no incluidos por tamaño): descargar [ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip](https://www.osha.gov/sites/default/largefiles/ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip) (fuente: [OSHA ITA](https://www.osha.gov/itadata)), extraer `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv` y colocarlo junto al notebook (o fijar la variable de entorno `OSHA_DATA_DIR`). Los cachés de resultados se regeneran al ejecutar (semillas fijas).
- Los artefactos de predicción conservan identificadores y no son datos anonimizados; limitar su distribución a la revisión académica.
- Autores: Jaime Andres Besada, Ivan Prada. Ejecutado en el entorno conda `ml_venv` (Python 3.9; versiones exactas en `requirements.txt`).
- Publicar el libro: `jupyter-book build .` y luego `ghp-import -n -p -f _build/html` (rama `gh-pages`; activar en *Settings → Pages*).
