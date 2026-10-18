# Traspaso del Entregable 3 (OSHA ITA 2023): estado al 2026-10-18

Decisiones cerradas D1 a D8: ver `CONTEXTO_CLAUDE_CODE.md`. Este documento dice qué se hizo, cómo reproducirlo y qué falta.

## Qué se hizo
- Validación anidada: 3 pliegues externos hacia adelante (los del entregable 1) y 3 internos con `StratifiedGroupKFold` por establecimiento (semilla 1042; la externa es 42).
- 7 modelos × 4 balanceos × 4 optimizadores = 108 combinaciones (KNN × `class_weight` no aplica) × 3 pliegues = 324 tareas. Más 2 referencias: Dummy y la logística del entregable 1.
- Corrida: 2026-10-18 02:53 a 12:52, sin errores. 20 evaluaciones por optimizador, búsqueda sobre el 25 % del entrenamiento externo (todas las muertes) y reentrenamiento con el 100 %. XGBoost en CPU.
- Resultados por tarea en `results/runs/<modelo>__<balanceo>__<optimizador>/fold<k>.json` (hiperparámetros, métricas, tiempos, historial de evaluaciones) y `fold<k>_pred.npz` (probabilidades, y_true, establecimiento, caso).
- Tabla maestra: `results/master_registry.csv` y `.parquet`. Resumen con los 3 pliegues juntos: `results/pooled_summary.csv`. Curvas: `results/convergence_curves.csv`.

## Cómo reproducirlo
1. Entorno: `environment.yml` / `requirements.txt` (Python 3.9, scikit-learn 1.2.2, imbalanced-learn 0.11.0, xgboost 2.1.4, optuna 4.1.0, DEAP 1.4.1). Es un clon de `ml_env` con paquetes añadidos; las versiones están fijadas en esos archivos.
2. Datos y código previo: el código del entregable 1 (`osha_deliverable_utils.py`, `osha_group_time_study.py`) ya está en la raíz del repo y `osha3` lo encuentra solo (no modificarlo). El CSV de 654 MB no está en el repo (ver el README de la raíz): ponerlo en la raíz o fijar `OSHA_DATA_DIR`. Si el código del entregable 1 está en otra carpeta, fijar `OSHA_CODE_DIR`.
3. `python -c "from osha3.data import materialize_folds; materialize_folds()"` crea `results/folds/` (parquet, ~25 s).
4. `python run_nested.py --workers 7 --n-evals 20` lanza o reanuda la corrida completa (~10 h, necesita CPU libre y ~15 GB de RAM). `python summarize.py` regenera la tabla maestra.

## Estructura del código (`osha3/`)
`data.py` (carga, pliegues, muestra de búsqueda) · `metrics.py` · `models.py` (registro de modelos: añadir el modelo original es añadir una entrada) · `pipeline.py` (pipeline sin fugas) · `optimizers.py` (Grid, Random, Optuna-TPE, DEAP) · `nested.py` (bucle anidado y control de memoria).

## Desviaciones respecto al PDF (para el JBook)
1. 108 combinaciones en vez de 112 (KNN no admite `class_weight` ni `sample_weight`; verificado).
2. Regresión omitida (el objetivo es categórico).
3. 3 pliegues externos (estructura temporal); las ventanas se solapan, así que la desviación estándar entre pliegues se subestima.
4. 20 evaluaciones por optimizador (el material usa 60; el contexto proponía 30 y se recortó por tiempo). Rejillas de 16 a 20 puntos.
5. Búsqueda sobre el 25 % del entrenamiento con reentrenamiento al 100 %.
6. SMOTE y ADASYN con `sampling_strategy='minority'` (solo Death), como los define el PDF; con la estrategia por defecto ADASYN falla en 7 de 9 pliegues internos. `k = min(5, muertes - 1)`; siempre se pudo usar k = 5. Se aplican sobre el one-hot, así que los casos sintéticos tienen valores fraccionarios (limitación declarada); `BernoulliNB` usa `binarize=0.5`.
7. Logística con `liblinear` (como el cap. 3) en vez de `saga`, que no convergía en 1.000 iteraciones.
8. XGBoost sin parada temprana; `n_estimators` se busca como hiperparámetro. Sin ajuste de la métrica en KNN (con one-hot las distancias ordenan igual).
9. SVM: `LinearSVC` solo o Nystroem (m ∈ {100, 300, 1000}) + `LinearSVC`, calibrado con `CalibratedClassifierCV`.
10. Semilla global 42; el bucle interno usa 1042 para ser independiente del externo.

## Resultados verificados contra el registro (media de los 3 pliegues externos)
- Dummy 0,250 de balanced accuracy; logística del entregable 1: 0,385 (recall de Death 0 de 12).
- Las 108 combinaciones van de 0,345 a 0,505; la mejor es XGBoost + `class_weight` + Bayesiano (detecta 8 de 12 muertes, IC de Wilson 95 %: 0,39 a 0,86).
- Esa combinación marca 5.337 de 61.649 casos como Death (precisión < 0,2 %) y tiene F1 macro bajo (0,352). 58 de 108 combinaciones no detectan ninguna muerte.
- Es el máximo de 108 valores ruidosos (hay 1, 3 y 8 muertes en los pliegues externos), así que está sesgado al alza. El conjunto de test (oct a dic) no se ha usado.
- El ranking depende de la métrica: KNN lidera en balanced accuracy pero tiene el peor log-loss y AUC; XGBoost y Random Forest lideran en AUC y log-loss.

## Problemas conocidos
- XGBoost en GPU dio una balanced accuracy interna de 0,29 frente a 0,38 en CPU con SMOTE (una configuración). Sin investigar; la corrida usa CPU.
- La memoria llegó a 14,6 de 15,4 GB. Los reentrenamientos grandes (SVM Nystroem, Random Forest grande) se serializan con archivos de bloqueo.
- Sin comprobar por optimizador que Random, Bayesiano y Genético usaron las 20 evaluaciones (mínimo 16 en el registro, coherente con rejillas de 16 y 18 puntos).

## Pendiente
Pruebas estadísticas (Friedman, Nemenyi y diagrama de diferencia crítica, DeLong con Holm, bootstrap, delta de Cliff) · matrices de confusión, ROC, calibración · gráficas de convergencia y diversidad genética · SHAP y LIME · complejidad O(·) y perfilamiento · EDA mejorado · modelo original (D8) · JBook · sustentación.
