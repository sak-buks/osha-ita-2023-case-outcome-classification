# Contexto para Claude Code (Entregable 3, Machine Learning)

Pega esto como primer mensaje en Claude Code, abierto en la carpeta de trabajo del entregable 3, o guárdalo como `CLAUDE.md` en esa carpeta. Fecha límite: **miércoles 2026-10-21 antes de medianoche**. Este documento recoge las decisiones tomadas con Ivan en la sesión del 17 al 18 de octubre de 2026; no las reabras sin avisarle.

## Quién y cómo trabajar con él
- Usuario: Ivan, maestría en Analítica de Datos (Uninorte), curso EST 1055-01 Machine Learning, profesor Lihki Rubio. Escribe en español e inglés.
- Respuestas cortas y claras. Explica la lógica de cada paso, no solo des código. Separa lo observado de lo inferido. Su dominio conceptual de estadística y ML es básico: usa ejemplos simples.
- Antes de lanzar cualquier corrida larga, dile qué va a correr y cuánto estimas que tardará.

## Fuentes de verdad
1. **PDF del entregable** (`Trabajo/Entregable2MachineLearningProject.pdf`; el profesor lo llama "segundo entregable", pero es el que se entrega el 21 de octubre). Es la **línea base**: se sigue al pie de la letra salvo donde haya una razón metodológica, conceptual o de desempeño. Cada desviación se documenta (ver "Desviaciones").
2. **Material de clase** (`Contenido curso/*.pdf`, capítulos 1 a 11). Donde exista una implementación o indicación del material, se usa esa.
3. Transcripciones (`Grabaciones/*.txt`) para lo que dijo el profesor. Están ruidosas.

## Problema y datos (del entregable 1)
- OSHA ITA 2023, un caso reportado = una fila. Objetivo `incident_outcome` en 4 clases: 1 Fallecimiento, 2 Días de ausencia, 3 Traslado/restricción, 4 Otro registrable. Solo clasificación (la parte de regresión del PDF no aplica).
- 8 variables: `state, naics_code, establishment_type, soc_code, type_of_incident, time_unknown, time_started_work, time_of_incident`. One-hot da 1.433 columnas. `time_unknown` es redundante con la hora "Unknown" (GVIF 3,80); se mantiene por coherencia con el entregable 1 salvo que Ivan decida quitarla.
- Desarrollo: ene a sep 2023, 269.516 casos, 69 muertes. Test (oct a dic, establecimientos reservados): 127.391 casos, 39 muertes. 60 % de los establecimientos reservados para test, semilla 42, brecha de 7 días. El **test no se toca** hasta el final (solo confirma el mejor modelo).
- Código previo reutilizable (en la carpeta del entregable 1 de Ivan): `osha_deliverable_utils.py` (`RAW_FEATURES`, `LABELS=[1,2,3,4]`, `SEED=42`, `prepare_features`, `make_pipeline`, `metrics`, `bootstrap_groups`) y `osha_group_time_study.py` (`assign_test_groups`, `_fold_data`, `WINDOWS`, `GAP_DAYS=7`). No los modifiques: impórtalos o copia lo necesario a módulos nuevos.
- CSV de 654 MB: `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv` (carpeta vía variable `OSHA_DATA_DIR`). Lee solo las columnas necesarias con `dtype=str, encoding='latin1'`.
- Entorno: el entregable 1 usó scikit-learn 1.2.2 (`ml_env`) y el notebook dice 1.3.0 (`ml_venv`). Crea un entorno nuevo con versiones fijadas (`requirements.txt` y `environment.yml`), compatible con imbalanced-learn, xgboost, optuna y DEAP, y documenta la versión.
- Hardware: 8 núcleos, 16 GB de RAM, sin GPU confirmada. Las matrices son dispersas; no las conviertas a densas sin necesidad.

## Decisiones cerradas
**D1. Solo clasificación.**

**D2. Validación anidada, diseño A** (el PDF la exige y la métrica reportada debe salir del bucle externo):
- *Externo:* los 3 pliegues hacia adelante del entregable 1 (`WINDOWS`, `_fold_data`, establecimientos separados, brecha de 7 días). Solo mide.
- *Interno:* dentro de cada entrenamiento externo, `StratifiedGroupKFold(n_splits=3, shuffle=True)` por establecimiento, con **otra semilla** que el externo (el material de clase lo recomienda para que sean independientes).
- Debilidad a declarar: las ventanas externas se solapan (cada una contiene a la anterior), así que la desviación estándar entre pliegues se subestima. El material de clase usa 5 externos por 3 internos; aquí son 3 por la estructura temporal (desviación documentada).
- El modelo final de cada pliegue externo se reentrena con **todo** el entrenamiento externo usando los mejores hiperparámetros.

**D3. Métrica de optimización: balanced accuracy.** Secundarias: F1 macro, AUC macro (uno contra el resto), recall por clase y **MCC multiclase** (el material de clase, capítulo 6, propone macro-F1 y balanced accuracy como primarias, y MCC, recall por clase y log-loss como secundarias). Reporta también log-loss.

**D4. Death.** Se mantiene como clase y la tratan los 4 balanceos del PDF. Además, el agrupamiento de categorías raras es un hiperparámetro más: `OneHotEncoder(min_frequency=...)` (valores a definir, incluyendo "sin agrupar").
- Los 4 balanceos: sin balanceo, SMOTE, ADASYN, `class_weight='balanced'`.
- **SMOTE y ADASYN necesitan más vecinos que muertes** en el conjunto que están remuestreando. Los pliegues internos tienen pocas muertes. Calcula las muertes en cada entrenamiento interno y fija `k_neighbors`/`n_neighbors = min(5, n_muertes - 1)`. Si hay menos de 2, deja constancia y usa `RandomOverSampler` para esa celda. Imprime esos conteos por pliegue antes de la corrida larga.
- SMOTE/ADASYN van **dentro** del pipeline (`imblearn.pipeline.Pipeline`), después del one-hot, de modo que solo toquen el entrenamiento de cada pliegue. Verifica que acepten matrices dispersas; si no, documenta la alternativa.
- `class_weight='balanced'` no existe en KNN (según scikit-learn; verifícalo). Para los demás modelos usa `class_weight` o pesos por muestra (`compute_sample_weight('balanced', y)`) según corresponda. **KNN × class_weight = "no aplica"**: 108 combinaciones en lugar de 112 (desviación documentada).

**D5. Presupuesto de cómputo.**
- Mismo número de evaluaciones para Random, Bayesiano y Genético (**30 por optimizador**, objetivo inicial); Grid se diseña con ≤30 puntos. Igualdad de presupuesto = comparación justa entre optimizadores (lo pide el PDF). El material de clase usa 60 evaluaciones con Optuna; aquí 30 por cómputo (desviación).
- Búsqueda (bucle interno) sobre una **muestra estratificada del 25 %** del entrenamiento externo, conservando **todas** las muertes; luego reentrenamiento con el 100 %. Es una forma de multi-fidelidad (el PDF la recomienda; se puede añadir Successive Halving/Hyperband como técnica adicional).
- **Antes de fijar el presupuesto, mide el tiempo real** de un ajuste por modelo y proyecta el total (combinaciones × 3 externos × 3 internos × evaluaciones). La ventana útil es ~una noche de cómputo. Si no cabe, el orden de recortes es: (1) bajar el número de evaluaciones, (2) bajar el tamaño de la muestra de búsqueda, (3) solo como último recurso, recortar combinaciones (el PDF permite reportar de forma exhaustiva solo las más informativas).
- Usa `n_jobs=8` con cuidado de memoria; guarda resultados por combinación para poder **reanudar** si se interrumpe.

**D6. Los 7 modelos y qué se les ajusta** (borrador; valida contra el material y justifica cada rango, con escalas logarítmicas donde corresponda):
- KNN: `n_neighbors`, `weights` (uniform/distance), métrica. (cap. 2)
- Naive Bayes: `BernoulliNB` encaja con variables binarias (el one-hot lo es; cap. 4); ajustar `alpha` y `fit_prior`.
- Regresión logística L1/L2: `C` (log), `penalty`; solver `saga` (admite L1). El entregable 1 usó newton-cg con L2.
- Árbol de decisión: `max_depth`, `min_samples_leaf`, `ccp_alpha` (poda, cap. 5), `criterion`.
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `criterion` (tabla del cap. 5).
- XGBoost: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda` (tabla del cap. 5), `tree_method='hist'`, parada temprana. Para el balanceo usa pesos por muestra (`scale_pos_weight` solo vale para binario).
- **SVM**: el kernel exacto es cuadrático en n y no cabe (270 mil casos). Siguiendo el capítulo 6, usa **aproximación de kernel (Nystroem/RBFSampler con m ∈ {100, 300, 1000}) + LinearSVC o SGDClassifier**, y/o LinearSVC solo. Documenta el compromiso entre m, tiempo y desempeño. Para obtener probabilidades (ROC, Brier), envuelve en `CalibratedClassifierCV` (sigmoide e isotónica).
- Modelo original: **pendiente (D8)**. Diseña el código para que añadir un octavo modelo sea una entrada más en un registro de modelos (nombre, estimador, espacio de búsqueda, soporte de pesos).

**D7. Plan estadístico** (se ejecuta después de la corrida, con las predicciones guardadas):
- Friedman + Nemenyi (diagrama de diferencia crítica) sobre el **mejor de cada familia** (7 más el original) con balanced accuracy por pliegue externo. Con solo 3 pliegues la potencia es baja: se declara.
- DeLong **uno contra el resto** por clase sobre las predicciones de los 3 pliegues externos juntas (DeLong está definido para dos clases), con corrección de Holm, bootstrap **por establecimiento** para los intervalos y delta de Cliff como tamaño de efecto.
- El material de clase (cap. 6) propone además la prueba t corregida de Nadeau y Bengio sobre validación cruzada repetida; se menciona, pero el PDF manda.

## Pipeline y fugas de datos
- Todo lo que aprende de los datos (imputación, one-hot con `min_frequency`, balanceo, calibración) vive dentro del pipeline, ajustado solo con el entrenamiento de cada pliegue.
- Mantén `prepare_features` (sin estadísticas aprendidas) como primer paso, igual que en el entregable 1.
- Bayesiano (Optuna) y Genético (DEAP): el PDF exige implementar a mano el pipeline completo dentro de cada iteración, sin fugas. Documenta: modelo sustituto y función de adquisición (Optuna/TPE), operadores de selección/cruce/mutación, probabilidades, elitismo, tamaño de población y evolución de la diversidad genética.

## Qué debe producir el código (antes de la mañana)
1. **Registro maestro** (CSV o parquet): por combinación (modelo, balanceo, optimizador) y pliegue externo: hiperparámetros finales, métricas del bucle externo (balanced accuracy, F1 macro, AUC macro, MCC, recall por clase, log-loss), tiempo de ejecución (total y por evaluación), número de evaluaciones, semilla.
2. **Predicciones del bucle externo guardadas** (probabilidades por clase, `y_true`, `establishment_id`, pliegue) por combinación: sin ellas no se pueden hacer después DeLong, calibración, matriz de confusión ni curvas ROC.
3. **Curva de desempeño en cualquier momento** (mejor valor hasta la evaluación t) para cada optimizador.
4. Semilla global única (42) propagada a numpy, scikit-learn, XGBoost y los algoritmos genéticos; módulos separados (carga/preprocesamiento, modelos, optimizadores, evaluación); docstrings NumPy o Google en funciones reutilizadas.
5. Registro de tiempos y de las advertencias (convergencia, SMOTE sin vecinos suficientes).

## Orden de trabajo sugerido
0. Revisa el entorno y crea el nuevo (con versiones fijadas). Confirma con Ivan antes de instalar paquetes.
1. Prueba de humo con una muestra pequeña: un modelo, un balanceo, un optimizador; verifica ausencia de fugas y mide tiempo.
2. Cuenta las muertes por entrenamiento externo e interno y ajusta `k_neighbors`.
3. Mide el tiempo por modelo y propone a Ivan el presupuesto final.
4. Escribe el resto de modelos, balanceos y optimizadores sobre el registro.
5. Lanza la corrida nocturna con reanudación y registro. Deja un script que resuma el avance.
6. Por la mañana: revisa el registro, errores y advertencias antes de seguir.

## Desviaciones respecto al PDF (documentar todas en el JBook)
- 108 combinaciones en vez de 112 (KNN × class_weight no aplica).
- Regresión omitida (el objetivo es categórico).
- 3 pliegues externos por la estructura temporal (el material de clase usa 5); la desviación estándar se subestima por el solapamiento de ventanas.
- 30 evaluaciones por optimizador (el material de clase sugiere 60).
- Búsqueda sobre una muestra del 25 % con reentrenamiento al 100 %.
- SVM: aproximación de kernel o lineal en lugar de kernel exacto.
- Friedman/Nemenyi solo sobre el mejor de cada familia.

## Pendiente (no depende del código)
- **D8: modelo original.** Ivan exportará de Scopus y Web of Science "mañana" (el día siguiente a esta sesión) las dos listas del método del profesor (aplicación y "modelos mejorados"); se verifican y se elige entre ~5 candidatos. Criterio del profesor (clases 1, 2 y 6), todas condiciones a la vez: (a) no usado en la aplicación según la literatura; (b) no red neuronal; (c) no un simple `import` y correr: hay que programar y estructurar el modelo (en la clase 1 parece nombrar `import lightgbm` como ejemplo de lo que no vale; la transcripción está ruidosa); (d) debe superar con diferencia estadísticamente significativa a los modelos del curso. Por (c), un CatBoost o LightGBM corrido tal cual probablemente no pasa, aunque no se haya usado en OSHA (inferencia). Un híbrido o variante implementada por Ivan sí cuadra con el ejemplo del profesor (el "Ridge + XGBoost" de un paper).
- Análisis posteriores a la corrida: matriz de confusión, ROC, calibración (diagrama de fiabilidad, Brier, ECE, recalibración Platt/isotónica), SHAP y LIME del mejor modelo, pruebas estadísticas, complejidad O(·) con tiempos empíricos (tabla modelo estándar vs optimizado), JBook y sustentación de 10 minutos.
- EDA mejorado: el PDF pide hacer todas las correcciones de la primera entrega (brecha entrenamiento–validación, tratamiento de Death, costo del 55 % descartado, código de entrenamiento visible, `time_unknown`).

## Notas de cuidado
- El profesor recomienda en el material de clase que "un modelo complejo solo se justifica si supera de forma clara a una referencia simple". Mantén el Dummy y la logística como referencias dentro de la misma validación anidada, aunque el profesor dijo que el Dummy no va en la entrega final de la tesis.
- Death tiene ~37 casos en entrenamiento: sus métricas por pliegue son muy ruidosas. Reporta intervalos (Wilson) para el recall de Death.
- Antes de afirmar un resultado en el texto, verifícalo contra el registro maestro.
