# Resultados y conclusiones

La evaluación primaria usa **127,391 incidentes de octubre–diciembre de 2023**, todos de establecimientos ausentes del desarrollo. El desarrollo contiene **269,516 casos** de **34,029 establecimientos**; la prueba contiene **34,182 establecimientos**. No hay establecimientos compartidos. Se purgaron siete días antes de la prueba y de cada ventana de validación. La asignación aleatoria de empresas (semilla 42) se hizo sin usar los resultados. Reservar 60 % de establecimientos es un compromiso de diseño, no un requisito del curso: el preanálisis de soporte da 39 fallecimientos en prueba y conserva más de 269.000 casos para entrenar.

Se excluyeron **493,465 filas de 2023** que no pertenecen a las celdas de desarrollo o prueba de este diseño: casos de establecimientos reservados fuera del trimestre de prueba, casos Q4 de establecimientos de desarrollo y la ventana temporal purgada.

| Métrica | Regresión logística | Referencia trivial |
|---|---:|---:|
| Exactitud | 0.5238 | 0.3624 |
| Exactitud balanceada | 0.4019 | 0.2500 |
| F1 macro | 0.3966 | 0.1330 |
| ROC AUC macro OvR | 0.6933 | 0.5000 |

En validación interna por empresa y tiempo, la regresión logística obtiene mayor exactitud balanceada que la referencia en **3 de 3 ventanas**. Las métricas por ventana y fracción de aprendizaje se muestran en el notebook; la pequeña cantidad de fallecimientos por pliegue hace inestable la comparación de esa clase.

## Métricas por resultado

| Resultado | Casos de prueba | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|---:|
| Fallecimiento | 39 | 0.0003 | 0.0256 | 0.0007 |
| Días de ausencia | 46,169 | 0.6005 | 0.4357 | 0.5050 |
| Traslado/restricción | 37,231 | 0.4776 | 0.5617 | 0.5163 |
| Otro caso registrable | 43,952 | 0.5456 | 0.5846 | 0.5644 |

La prueba contiene **39 fallecimientos en 39 establecimientos distintos**. Para la regresión, la sensibilidad de fallecimiento es **2.6%** (intervalo binomial de Wilson del 95 %: **0.5%–13.2%**; [Wilson, 1927](https://doi.org/10.2307/2276774)). La planificación de sensibilidad debe fijar precisión y soporte positivo mediante una distribución binomial; no hay un mínimo universal [Flahault et al., 2005](https://doi.org/10.1016/j.jclinepi.2004.12.009). Treinta y nueve positivos equivalen aproximadamente a ±15 puntos porcentuales si la sensibilidad real ronda 50 %, pero no es un umbral del curso ni garantía de precisión. El intervalo observado es ancho; no permite sostener que el modelo detecte fallecimientos de forma fiable. Se mantienen las cuatro clases y no se cambia el umbral después de ver la prueba.

## Sectores industriales

Se reportan **12 sectores**, seleccionados por tener al menos 1.000 casos de desarrollo, criterio fijado antes de evaluar su desempeño. Sus resultados incluyen soporte por clase e intervalos agrupados; no son tasas de lesión ni rankings de seguridad. En sectores sin todas las clases, AUC macro no se calcula.

## Incertidumbre y límites

El bootstrap remuestrea establecimientos completos y conserva la comparación pareada de modelos. Sus intervalos están condicionados al trimestre observado y a modelos ya ajustados; no cubren incertidumbre de entrenamiento ni shocks temporales comunes. La clasificación es retrospectiva: la fecha del incidente no demuestra que todos los predictores o etiquetas estuvieran disponibles históricamente. El proyecto no estima tasas de lesión, no establece causas y no valida una alerta operativa.

El cuarto trimestre y métricas de la evaluación anterior ya se habían examinado durante el proyecto. La asignación nueva de establecimientos no se eligió por sus resultados, pero la prueba **no es completamente inédita** y esa limitación se declara.

## Reproducibilidad y cumplimiento

La [matriz de requisitos](compliance_matrix.md) registra la evidencia y los límites. El notebook conserva 65 celdas. La ejecución registra la partición, versiones, huellas de los datos y del código, predicciones, métricas y convergencia. El modelo es una regresión logística base frente a `DummyClassifier`, sin búsqueda de hiperparámetros ni cambios de objetivo.
