"""Build the Spanish results page from the completed group-time study."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
r = json.loads((root / 'osha_group_time_results/evaluation.json').read_text(encoding='utf-8'))
lr = r['holdout_metrics']['LogisticRegression']
dummy = r['holdout_metrics']['DummyClassifier']
death = r['classification_reports']['LogisticRegression']['1']
labels = ['Fallecimiento', 'Días de ausencia', 'Traslado/restricción', 'Otro caso registrable']
rows = []
for metric, label in [
    ('accuracy', 'Exactitud'), ('balanced_accuracy', 'Exactitud balanceada'),
    ('macro_f1', 'F1 macro'), ('roc_auc_macro_ovr', 'ROC AUC macro OvR'),
]:
    rows.append(f"| {label} | {lr[metric]:.4f} | {dummy[metric]:.4f} |")
death_rows = []
for k, label in zip(['1', '2', '3', '4'], labels):
    m = r['classification_reports']['LogisticRegression'][k]
    death_rows.append(f"| {label} | {int(m['support']):,} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} |")
ci = r['death_sensitivity_wilson_ci95']
sectors = len(r['sector_analysis']) // 2
cv_lr = [x for x in r['cv'] if x['model'] == 'LogisticRegression']
cv_better = sum(
    a['metrics']['balanced_accuracy'] > b['metrics']['balanced_accuracy']
    for a, b in zip(cv_lr, [x for x in r['cv'] if x['model'] == 'DummyClassifier'])
)

text = f'''# Resultados y conclusiones

La evaluación primaria usa **{r['test_rows']:,} incidentes de octubre–diciembre de 2023**, todos de establecimientos ausentes del desarrollo. El desarrollo contiene **{r['train_rows']:,} casos** de **{r['train_groups']:,} establecimientos**; la prueba contiene **{r['test_groups']:,} establecimientos**. No hay establecimientos compartidos. Se purgaron siete días antes de la prueba y de cada ventana de validación. La asignación aleatoria de empresas (semilla 42) se hizo sin usar los resultados. Reservar 60 % de establecimientos es un compromiso de diseño, no un requisito del curso: el preanálisis de soporte da 39 fallecimientos en prueba y conserva más de 269.000 casos para entrenar.

Se excluyeron **{r['excluded_2023_rows']:,} filas de 2023** que no pertenecen a las celdas de desarrollo o prueba de este diseño: casos de establecimientos reservados fuera del trimestre de prueba, casos Q4 de establecimientos de desarrollo y la ventana temporal purgada.

| Métrica | Regresión logística | Referencia trivial |
|---|---:|---:|
{chr(10).join(rows)}

En validación interna por empresa y tiempo, la regresión logística obtiene mayor exactitud balanceada que la referencia en **{cv_better} de {len(cv_lr)} ventanas**. Las métricas por ventana y fracción de aprendizaje se muestran en el notebook; la pequeña cantidad de fallecimientos por pliegue hace inestable la comparación de esa clase.

## Métricas por resultado

| Resultado | Casos de prueba | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(death_rows)}

La prueba contiene **{int(death['support'])} fallecimientos en {r['test_death_groups']} establecimientos distintos**. Para la regresión, la sensibilidad de fallecimiento es **{death['recall']:.1%}** (intervalo binomial de Wilson del 95 %: **{ci[0]:.1%}–{ci[1]:.1%}**; [Wilson, 1927](https://doi.org/10.2307/2276774)). La planificación de sensibilidad debe fijar precisión y soporte positivo mediante una distribución binomial; no hay un mínimo universal [Flahault et al., 2005](https://doi.org/10.1016/j.jclinepi.2004.12.009). Treinta y nueve positivos equivalen aproximadamente a ±15 puntos porcentuales si la sensibilidad real ronda 50 %, pero no es un umbral del curso ni garantía de precisión. El intervalo observado es ancho; no permite sostener que el modelo detecte fallecimientos de forma fiable. Se mantienen las cuatro clases y no se cambia el umbral después de ver la prueba.

## Sectores industriales

Se reportan **{sectors} sectores**, seleccionados por tener al menos 1.000 casos de desarrollo, criterio fijado antes de evaluar su desempeño. Sus resultados incluyen soporte por clase e intervalos agrupados; no son tasas de lesión ni rankings de seguridad. En sectores sin todas las clases, AUC macro no se calcula.

## Incertidumbre y límites

El bootstrap remuestrea establecimientos completos y conserva la comparación pareada de modelos. Sus intervalos están condicionados al trimestre observado y a modelos ya ajustados; no cubren incertidumbre de entrenamiento ni shocks temporales comunes. La clasificación es retrospectiva: la fecha del incidente no demuestra que todos los predictores o etiquetas estuvieran disponibles históricamente. El proyecto no estima tasas de lesión, no establece causas y no valida una alerta operativa.

El cuarto trimestre y métricas de la evaluación anterior ya se habían examinado durante el proyecto. La asignación nueva de establecimientos no se eligió por sus resultados, pero la prueba **no es completamente inédita** y esa limitación se declara.

## Reproducibilidad y cumplimiento

La [matriz de requisitos](compliance_matrix.md) registra la evidencia y los límites. El notebook conserva 65 celdas. La ejecución registra la partición, versiones, huellas de los datos y del código, predicciones, métricas y convergencia. El modelo es una regresión logística base frente a `DummyClassifier`, sin búsqueda de hiperparámetros ni cambios de objetivo.
'''
(root / 'results_summary.md').write_text(text, encoding='utf-8')
print('Group-time results summary saved.')
