# Requisitos, evidencia y límites

Referencia: *Entregable1MachineLearningProject.pdf*, secciones de datos, EDA, modelado y entrega; revisión centrada en pp. 3, 5–6 y 9–11. El peso del EDA exige interpretar hallazgos y decisiones, no solo producir gráficas. Esta matriz no sustituye la evaluación docente.

| Requisito | Evidencia en la entrega | Estado y límite |
|---|---|---|
| Datos, ≥20.000 filas, entidades, n/p y diccionario | Secciones 1–4 y salida del pipeline | Documentado: 890.934 filas originales; 269.516 desarrollo, 127.391 prueba, restantes excluidas por diseño grupo-tiempo; entidades y dimensión codificada informadas |
| Fuente, acceso, calidad y ética | Fuentes oficiales; cobertura en sección 1; auditorías de faltantes, códigos y claves | Documentado; acceso público no equivale a licencia declarada ni anonimato |
| EDA univariado, bivariado, multivariado | Distribuciones, tablas por resultado, Cramér/NMI, horas, SVD | Implementado; p-valores clásicos ilustrativos, no inferencia válida con filas dependientes; no causalidad |
| Faltantes, valores imposibles y transformaciones | EDA de calidad, máscaras de hora, SOC, categorías desconocidas | Implementado; código 0 de establecimiento no documentado se conserva literalmente |
| Claves repetidas y separación de la prueba | Auditoría de ID y claves dentro del desarrollo; partición agrupada por establecimiento | Implementado: ningún ID ITA repetido en desarrollo y ningún establecimiento compartido entre desarrollo y prueba; las claves repetidas se señalan, no se eliminan automáticamente |
| Dependencia entre entidades y separación agrupada | Identificadores de partición, conteos por empresa, CV agrupada y bootstrap | **Separación primaria estricta:** empresas distintas en desarrollo y prueba; cero solapamiento. Validación interna también separa grupos |
| Prueba separada por grupo y tiempo; sin ajuste por métricas de prueba | Partición fija por establecimiento y fecha; regresión logística base frente a Dummy, sin búsqueda de hiperparámetros | **Implementado:** establecimientos distintos entre desarrollo y prueba; la evaluación primaria usa un periodo posterior y no se usa para seleccionar empresas, variables, modelos ni parámetros por desempeño |
| Fechas: orden temporal, validación futura, gap | Corte octubre; tres ventanas futuras y purga temporal | Implementado: entrenamiento anterior a validación/prueba, separación por grupos y purga fija de siete días. Persisten dependencia temporal común y disponibilidad histórica desconocida |
| EDA temporal | Media/varianza móvil, distribuciones por semana/mes, STL, ADF/KPSS, ACF/PACF, deriva Q1–Q3 | Implementado en desarrollo. ADF/KPSS inconclusos; un solo año no prueba ciclo anual estable |
| Rezagos, calendario, cambios y heterogeneidad por entidad | Proporciones diarias manufactura/ausencia en rezagos 0/1/7/14; ventanas de 28 días; resúmenes trimestrales por entidad | Descriptivo: sin causalidad ecológica, fechas de cambio confirmadas ni atribución a festivos; selección de entidades ≥30 casos y presencia en Q1–Q3 explícita |
| Análisis espacial | Estado categórico; sin coordenadas en los predictores | Mapas de distancias, variogramas y autocorrelación geográfica no aplicables sin georreferenciación adecuada |
| Modelos y pipeline | Dummy prior + logística balanceada; imputación y one-hot dentro de cada ajuste | Implementado, sin tuning; ocho predictores, cuatro clases; ajustes primarios convergentes |
| Métricas y explicación | Métricas macro/por clase, ROC/PR, calibración, coeficientes y curvas de aprendizaje | Implementado; mala precisión de Death explícita; brecha train/validación no desaparece |
| Incertidumbre y análisis secundario | Bootstrap pareado por establecimiento; Wilson para sensibilidad Death; sectores | Condicional al trimestre/modelos; Wilson para 39 fallecimientos en empresas distintas es amplio y no respalda una afirmación precisa de detección. No cubre choques comunes de calendario ni variación de entrenamiento |
| Hallazgo → implicación → decisión | Tabla de síntesis EDA, sección 13 | Implementado; sin cambiar datos o objetivo para mejorar desempeño |
| Reproducibilidad y Jupyter Book | Notebook de 65 celdas, dependencias, semillas, código de cribado, manifiesto de artefactos, verificador | Verificable por ejecución y extracción del paquete; la matriz resume evidencia y límites para revisión académica |

## Antes de entregar

Documentar la purga temporal y la exclusión de filas necesarias para separar empresa y tiempo. No afirmar que la sensibilidad de Death está estimada con precisión. Añadir los nombres de los estudiantes en la portada y revisar las conclusiones conjuntamente. No continuar afinando modelos sobre estos resultados.
