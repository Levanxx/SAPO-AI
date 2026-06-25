# Evaluación del Modelo SAPO

SAPO AI utiliza métricas de clasificación para evaluar el desempeño del modelo Random Forest en la detección acústica de disparos.

## Métricas principales

El modelo actual reporta un accuracy aproximado de **97%**. En la evaluación documentada se obtuvo:

| Métrica | Valor |
|---|---:|
| Accuracy | 96.89% |
| Precision | 95.77% |
| Recall | 97.63% |
| F1-Score | 96.69% |

## Matriz de confusión

```text
[[793  31]
 [ 17 701]]
```

Interpretación:

- 793 audios **NO DISPARO** fueron clasificados correctamente.
- 701 audios **DISPARO** fueron detectados correctamente.
- 31 audios fueron falsos positivos.
- 17 audios fueron falsos negativos.

## Revisión de overfitting

Resultados reportados:

| Indicador | Valor |
|---|---:|
| Train Accuracy | 99.89% |
| Test Accuracy | 96.89% |
| Diferencia | 3% |
| Cross Validation media | 97.15% |

La diferencia entre entrenamiento y prueba es moderada y la validación cruzada mantiene un rendimiento cercano al test. Por ello, la conclusión documentada es:

**No se observa overfitting fuerte.**

## Modelo limpio sin data leakage

El entrenamiento limpio evita data leakage ajustando el scaler únicamente con los datos de entrenamiento. Esta decisión es importante porque el escalado previo sobre todo el dataset puede transferir información estadística del test hacia el entrenamiento.

Script relacionado:

```text
src/models/train_random_forest_clean.py
```

## Alcance de las métricas

Las métricas describen el comportamiento del modelo bajo el dataset y la partición evaluada. No sustituyen validaciones adicionales en escenarios reales, con otros dispositivos de grabación, diferentes niveles de ruido o condiciones acústicas no representadas en el conjunto de entrenamiento.

## Limitaciones actuales

- El modelo depende de la representatividad de los audios disponibles.
- Las grabaciones con ruido extremo, saturación, eco o baja calidad pueden reducir el desempeño.
- La aplicación no realiza análisis visual del video; solo extrae y clasifica el audio.
- La ubicación GPS depende del navegador, permisos del usuario y servicios externos.
- El sistema es un apoyo de clasificación acústica y no debe considerarse una conclusión forense definitiva sin revisión complementaria.

## Mejoras futuras

- **SAPO Vision:** incorporación de análisis visual de video.
- **SAPO Fusion:** combinación de evidencia acústica y visual.
- Historial de detecciones por fecha, archivo, resultado y ubicación.
- Mejor integración de mapas y almacenamiento geoespacial.
- Despliegue web con autenticación y monitoreo.
- Evaluación con datasets más amplios, balanceados y capturados en escenarios reales.
