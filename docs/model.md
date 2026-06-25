# Modelo de Clasificación

SAPO AI utiliza aprendizaje automático supervisado para clasificar eventos acústicos en dos clases:

- **DISPARO**
- **NO DISPARO**

## Modelo actual: SAPO

El modelo actual se denomina **SAPO**. Corresponde a un clasificador basado en **Random Forest**, entrenado con características acústicas extraídas de audios etiquetados.

Artefactos principales:

```text
models/sapo.pkl
models/sapo_scaler.pkl
```

`sapo.pkl` contiene el modelo entrenado. `sapo_scaler.pkl` contiene el escalador utilizado para transformar las características antes de la clasificación.

## Modelo anterior: Sapito

**Sapito** es la versión anterior del modelo. Se conserva en el repositorio como referencia histórica y respaldo del desarrollo.

Artefactos:

```text
models/sapito.pkl
models/sapito_scaler.pkl
```

La evolución de Sapito a SAPO representa una consolidación del proyecto: SAPO es el modelo actual presentado con identidad de producto, pipeline de app y documentación técnica.

## Variables de entrada

El modelo trabaja con 18 características:

```text
mfcc_1
mfcc_2
mfcc_3
mfcc_4
mfcc_5
mfcc_6
mfcc_7
mfcc_8
mfcc_9
mfcc_10
mfcc_11
mfcc_12
mfcc_13
zcr
rms
spectral_centroid
spectral_bandwidth
spectral_rolloff
```

Estas variables resumen propiedades temporales, energéticas y espectrales del audio.

## Clasificación y predicción

En este proyecto, **clasificación** es el problema general: asignar cada archivo a una de dos clases posibles. SAPO AI resuelve una clasificación binaria porque solo existen dos salidas esperadas.

**Predicción** es la acción individual de aplicar el modelo ya entrenado sobre un archivo nuevo para obtener una clase. Por ejemplo, cuando el usuario sube un MP3 y presiona **Analizar con SAPO**, la app realiza una predicción dentro del problema de clasificación.

## Entrenamiento limpio sin data leakage

El script:

```text
src/models/train_random_forest_clean.py
```

entrena un Random Forest evitando data leakage. El flujo correcto es:

1. Cargar `data/processed/features.csv`.
2. Separar características y etiquetas.
3. Dividir en train y test.
4. Ajustar el escalador solo con `X_train`.
5. Transformar `X_train` y `X_test`.
6. Entrenar Random Forest.
7. Evaluar en test.
8. Guardar modelo y scaler.

Este orden es importante porque impide que el conjunto de prueba influya en el escalado durante el entrenamiento.

## Random Forest

Random Forest combina múltiples árboles de decisión. Cada árbol aprende patrones parciales del dataset y el bosque decide mediante votación. Esta técnica es apropiada para SAPO AI porque:

- Maneja relaciones no lineales entre características acústicas.
- Suele ser robusta ante ruido moderado.
- Permite buen desempeño sin requerir una red neuronal profunda.
- Es interpretable a nivel de importancia de variables y métricas de clasificación.
