# SAPO AI - Descripción General

**SAPO AI** significa **Sistema Acústico de Protección y Observación**. Es una aplicación de inteligencia artificial orientada a la clasificación acústica de disparos a partir de archivos de audio y video.

El sistema recibe un archivo, extrae o carga su señal acústica, calcula características representativas del audio y utiliza un modelo Random Forest para clasificar el contenido como **DISPARO** o **NO DISPARO**. Si se detecta un disparo, la app muestra una alerta y solicita ubicación GPS para visualizar un mapa.

## Objetivo del sistema

Construir un sistema de clasificación acústica capaz de identificar eventos sonoros asociados a disparos mediante procesamiento digital de señales, extracción de características y aprendizaje automático.

## Problema que resuelve

La revisión manual de grabaciones puede ser lenta, subjetiva y difícil de escalar. SAPO AI automatiza la clasificación inicial del contenido acústico y entrega una respuesta clara para apoyar flujos de monitoreo, análisis y alerta temprana.

## Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python |
| Interfaz | Streamlit |
| Procesamiento de audio | Librosa, SoundFile |
| Procesamiento de video | MoviePy |
| Datos y cálculo | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Serialización | Joblib |
| Visualización | Matplotlib, Folium, Leaflet, OpenStreetMap |
| Pruebas | Pytest |

## Estructura del repositorio

```text
DeteccionDeDisparos/
├── data/
│   ├── raw/
│   ├── processed/
│   └── videos IA/
├── docs/
├── models/
│   ├── sapo.pkl
│   ├── sapo_scaler.pkl
│   ├── sapito.pkl
│   └── sapito_scaler.pkl
├── notebooks/
├── reports/
├── src/
│   ├── app/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── preprocessing/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Componentes principales

- `src/app/streamlit_app.py`: aplicación web de SAPO AI.
- `src/app/predict.py`: función de predicción sobre características ya calculadas.
- `src/features/extract_features.py`: extracción de MFCC, ZCR, RMS y características espectrales.
- `src/data/load_data.py`: carga uniforme de audio, normalización, recorte y padding.
- `src/data/prepare_data.py`: escalado de características.
- `src/models/train_random_forest_clean.py`: entrenamiento limpio de Random Forest sin data leakage.
- `src/models/check_overfitting.py`: revisión de desempeño en train/test y validación cruzada.
- `models/sapo.pkl`: modelo actual SAPO.
- `models/sapo_scaler.pkl`: escalador asociado al modelo SAPO.

## Identidad del proyecto

SAPO AI se presenta como una solución académica y profesional de clasificación acústica. Su identidad combina el análisis técnico de señales con una experiencia de uso clara: cargar, analizar, clasificar y alertar.

Autor: **Levanx**.
