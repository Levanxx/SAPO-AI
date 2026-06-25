# Uso de SAPO AI

Esta guía resume cómo preparar el entorno, ejecutar la app, generar características, entrenar el modelo limpio y evaluar overfitting.

## Requisitos

- Python 3.x
- Acceso a la raíz del repositorio
- Dependencias listadas en `requirements.txt`

## Instalación del entorno

Desde la raíz del proyecto:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Activación del entorno virtual

macOS o Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Ejecutar la aplicación con Streamlit

```bash
streamlit run src/app/streamlit_app.py
```

La app permite cargar:

- WAV
- MP3
- MP4
- MOV
- AVI

## Cómo usar la aplicación

1. Ejecutar Streamlit.
2. Subir un archivo de audio o video.
3. Revisar los datos del archivo cargado.
4. Presionar **Analizar con SAPO**.
5. Leer el resultado:
   - **DISPARO:** SAPO AI muestra alerta y solicita ubicación GPS.
   - **NO DISPARO:** SAPO AI informa que no se detectó disparo.

Si se sube un video, la app extrae automáticamente su audio antes de clasificarlo.

## Generar características acústicas

El archivo principal para generar `features.csv` es:

```text
main.py
```

Ejecución:

```bash
python main.py
```

El script lee audios desde:

```text
data/processed/
```

con esta estructura esperada:

```text
data/processed/
├── disparos/
└── no_disparos/
```

Luego genera:

```text
data/processed/features.csv
```

## Preparar dataset escalado

Para generar un dataset escalado y guardar un scaler general:

```bash
python src/data/prepare_data.py
```

Salida esperada:

```text
data/processed/features_scaled.csv
models/scaler.pkl
```

## Entrenar el modelo limpio

Para entrenar un Random Forest evitando data leakage:

```bash
python src/models/train_random_forest_clean.py
```

Este flujo divide primero los datos y ajusta el escalador solo sobre el conjunto de entrenamiento.

Salida definida por el script:

```text
models/random_forest_model_clean.pkl
models/scaler_clean.pkl
```

## Evaluar overfitting

```bash
python src/models/check_overfitting.py
```

El script compara accuracy en entrenamiento y prueba, calcula precision, recall, F1-score y ejecuta validación cruzada.

## Pruebas automatizadas

Para ejecutar pruebas:

```bash
pytest
```

Las pruebas disponibles validan aspectos básicos de carga, tipos de audio, archivos no vacíos y waveforms.

## Consideraciones de datos

Los audios y CSV generados pueden ser pesados o locales. Por ello, el repositorio puede no incluir todos los datos usados durante el entrenamiento. Para reproducir el flujo completo, se requiere contar con audios organizados por clase.
