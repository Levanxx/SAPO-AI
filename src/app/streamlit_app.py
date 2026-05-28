import streamlit as st
import pandas as pd
import librosa
import joblib
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="AcousticForensics ML",
    page_icon="🎧",
    layout="wide"
)

# =========================
# ESTILOS
# =========================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1326;
        color: #dae2fd;
    }

    section[data-testid="stSidebar"] {
        background-color: #060e20;
    }

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #dae2fd;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #adc6ff;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 25px;
    }

    .card {
        background: rgba(19, 27, 46, 0.9);
        border: 1px solid rgba(173, 198, 255, 0.15);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .upload-card {
        border: 2px dashed rgba(173, 198, 255, 0.35);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background: rgba(19, 27, 46, 0.65);
    }

    .danger {
        background-color: #93000a;
        color: #ffdad6;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }

    .safe {
        background-color: #00320e;
        color: #72fe88;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }

    .metric-label {
        color: #c1c6d7;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #adc6ff;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/random_forest_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

FEATURE_COLUMNS = [
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
    "mfcc_11", "mfcc_12", "mfcc_13",
    "zcr", "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff"
]

def extraer_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = []

    for i in range(13):
        features.append(mfcc[i].mean())

    features.append(zcr.mean())
    features.append(rms.mean())
    features.append(spectral_centroid.mean())
    features.append(spectral_bandwidth.mean())
    features.append(spectral_rolloff.mean())

    return features


def predecir_audio(audio_file):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    features = extraer_features(temp_audio_path)

    df_features = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    scaled_features = scaler.transform(df_features)

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(scaled_df)[0]

    probabilities = model.predict_proba(scaled_df)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        return "DISPARO", round(confidence, 2)
    else:
        return "NO DISPARO", round(confidence, 2)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("AcousticForensics ML")
st.sidebar.caption("Forensic Unit · Precision ML v4.2")

st.sidebar.markdown("---")
st.sidebar.button("Dashboard")
st.sidebar.button("Detection History")
st.sidebar.button("Geospatial Analysis")
st.sidebar.button("System Health")
st.sidebar.button("Settings")

st.sidebar.markdown("---")
st.sidebar.button("+ New Analysis")

# =========================
# HEADER
# =========================

st.markdown('<div class="subtitle">Módulo de Vigilancia Activa</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Terminal de Análisis Acústico</div>', unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================

col_main, col_side = st.columns([2.2, 1])

with col_main:

    st.markdown("""
    <div class="upload-card">
        <h3>Subir Archivo de Audio</h3>
        <p>Arrastra o selecciona un archivo WAV o MP3 para análisis acústico.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
    "Selecciona un archivo de audio",
    type=["wav", "mp3"]
)

# =========================================
# VALIDACIÓN DE AUDIO
# =========================================

if uploaded_file is not None:

    # Mostrar mensaje
    st.success("Archivo cargado correctamente.")

    # Información del archivo
    file_details = {
        "Nombre": uploaded_file.name,
        "Tipo": uploaded_file.type,
        "Tamaño (KB)": round(uploaded_file.size / 1024, 2)
    }

    # Tarjeta de información
    st.markdown("""
    <div class="card">
        <p class="metric-label">ARCHIVO CARGADO</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(file_details)

    # Reproductor de audio
    st.audio(uploaded_file)
    if st.button("Analizar audio"):
        resultado, confianza = predecir_audio(uploaded_file)

        if resultado == "DISPARO":
            st.error("DISPARO DETECTADO")
        else:
            st.success("NO SE DETECTÓ DISPARO")

else:

    st.warning(
        "Aún no se ha cargado ningún archivo de audio."
    )

    st.markdown("""
    <div class="card">
        <p class="metric-label">Estado del Motor ML</p>
        <h3>Resultado de Clasificación</h3>
        <br>
        <p>Aún no se ha ejecutado una predicción.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Análisis Geoespacial Táctico</h3>
        <p>Mapa pendiente de integración.</p>
    </div>
    """, unsafe_allow_html=True)

with col_side:

    st.markdown("""
    <div class="card">
        <h3>Detecciones</h3>
        <p>Sin detecciones recientes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Telemetría de Entorno</h3>
        <p>Temp. Aire: 24.2 °C</p>
        <p>Humedad: 45%</p>
        <p>Presión: 1013 hPa</p>
        <p>Viento: 12 km/h</p>
    </div>
    """, unsafe_allow_html=True)