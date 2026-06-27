import streamlit as st
import pandas as pd
import librosa
import joblib
import tempfile
from pathlib import Path
from moviepy import VideoFileClip
import os

st.set_page_config(
    page_title="SAPO AI",
    page_icon="🐸",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07150d 0%, #0b1f14 45%, #05100a 100%);
        color: #eafff0;
    }

    section[data-testid="stSidebar"] {
        background: #06140c;
        border-right: 1px solid rgba(91, 255, 145, 0.18);
    }

    .hero {
        background: linear-gradient(135deg, rgba(25, 120, 64, 0.35), rgba(8, 36, 20, 0.92));
        border: 1px solid rgba(91, 255, 145, 0.25);
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    }

    .subtitle {
        color: #7dffae;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .main-title {
        font-size: 48px;
        font-weight: 900;
        margin: 8px 0 4px 0;
        color: #eafff0;
    }

    .description {
        color: #b8e8c7;
        font-size: 16px;
        margin-top: 8px;
    }

    .card {
        background: rgba(8, 28, 16, 0.92);
        border: 1px solid rgba(91, 255, 145, 0.18);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
    }

    .upload-card {
        border: 2px dashed rgba(125, 255, 174, 0.45);
        border-radius: 20px;
        padding: 34px;
        text-align: center;
        background: rgba(10, 43, 24, 0.72);
        margin-bottom: 18px;
    }

    .metric-label {
        color: #7dffae;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .big-text {
        font-size: 22px;
        font-weight: 800;
        color: #eafff0;
    }

    .small-muted {
        color: #a8cdb4;
        font-size: 14px;
    }

    .sapo-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(125, 255, 174, 0.12);
        border: 1px solid rgba(125, 255, 174, 0.35);
        color: #7dffae;
        font-weight: 800;
        font-size: 13px;
        margin-top: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(125, 255, 174, 0.35);
        background: #21c45d;
        color: #031107;
        font-weight: 800;
        padding: 0.75rem 1rem;
    }

    div.stButton > button:hover {
        background: #7dffae;
        color: #031107;
        border: 1px solid #7dffae;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/sapo.pkl")
SCALER_PATH = Path("models/sapo_scaler.pkl")

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

def convertir_video_a_audio(video_path):
    audio_path = video_path + ".wav"

    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        raise ValueError("El video no tiene audio.")

    video.audio.write_audiofile(
        audio_path,
        codec="pcm_s16le",
        logger=None
    )

    video.close()

    return audio_path

def mostrar_mapa_con_gps(confianza, archivo):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; background: #07150d; font-family: sans-serif; }
            #map { height: 400px; width: 100%; border-radius: 14px; }
            #address { padding: 12px; background: rgba(8,28,16,0.95); border-radius: 10px; margin-top: 8px; color: #eafff0; font-size: 14px; }
            #status { color: #7dffae; padding: 8px 0; font-size: 14px; }
        </style>
    </head>
    <body>
        <div id="status">📍 Obteniendo ubicación GPS...</div>
        <div id="map"></div>
        <div id="address"></div>
        <script>
        var map, marker;
        function initMap(lat, lon) {
            if (!map) {
                map = L.map('map').setView([lat, lon], 16);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap'
                }).addTo(map);
            } else { map.setView([lat, lon], 16); }
            if (marker) marker.remove();
            marker = L.marker([lat, lon]).addTo(map);
        }
        function reverseGeocode(lat, lon) {
            fetch('https://nominatim.openstreetmap.org/reverse?lat=' + lat + '&lon=' + lon + '&format=json')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('address').innerHTML = '📍 <b>Dirección:</b> ' + (data.display_name || 'No disponible');
                    document.getElementById('status').style.display = 'none';
                })
                .catch(() => {
                    document.getElementById('address').innerHTML = '📍 Dirección no disponible';
                    document.getElementById('status').style.display = 'none';
                });
        }
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    var lat = pos.coords.latitude;
                    var lon = pos.coords.longitude;
                    initMap(lat, lon);
                    reverseGeocode(lat, lon);
                    navigator.sendBeacon("https://mofuel.app.n8n.cloud/webhook/b1b7a026-9424-4218-a1a1-d8d0fedebfbd", JSON.stringify({
                        resultado: "DISPARO",
                        confianza: "{{CONFIANZA}}",
                        latitud: lat,
                        longitud: lon,
                        archivo: "{{ARCHIVO}}",
                        timestamp: new Date().toISOString()
                    }));
                },
                function(err) {
                    document.getElementById('status').innerHTML = '❌ Permiso de ubicación denegado. Actívalo en tu navegador.';
                    document.getElementById('status').style.color = '#ff6b6b';
                },
                { enableHighAccuracy: true }
            );
        } else {
            document.getElementById('status').innerHTML = '❌ GPS no soportado';
        }
        </script>
    </body>
    </html>
    """
    html = html.replace("{{CONFIANZA}}", str(confianza))
    html = html.replace("{{ARCHIVO}}", archivo)
    st.components.v1.html(html, height=500)

def predecir_audio(uploaded_file):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if extension in ["mp4", "mov", "avi"]:
        temp_audio_path = convertir_video_a_audio(temp_file_path)
    else:
        temp_audio_path = temp_file_path

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

    try:
        os.unlink(temp_file_path)
        if temp_audio_path != temp_file_path:
            os.unlink(temp_audio_path)
    except:
        pass

    if prediction == 1:
        return "DISPARO", round(confidence, 2)
    else:
        return "NO DISPARO", round(confidence, 2)

st.sidebar.markdown("## 🐸 SAPO AI")
st.sidebar.caption("Sistema Acústico de Protección y Observación")
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="card">
    <p class="metric-label">Modelo activo</p>
    <p class="big-text">SAPO</p>
    <p class="small-muted">Random Forest · Clasificación binaria</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="card">
    <p class="metric-label">Entradas soportadas</p>
    <p class="small-muted">WAV · MP3 · MP4 · MOV · AVI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="subtitle">Clasificación acústica inteligente</div>
    <div class="main-title">🐸 SAPO AI</div>
    <div class="description">
        Sistema Acústico de Protección y Observación para detectar eventos sonoros asociados a disparos.
    </div>
    <div class="sapo-badge">Escucha · Clasifica · Alerta</div>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([2.2, 1])

with col_main:
    st.markdown("""
    <div class="upload-card">
        <h3>Subir archivo de audio o video</h3>
        <p>Selecciona un archivo WAV, MP3, MP4, MOV o AVI. Si subes un video, SAPO extraerá el audio automáticamente.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Selecciona un archivo",
        type=["wav", "mp3", "mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        st.success("Archivo cargado correctamente.")

        extension = uploaded_file.name.split(".")[-1].lower()

        st.markdown("""
        <div class="card">
            <p class="metric-label">Archivo cargado</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Nombre", uploaded_file.name)

        with col_b:
            st.metric("Tipo", uploaded_file.type if uploaded_file.type else extension.upper())

        with col_c:
            st.metric("Tamaño", f"{round(uploaded_file.size / 1024, 2)} KB")

        if extension in ["wav", "mp3"]:
            st.audio(uploaded_file)
        else:
            st.video(uploaded_file)

        if st.button("Analizar con SAPO"):
            try:
                with st.spinner("SAPO está escuchando el archivo..."):
                    resultado, confianza = predecir_audio(uploaded_file)

                if resultado == "DISPARO":
                    st.error(f"DISPARO DETECTADO — Confianza: {confianza}%")
                    st.markdown("""
                    <div class="card">
                        <p class="metric-label">Alerta geoespacial</p>
                        <h3>Ubicación aproximada del dispositivo</h3>
                        <p class="small-muted">El navegador solicitará permiso para acceder al GPS.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    mostrar_mapa_con_gps(confianza, uploaded_file.name)
                else:
                    st.success(f"NO SE DETECTÓ DISPARO — Confianza: {confianza}%")

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Ocurrió un error al analizar el archivo: {e}")

    else:
        st.info("Sube un archivo para iniciar el análisis.")

        st.markdown("""
        <div class="card">
            <p class="metric-label">Estado del sistema</p>
            <h3>En espera de archivo</h3>
            <p class="small-muted">SAPO está listo para clasificar audio o video.</p>
        </div>
        """, unsafe_allow_html=True)

with col_side:
    st.markdown("""
    <div class="card">
        <p class="metric-label">Resultado</p>
        <h3>Clasificación binaria</h3>
        <p class="small-muted">El modelo clasifica el archivo como DISPARO o NO DISPARO.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p class="metric-label">Pipeline</p>
        <p class="small-muted">
            1. Carga del archivo<br>
            2. Extracción de audio si es video<br>
            3. Extracción de características<br>
            4. Escalado<br>
            5. Clasificación con SAPO
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p class="metric-label">Modelo</p>
        <h3>SAPO</h3>
        <p class="small-muted">Modelo Random Forest entrenado para detección acústica.</p>
    </div>
    """, unsafe_allow_html=True)