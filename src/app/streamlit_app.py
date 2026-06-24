import streamlit as st
import pandas as pd
import librosa
import joblib
import tempfile
from pathlib import Path
from moviepy import VideoFileClip

import requests
import folium
import os

st.set_page_config(
    page_title="AcousticForensics ML",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0b1326; color: #dae2fd; }
    section[data-testid="stSidebar"] { background-color: #060e20; }
    .main-title { font-size: 38px; font-weight: 700; color: #dae2fd; margin-bottom: 5px; }
    .subtitle { color: #adc6ff; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 25px; }
    .card { background: rgba(19, 27, 46, 0.9); border: 1px solid rgba(173, 198, 255, 0.15); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .upload-card { border: 2px dashed rgba(173, 198, 255, 0.35); border-radius: 12px; padding: 40px; text-align: center; background: rgba(19, 27, 46, 0.65); }
    .metric-label { color: #c1c6d7; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
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

def mostrar_mapa_con_gps():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; background: #0b1326; font-family: sans-serif; }
            #map { height: 400px; width: 100%; border-radius: 12px; }
            #address { padding: 12px; background: rgba(19,27,46,0.9); border-radius: 8px; margin-top: 8px; color: #dae2fd; font-size: 14px; }
            #status { color: #adc6ff; padding: 8px 0; font-size: 14px; }
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
                    initMap(pos.coords.latitude, pos.coords.longitude);
                    reverseGeocode(pos.coords.latitude, pos.coords.longitude);
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

st.markdown('<div class="subtitle">Módulo de Vigilancia Activa</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Terminal de Análisis Acústico</div>', unsafe_allow_html=True)

col_main, col_side = st.columns([2.2, 1])

with col_main:

    st.markdown("""
    <div class="upload-card">
        <h3>Subir Archivo de Audio o Video</h3>
        <p>Arrastra o selecciona un archivo WAV, MP3, MP4, MOV o AVI para análisis acústico.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio o video",
        type=["wav", "mp3", "mp4", "mov", "avi"]
    )

if uploaded_file is not None:

    st.success("Archivo cargado correctamente.")

    file_details = {
        "Nombre": uploaded_file.name,
        "Tipo": uploaded_file.type,
        "Tamaño (KB)": round(uploaded_file.size / 1024, 2)
    }

    st.markdown("""
    <div class="card">
        <p class="metric-label">ARCHIVO CARGADO</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(file_details)

    extension = uploaded_file.name.split(".")[-1].lower()

    if extension in ["wav", "mp3"]:
        st.audio(uploaded_file)
    else:
        st.video(uploaded_file)

    if st.button("Analizar archivo"):
        try:
            resultado, confianza = predecir_audio(uploaded_file)

            if resultado == "DISPARO":
                st.error(f"DISPARO DETECTADO — Confianza: {confianza}%")
                mostrar_mapa_con_gps()
            else:
                st.success(f"NO SE DETECTÓ DISPARO — Confianza: {confianza}%")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Ocurrió un error al analizar el archivo: {e}")

else:

    st.warning("Aún no se ha cargado ningún archivo de audio o video.")

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