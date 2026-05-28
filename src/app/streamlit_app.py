import streamlit as st

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