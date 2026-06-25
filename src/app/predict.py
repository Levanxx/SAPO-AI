import pandas as pd
from pathlib import Path
import joblib

# =========================
# RUTAS
# =========================

MODEL_PATH = Path(
    "models/sapo.pkl"
)

SCALER_PATH = Path(
    "models/sapo_scaler.pkl"
)

# =========================
# FUNCIÓN DE PREDICCIÓN
# =========================

def predict_audio(features):

    # =========================
    # CARGAR MODELO Y SCALER
    # =========================

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "\nModelo y scaler cargados correctamente."
    )

    # =========================
    # NOMBRES DE FEATURES
    # =========================

    columns = [
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "mfcc_6",
        "mfcc_7",
        "mfcc_8",
        "mfcc_9",
        "mfcc_10",
        "mfcc_11",
        "mfcc_12",
        "mfcc_13",
        "zcr",
        "rms",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]

    # =========================
    # CREAR DATAFRAME
    # =========================

    df = pd.DataFrame(
        [features],
        columns=columns
    )

    # =========================
    # ESCALAR FEATURES
    # =========================

    scaled_features = scaler.transform(
        df
    )

    # =========================
    # CONVERTIR ESCALADO A DF
    # =========================

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=columns
    )

    # =========================
    # PREDICCIÓN
    # =========================

    prediction = model.predict(
        scaled_df
    )[0]

    # =========================
    # RESULTADO FINAL
    # =========================

    if prediction == 1:
        result = "DISPARO"
    else:
        result = "NO DISPARO"

    print(
        f"\nPredicción: {result}"
    )

    return result


# =========================
# EJEMPLO DE USO
# =========================

if __name__ == "__main__":

    sample_features = [
        -481.08,
        12.44,
        1.48,
        3.24,
        5.11,
        -2.20,
        0.55,
        -1.32,
        4.87,
        -3.11,
        1.23,
        -0.88,
        2.91,
        0.12,
        0.25,
        1294.14,
        2200.45,
        3500.90
    ]

    predict_audio(
        sample_features
    )