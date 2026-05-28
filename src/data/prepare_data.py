import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# RUTAS
# =========================

INPUT_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("data/processed/features_scaled.csv")
SCALER_PATH = Path("models/scaler.pkl")

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def prepare_dataset():

    # =========================
    # CARGAR DATASET
    # =========================

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset cargado correctamente.")

    # =========================
    # SELECCIÓN DE FEATURES
    # =========================

    # Eliminar columnas no necesarias
    X = df.drop(columns=["archivo", "clase", "label"])

    # Labels
    y = df["label"]

    print(f"\nFeatures seleccionadas: {X.columns.tolist()}")

    # =========================
    # ESCALADO
    # =========================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # =========================
    # CONVERTIR A DATAFRAME
    # =========================

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Agregar labels nuevamente
    X_scaled_df["label"] = y

    # =========================
    # CREAR CARPETAS SI NO EXISTEN
    # =========================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SCALER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # GUARDAR DATASET PROCESADO
    # =========================

    X_scaled_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # =========================
    # GUARDAR SCALER
    # =========================

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # =========================
    # MENSAJES FINALES
    # =========================

    print("\nDataset preparado correctamente.")

    print(f"\nArchivo guardado en:")
    print(OUTPUT_PATH)

    print(f"\nScaler guardado en:")
    print(SCALER_PATH)


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    prepare_dataset()