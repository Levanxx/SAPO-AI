import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# =========================
# RUTAS
# =========================

INPUT_PATH = Path(
    "data/processed/features_scaled.csv"
)

MODEL_PATH = Path(
    "models/random_forest_model.pkl"
)

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def generate_confusion_matrix():

    # =========================
    # CARGAR DATASET
    # =========================

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        "\nDataset cargado correctamente."
    )

    # =========================
    # FEATURES Y LABELS
    # =========================

    X = df.drop(
        columns=["label"]
    )

    y = df["label"]

    # =========================
    # DIVISIÓN DEL DATASET
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        "\nDataset dividido correctamente."
    )

    # =========================
    # CARGAR MODELO
    # =========================

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "\nModelo cargado correctamente."
    )

    # =========================
    # PREDICCIONES
    # =========================

    predictions = model.predict(
        X_test
    )

    # =========================
    # MATRIZ DE CONFUSIÓN
    # =========================

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # =========================
    # RESULTADOS
    # =========================

    print(
        "\n=== MATRIZ DE CONFUSIÓN ===\n"
    )

    print(matrix)


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    generate_confusion_matrix()