import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

def evaluate_model():

    # =========================
    # CARGAR DATASET
    # =========================

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset cargado correctamente.")

    # =========================
    # FEATURES Y LABELS
    # =========================

    X = df.drop(columns=["label"])
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

    # =========================
    # CARGAR MODELO
    # =========================

    model = joblib.load(MODEL_PATH)

    print("\nModelo cargado correctamente.")

    # =========================
    # PREDICCIONES
    # =========================

    predictions = model.predict(X_test)

    # =========================
    # MÉTRICAS
    # =========================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    # =========================
    # RESULTADOS
    # =========================

    print("\n=== MÉTRICAS DEL MODELO ===")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    evaluate_model()