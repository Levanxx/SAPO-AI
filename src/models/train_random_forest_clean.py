import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# RUTAS
# =========================

INPUT_PATH = Path("data/processed/features.csv")

MODEL_PATH = Path("models/random_forest_model_clean.pkl")
SCALER_PATH = Path("models/scaler_clean.pkl")

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def train_random_forest_clean():

    # =========================
    # CARGAR DATASET SIN ESCALAR
    # =========================

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset original cargado correctamente.")

    # =========================
    # FEATURES Y LABELS
    # =========================

    X = df.drop(columns=["archivo", "clase", "label"])
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

    print("\nDataset dividido correctamente.")

    # =========================
    # ESCALADO CORRECTO
    # FIT SOLO CON TRAIN
    # =========================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    print("\nScaler ajustado solo con datos de entrenamiento.")

    # =========================
    # MODELO RANDOM FOREST
    # =========================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # =========================
    # ENTRENAMIENTO
    # =========================

    model.fit(
        X_train_scaled,
        y_train
    )

    print("\nModelo Random Forest limpio entrenado correctamente.")

    # =========================
    # PREDICCIONES
    # =========================

    predictions = model.predict(
        X_test_scaled
    )

    # =========================
    # MÉTRICAS
    # =========================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n=== MÉTRICAS MODELO LIMPIO ===")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    # =========================
    # GUARDAR MODELO Y SCALER
    # =========================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("\nModelo limpio guardado en:")
    print(MODEL_PATH)

    print("\nScaler limpio guardado en:")
    print(SCALER_PATH)


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    train_random_forest_clean()