import pandas as pd
from pathlib import Path

FEATURES_PATH = Path("data/processed/features.csv")

def validate_features_csv(path=FEATURES_PATH):
    df = pd.read_csv(path)

    print("\n=== INFO DATASET ===")
    print(df.info())

    print("\n=== PRIMERAS FILAS ===")
    print(df.head())

    print("\n=== NULOS POR COLUMNA ===")
    print(df.isnull().sum())

    print("\n=== DUPLICADOS ===")
    print(df.duplicated().sum())

    print("\n=== ESTADÍSTICAS ===")
    print(df.describe())

    print("\n=== COLUMNAS ===")
    print(df.columns.tolist())

    # Validaciones básicas
    errors = []

    if df.empty:
        errors.append("El archivo features.csv está vacío.")

    if df.isnull().sum().sum() > 0:
        errors.append("Existen valores nulos en el dataset.")

    if df.duplicated().sum() > 0:
        errors.append("Existen filas duplicadas.")

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        errors.append("No se encontraron columnas numéricas.")

    print("\n=== RESULTADO DE VALIDACIÓN ===")
    if errors:
        for error in errors:
            print(f"❌ {error}")
    else:
        print("Dataset validado correctamente. Ready para el siguiente sprint.")

    return df


if __name__ == "__main__":
    validate_features_csv()