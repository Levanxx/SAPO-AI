import os
import numpy as np
import pandas as pd

from src.data.load_data import cargar_dataset
from src.features.visualization import mostrar_waveforms_por_clase
from src.preprocessing.save_processed_audio import guardar_audio_procesado
from src.features.extract_features import extraer_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(BASE_DIR, "data", "raw")

dataset = cargar_dataset(carpeta)

if not dataset:
    print("No se cargaron audios")
    exit()

print(f"\nAudios cargados: {len(dataset)}")


# ================================
# INFORMACIÓN DEL DATASET
# ================================

for item in dataset[:10]:
    amplitud_maxima = np.max(np.abs(item["audio"]))

    print("-" * 40)
    print(f"Archivo: {item['archivo']}")
    print(f"Clase: {item['clase']}")
    print(f"Label: {item['label']}")
    print(f"Sample rate: {item['sample_rate']}")
    print(f"Muestras: {len(item['audio'])}")
    print(f"Duración: {item['duracion']:.2f} segundos")
    print(f"Amplitud máxima: {amplitud_maxima:.4f}")


# ================================
# EXTRACCIÓN DE FEATURES
# ================================

print("\nExtrayendo características acústicas...\n")

features_lista = []

for item in dataset:
    try:
        features = extraer_features(
            item["audio"],
            item["sample_rate"]
        )

        features["archivo"] = item["archivo"]
        features["clase"] = item["clase"]
        features["label"] = item["label"]

        features_lista.append(features)

    except Exception as e:
        print(f"Error extrayendo features de {item['archivo']}: {e}")

df_features = pd.DataFrame(features_lista)

ruta_salida = os.path.join(BASE_DIR, "data", "processed", "features.csv")
df_features.to_csv(ruta_salida, index=False)

print("Features extraídas correctamente.")
print(f"Total de registros procesados: {len(df_features)}")
print(f"Archivo guardado en: {ruta_salida}")

print("\nVista previa del dataset de features:")
print(df_features.head())


# ================================
# GUARDAR AUDIOS PROCESADOS
# ================================
'''
print("\nGuardando audios procesados...\n")

for item in dataset:

    guardar_audio_procesado(
        item["audio"],
        item["sample_rate"],
        item["clase"],
        item["archivo"]
    )

print("\nTodos los audios fueron guardados correctamente.")
'''


'''
# ================================
# ANÁLISIS DE DURACIÓN
# ================================

duraciones = [d["duracion"] for d in dataset]

print("\n--- Estadísticas de duración ---")
print(f"Duración mínima: {min(duraciones):.2f} s")
print(f"Duración máxima: {max(duraciones):.2f} s")
print(f"Duración promedio: {sum(duraciones)/len(duraciones):.2f} s")


print("\n--- Audios muy cortos (<1s) ---")
for d in dataset:
    if d["duracion"] < 1:
        print(d["archivo"], f"{d['duracion']:.2f}s")


print("\n--- Audios muy largos (>10s) ---")
for d in dataset:
    if d["duracion"] > 10:
        print(d["archivo"], f"{d['duracion']:.2f}s")
'''


# ================================
# WAVEFORMS POR CLASE
# ================================

# mostrar_waveforms_por_clase(dataset, cantidad=5)