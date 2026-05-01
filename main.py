import os
from src.data.load_data import cargar_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(BASE_DIR, "data", "raw")

dataset = cargar_dataset(carpeta)

if not dataset:
    print("No se cargaron audios")
    exit()

print(f"\nAudios cargados: {len(dataset)}")

for d in dataset[:5]:
    print("-" * 40)
    print(f"Archivo: {d['archivo']}")
    print(f"Etiqueta: {d['label']}")
    print(f"Sample rate: {d['sample_rate']}")
    print(f"Muestras: {len(d['audio'])}")
    print(f"Duración: {d['duracion']:.2f} s")


# ================================
#ANÁLISIS DE DURACIÓN
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