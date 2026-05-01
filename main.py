import os
from src.data.load_data import cargar_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(BASE_DIR, "data", "raw")

dataset = cargar_dataset(carpeta)

if not dataset:
    print("No se cargaron audios")
    exit()

print(f"Audios cargados: {len(dataset)}")

for d in dataset[:5]:
    print("-" * 40)
    print(f"Archivo: {d['archivo']}")
    print(f"Etiqueta: {d['label']}")
    print(f"Sample rate: {d['sample_rate']}")
    print(f"Muestras: {len(d['audio'])}")
    print(f"Duración aprox: {d['duracion']:.2f} segundos")