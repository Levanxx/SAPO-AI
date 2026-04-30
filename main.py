from src.data.load_data import cargar_dataset

carpeta = "data/raw"

dataset = cargar_dataset(carpeta)

print(f"Audios cargados: {len(dataset)}")

for d in dataset:
    print(d["archivo"], d["sample_rate"], len(d["audio"]))

    print(f"Duración aprox: {len(d['audio']) / d['sample_rate']:.2f} segundos")