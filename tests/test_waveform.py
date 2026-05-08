import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import cargar_dataset
from src.features.visualization import mostrar_waveform


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")


def test_waveform():
    dataset = cargar_dataset(DATA_PATH)

    assert len(dataset) > 0, "El dataset no debe estar vacío"

    item = dataset[0]

    mostrar_waveform(
        item["audio"],
        item["sample_rate"],
        titulo=f"Prueba waveform - {item['clase']}"
    )

    print("Prueba de waveform ejecutada correctamente")


if __name__ == "__main__":
    test_waveform()
