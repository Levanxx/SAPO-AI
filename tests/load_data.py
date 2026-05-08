import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import cargar_dataset, SAMPLE_RATE, DURACION_OBJETIVO


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")


def test_carga_dataset():
    dataset = cargar_dataset(DATA_PATH)

    assert len(dataset) > 0, "El dataset no debe estar vacío"

    for item in dataset:
        assert item["audio"] is not None, "El audio no debe ser None"
        assert item["sample_rate"] == SAMPLE_RATE, "Sample rate incorrecto"
        assert len(item["audio"]) == SAMPLE_RATE * DURACION_OBJETIVO, "Duración no uniforme"
        assert item["label"] in [0, 1], "Label inválido"
        assert item["clase"] in ["disparos", "no_disparos"], "Clase inválida"

    print("Prueba de carga uniforme completada correctamente")


if __name__ == "__main__":
    test_carga_dataset()
