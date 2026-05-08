import numpy as np

def test_audio_types():
    dataset = cargar_dataset(./src/data)

    item = dataset[0]

    assert isinstance(item["audio"], np.ndarray)
    assert isinstance(item["sample_rate"], int)
    assert isinstance(item["clase"], str)
