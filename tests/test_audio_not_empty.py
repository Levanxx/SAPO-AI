def test_audio_not_empty():
    dataset = cargar_dataset(./src/data)

    for item in dataset:
        assert len(item["audio"]) > 0
