import os
import librosa

def cargar_audio(ruta):
    try:
        audio, sr = librosa.load(ruta, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Error cargando {ruta}: {e}")
        return None, None


def cargar_dataset(carpeta):
    datos = []

    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)

        if archivo.endswith((".wav", ".mp3")):
            audio, sr = cargar_audio(ruta)

            if audio is not None:
                datos.append({
                    "archivo": archivo,
                    "audio": audio,
                    "sample_rate": sr
                })

    return datos