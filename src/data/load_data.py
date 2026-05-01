import os
import soundfile as sf


def cargar_audio(ruta):
    try:
        audio, sr = sf.read(ruta)

        #Convertir audio estereo a mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return audio, sr

    except Exception as e:
        print(f"Error cargando {ruta}: {e}")
        return None, None


def cargar_dataset(base_path):
    datos = []

    clases = {
        "disparos": 1,
        "no_disparos": 0
    }

    for clase, label in clases.items():
        carpeta = os.path.join(base_path, clase)

        if not os.path.exists(carpeta):
            print(f"No existe la carpeta: {carpeta}")
            continue

        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith((".wav", ".mp3")):
                ruta = os.path.join(carpeta, archivo)

                audio, sr = cargar_audio(ruta)

                if audio is not None:
                    datos.append({
                        "archivo": archivo,
                        "ruta": ruta,
                        "audio": audio,
                        "sample_rate": sr,
                        "label": label,
                        "clase": clase,
                        "duracion": len(audio) / sr
                    })

    return datos