import numpy as np
import librosa


def extraer_mfcc(audio, sr, n_mfcc=13):
    """
    Extrae coeficientes MFCC del audio, lo que representa las características importantes del timbre del sonido.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def calcular_zcr(audio):
    """
    Calcula Zero Crossing Rate (Mide cuántas veces la señal cruza por cero)
    """
    zcr = librosa.feature.zero_crossing_rate(audio)
    return np.mean(zcr)


def calcular_rms(audio):
    """
    Calcula RMS Energy (Mide la energía promedio del audio)
    """
    rms = librosa.feature.rms(y=audio)
    return np.mean(rms)


def calcular_spectral_features(audio, sr):
    """
    Calcula características espectrales del audio.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    return {
        "spectral_centroid": np.mean(spectral_centroid),
        "spectral_bandwidth": np.mean(spectral_bandwidth),
        "spectral_rolloff": np.mean(spectral_rolloff)
    }


def extraer_features(audio, sr):
    """
    Extrae todas las características principales del audio y las devuelve en un diccionario
    """
    features = {}

    mfcc = extraer_mfcc(audio, sr)

    for i, valor in enumerate(mfcc, start=1):
        features[f"mfcc_{i}"] = valor

    features["zcr"] = calcular_zcr(audio)
    features["rms"] = calcular_rms(audio)

    spectral_features = calcular_spectral_features(audio, sr)
    features.update(spectral_features)

    return features