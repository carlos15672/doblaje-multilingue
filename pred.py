import joblib
import numpy as np
import librosa

N_BANDAS = 25
MAX_FREQ = 3000

def extrae_fft(audio_path, n_bandas=N_BANDAS, max_freq=MAX_FREQ):
    y, sr = librosa.load(audio_path, sr=None)
    N = len(y)
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1/sr)
    idx = np.where(xf <= max_freq)[0]
    yf = yf[idx]
    bandas = np.array_split(yf, n_bandas)
    features = [np.sqrt(np.mean(b**2)) for b in bandas]
    return features

scaler = joblib.load("scaler_vocales_fft.pkl")
clf = joblib.load("modelo_vocales_fft.pkl")


audio_path = "pruebas/a5.wav"
#audio_path = "audios_dataset/a/1.wav"
features = extrae_fft(audio_path)
features_scaled = scaler.transform([features])
pred = clf.predict(features_scaled)[0]
print(f"Predicción para '{audio_path}': {pred}")
