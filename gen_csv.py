import os
import numpy as np
import librosa
import pandas as pd

DATASET_PATH = "audios_dataset"  
INTERVALOS = 25
MAX_FREQ = 3000
ETIQUETAS = ['a', 'e', 'i', 'o', 'u']

def extrae_fft(audio_path, n_intervalos=INTERVALOS, max_freq=MAX_FREQ):
    y, sr = librosa.load(audio_path, sr=None)
    N = len(y)
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1/sr)
    idx = np.where(xf <= max_freq)[0]
    yf = yf[idx]
    intervalos = np.array_split(yf, n_intervalos)
    features = [np.sqrt(np.mean(b**2)) for b in intervalos]
    return features

X, y = [], []
for label in ETIQUETAS:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            path = os.path.join(folder, file)
            try:
                features = extrae_fft(path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error en {path}: {e}")

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(INTERVALOS)])
df['label'] = y
df.to_csv('dataset_vocales_fft.csv', index=False)
print("¡Listo! Dataset guardado en dataset_vocales_fft.csv")
