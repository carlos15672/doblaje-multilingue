#(ia-proyectos) dilanmc@dilanmc-msi-Cyborg:~/Documentos/ML/vocales$ streamlit run app.py
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

AUDIO_PATH = "pruebas"
VOCALS = ['A', 'E', 'I', 'O', 'U']
VOCAL_FILES = {
    'A': 'A.wav',
    'E': 'E.wav',
    'I': 'I.wav',
    'O': 'O.wav',
    'U': 'U.wav'
}

st.title("Comparador de espectros de vocales")

# Selección de vocales a comparar
vocales_seleccionadas = st.multiselect(
    "Selecciona las vocales a comparar:",
    VOCALS,
    default=['A', 'E'] 
)

# Parámetros de la FFT
max_freq = st.slider("Frecuencia máxima a mostrar (Hz)", min_value=1000, max_value=4000, value=3000, step=100)

if len(vocales_seleccionadas) > 0:
    plt.figure(figsize=(10, 5))
    for vocal in vocales_seleccionadas:
        audio_file = os.path.join(AUDIO_PATH, VOCAL_FILES[vocal])
        y, sr = librosa.load(audio_file, sr=None)
        N = len(y)
        yf = np.abs(np.fft.rfft(y))
        xf = np.fft.rfftfreq(N, 1/sr)
        idx = xf <= max_freq
        plt.plot(xf[idx], yf[idx], label=vocal)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.title("Espectro de frecuencias de las vocales seleccionadas")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write("Selecciona al menos una vocal para comparar.")

