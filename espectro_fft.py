import librosa
import numpy as np
import matplotlib.pyplot as plt

def grafica_fft(audio_path, max_freq=3000, n_bandas=25):
    y, sr = librosa.load(audio_path, sr=None)
    N = len(y)
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1/sr)
    
    # Solo hasta max_freq Hz
    idx = xf <= max_freq
    xf = xf[idx]
    yf = yf[idx]
    
    band_edges = np.linspace(0, max_freq, n_bandas + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(xf, yf)
    plt.xlim([0, max_freq])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.title(f"Espectro FFT de {audio_path.split('/')[-1]}")
    for edge in band_edges:
        plt.axvline(edge, color='red', linestyle='--', alpha=0.7)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

grafica_fft('pruebas/a2g.wav')
