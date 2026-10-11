import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- CONFIGURACIÓN ---
N_BANDAS = 25
MAX_FREQ = 3000
SAMPLE_RATE_FIJO = 22050  # Frecuencia estándar para evitar errores
MODELO_VOCALES = "modelo_vocales_fft.pkl"
SCALER_VOCALES = "scaler_vocales_fft.pkl"

# --- FUNCIONES ---
def extrae_fft_features(y, sr, n_bandas=N_BANDAS, max_freq=MAX_FREQ):
    # 1. Normalizar el volumen (CRÍTICO para micrófono)
    if len(y) > 0:
        y = librosa.util.normalize(y)
    
    # 2. Calcular FFT
    N = len(y)
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1/sr)
    
    # 3. Filtrar frecuencias
    idx = np.where(xf <= max_freq)[0]
    yf_filt = yf[idx]
    
    # 4. Dividir en bandas y calcular RMS
    if len(yf_filt) == 0:
        return [0] * n_bandas

    bandas = np.array_split(yf_filt, n_bandas)
    features = []
    for b in bandas:
        if len(b) > 0:
            features.append(np.sqrt(np.mean(b**2)))
        else:
            features.append(0.0)
    return features

def cargar_modelo():
    try:
        clf = joblib.load(MODELO_VOCALES)
        scaler = joblib.load(SCALER_VOCALES)
        return clf, scaler
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

# --- INTERFAZ PRINCIPAL ---
st.set_page_config(page_title="Detector de Vocales", layout="centered")
st.title("🎙️ Detector de Vocales")
st.write("Di una vocal fuerte y clara (A, E, I, O, U).")

clf, scaler = cargar_modelo()

if clf:
    # Selector de entrada
    tipo_entrada = st.radio("Fuente:", ["Grabar Micrófono", "Subir Archivo"], horizontal=True)
    
    y = None
    sr = SAMPLE_RATE_FIJO 

    if tipo_entrada == "Grabar Micrófono":
        audio_buffer = st.audio_input("🔴 Presiona para Grabar")
        
        if audio_buffer:
            # 1. Reproducir el audio grabado (Para que lo escuches)
            st.audio(audio_buffer, format="audio/wav")
            
            # 2. Rebobinar el buffer (IMPORTANTE: Resetear el puntero al inicio)
            audio_buffer.seek(0) 
            
            # 3. Cargar en Librosa
            y, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE_FIJO)
            
    else:
        archivo = st.file_uploader("Sube un WAV/OGG", type=["wav", "ogg"])
        if archivo:
            st.audio(archivo, format="audio/wav")
            archivo.seek(0) # Rebobinar por seguridad
            y, sr = librosa.load(archivo, sr=SAMPLE_RATE_FIJO)

    # --- PROCESAMIENTO E INFERENCIA ---
    if y is not None:
        st.divider()
        with st.spinner("Analizando..."):
            
            # Recortar silencios extremos
            y_trim, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trim) > 1024: y = y_trim

            # Extraer features
            features = extrae_fft_features(y, sr)
            
            # Escalar y Predecir
            features_scaled = scaler.transform([features])
            pred_vocal = clf.predict(features_scaled)[0]
            
            # Probabilidad (si el modelo lo permite)
            try:
                probs = clf.predict_proba(features_scaled)[0]
                confianza = np.max(probs)
            except:
                confianza = 0.0

            # --- MOSTRAR RESULTADOS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🗣️ Vocal")
                color = "green" if confianza > 0.6 else "orange"
                st.markdown(f":{color}[# {pred_vocal.upper()}]")
                st.progress(float(confianza))
                st.caption(f"Certeza: {confianza:.1%}")

            with col2:
                st.markdown("### 👤 Género")
                st.info("# Hombre") 
                st.caption("*(Modelo de género pendiente)*")

            # --- GRÁFICA ---
            st.write("---")
            st.subheader("📊 Espectro de Frecuencia")
            
            # FFT para graficar
            N = len(y)
            yf_plot = np.abs(np.fft.rfft(y))
            xf_plot = np.fft.rfftfreq(N, 1/sr)
            mask = xf_plot <= MAX_FREQ
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(xf_plot[mask], yf_plot[mask], color='#4CAF50', alpha=0.8)
            ax.fill_between(xf_plot[mask], yf_plot[mask], color='#4CAF50', alpha=0.3)
            ax.set_title("Frecuencias (0-3kHz)")
            ax.set_xlabel("Hz")
            ax.set_yticks([]) 
            ax.grid(True, alpha=0.2)
            
            # Líneas de las bandas
            band_edges = np.linspace(0, MAX_FREQ, N_BANDAS + 1)
            for edge in band_edges:
                ax.axvline(edge, color='red', linestyle=':', alpha=0.3)
                
            st.pyplot(fig)