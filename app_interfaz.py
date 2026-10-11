import streamlit as st
import librosa
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN "PRO" ---
st.set_page_config(
    page_title="Vocal Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS (DARK MODE NATIVO MEJORADO) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 0px;
    }
    
    .subtitle {
        text-align: center; color: #8b92a6; font-size: 1.2rem; margin-bottom: 30px;
    }

    div[data-testid="metric-container"] {
        background-color: #1A1C24; border: 1px solid #2C2F3A; padding: 20px;
        border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center; transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02); border-color: #00C9FF;
    }
    div[data-testid="metric-container"] > div {
        display: flex; justify-content: center; flex-direction: column;
    }

    /* ESTO ASEGURA QUE EL AUDIO OCUPE TODO EL ANCHO */
    .stAudio, div[data-testid="stAudioInput"] {
        width: 100% !important;
        margin-top: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24; border-radius: 5px;
        border: 1px solid #2C2F3A; color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00C9FF !important; color: black !important; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LÓGICA ---
N_BANDAS = 25
MAX_FREQ = 3000
SAMPLE_RATE = 22050
MODELO_PKL = "modelo_vocales_fft.pkl"
SCALER_PKL = "scaler_vocales_fft.pkl"

@st.cache_resource
def cargar_inteligencia():
    try: return joblib.load(MODELO_PKL), joblib.load(SCALER_PKL)
    except: return None, None

def procesar_audio(y, sr):
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trim) > 512: y = y_trim 
    
    N = len(y)
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1/sr)
    idx = np.where(xf <= MAX_FREQ)[0]
    yf_filt = yf[idx]
    
    bandas = np.array_split(yf_filt, N_BANDAS)
    feats = [np.sqrt(np.mean(b**2)) if len(b) > 0 else 0.0 for b in bandas]
    
    genero, pitch = "Indefinido", 0
    try:
        f0, _, _ = librosa.pyin(y[:int(len(y)*0.6)], fmin=50, fmax=300, sr=sr)
        avg_pitch = np.nanmean(f0)
        if not np.isnan(avg_pitch):
            pitch = avg_pitch
            genero = "Masculino" if pitch < 165 else "Femenino"
    except: pass
    
    return feats, genero, pitch, y

def graficar_pro(y, sr, pitch):
    tiempo = np.linspace(0, len(y)/sr, len(y))
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(len(y), 1/sr)
    mask = xf <= MAX_FREQ

    # Gráfica Onda
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=tiempo, y=y, mode='lines', name='Amplitud', 
                                 line=dict(color="#00C9FF", width=1.5)))
    fig_wave.update_layout(template="plotly_dark", title="Forma de Onda", xaxis_title="Tiempo (s)", 
                          yaxis_title="Amplitud", height=350, margin=dict(l=20, r=20, t=40, b=20),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Gráfica FFT
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf[mask], y=yf[mask], fill='tozeroy', mode='lines', name='Frecuencia',
                                line=dict(color="#92FE9D", width=1.5)))
    if pitch > 0:
        fig_fft.add_vline(x=pitch, line_width=2, line_dash="dash", line_color="red", 
                         annotation_text=f"Pitch: {pitch:.0f}Hz")
    fig_fft.update_layout(template="plotly_dark", title="Espectro de Frecuencias", xaxis_title="Hz", 
                          yaxis_title="dB", height=350, margin=dict(l=20, r=20, t=40, b=20),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return fig_wave, fig_fft

# --- 4. GESTIÓN DEL MODAL (REQUISITO 1) ---
if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False

@st.dialog("🚀 Guía de Operación")
def mostrar_instrucciones():
    st.markdown("""
    ### Bienvenido al Vocal Studio AI
    
    Sigue estos pasos para un análisis correcto:
    
    1.  **Selecciona la fuente** en el menú lateral (Micrófono o Archivo).
    2.  **Graba tu voz** pronunciando una vocal (**A, E, I, O, U**) de forma sostenida por 2 segundos.
    3.  **Observa el Dashboard**: Los KPIs y gráficas se actualizarán automáticamente.
    
    *Nota: Asegúrate de estar en un ambiente con poco ruido.*
    """)
    if st.button("Entendido, Iniciar Sistema", type="primary", use_container_width=True):
        st.session_state.intro_seen = True
        st.rerun()

if not st.session_state.intro_seen:
    mostrar_instrucciones()

# --- 5. INTERFAZ ---

st.markdown('<h1 class="title-gradient">VOCAL STUDIO AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análisis Espectral y Detección Biométrica de Voz</p>', unsafe_allow_html=True)

clf, scaler = cargar_inteligencia()
if not clf:
    st.error("Error: Modelos no encontrados.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9308/9308946.png", width=80)
    st.markdown("### 🎛️ Panel de Control")
    fuente = st.radio("Entrada de Audio", ["🎤 Micrófono", "📁 Archivo"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 Reiniciar Sistema", use_container_width=True):
        st.rerun()

# ENTRADA DE AUDIO (REQUISITO 2: SIN COLUMNAS PARA QUE OCUPE TODO EL ANCHO)
audio_data = None
sr_rate = SAMPLE_RATE

# --- AQUÍ ESTÁ EL CAMBIO CLAVE: Quitamos st.columns para que ocupe el 100% ---
if fuente == "🎤 Micrófono":
    st.caption("Grabar muestra de voz (Ancho Completo)")
    audio_buffer = st.audio_input("Grabar muestra", label_visibility="collapsed")
    if audio_buffer:
        audio_data, sr_rate = librosa.load(audio_buffer, sr=SAMPLE_RATE)
        st.toast("Audio capturado", icon="✅")
else:
    st.caption("Cargar archivo de audio")
    archivo = st.file_uploader("Subir WAV/MP3", type=["wav", "mp3"], label_visibility="collapsed")
    if archivo:
        audio_data, sr_rate = librosa.load(archivo, sr=SAMPLE_RATE)
        st.audio(archivo)

# DASHBOARD RESULTADOS
if audio_data is not None:
    st.markdown("---")
    with st.spinner("⚡ Procesando redes neuronales..."):
        features, genero, pitch, y_proc = procesar_audio(audio_data, sr_rate)
        f_scaled = scaler.transform([features])
        vocal_pred = clf.predict(f_scaled)[0].upper()
        probs = clf.predict_proba(f_scaled)[0]
        confianza = np.max(probs) * 100

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Vocal Detectada", f"{vocal_pred}", delta="IA Result")
    with k2: 
        color = "normal" if confianza > 70 else "off"
        st.metric("Confianza", f"{confianza:.1f}%", delta_color=color, delta="Certeza")
    with k3: 
        icon = "👨🏻" if genero == "Masculino" else "👩🏻"
        st.metric("Género Estimado", f"{genero}", delta=icon)
    with k4: st.metric("Frecuencia (Pitch)", f"{pitch:.0f} Hz", "Fundamental")

    # ADN Predicción
    st.markdown("##### 🧬 ADN de la Predicción")
    classes = clf.classes_
    cols_prob = st.columns(5)
    for i, (vocal, prob) in enumerate(zip(classes, probs)):
        with cols_prob[i]:
            pct = int(prob*100)
            bg = "#00C9FF" if pct > 50 else "#2C2F3A"
            txt = "#FFFFFF" if pct > 50 else "#666"
            st.markdown(f"""
            <div style="background-color: #1A1C24; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid {bg}40;">
                <div style="font-weight: bold; font-size: 1.2rem; color: {txt};">{vocal.upper()}</div>
                <div style="font-size: 0.8rem; color: #aaa;">{pct}%</div>
                <div style="width: 100%; height: 4px; background: #333; margin-top: 5px; border-radius: 2px;">
                    <div style="width: {pct}%; height: 100%; background: {bg}; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Gráficas
    st.markdown("<br>", unsafe_allow_html=True)
    fig_wave, fig_fft = graficar_pro(y_proc, sr_rate, pitch)
    t1, t2 = st.tabs(["🌊 Forma de Onda", "📊 Espectro FFT"])
    with t1: st.plotly_chart(fig_wave, use_container_width=True)
    with t2: st.plotly_chart(fig_fft, use_container_width=True)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 50px; border: 2px dashed #2C2F3A; border-radius: 20px; color: #555;">
        <h2>👋 Sistema Listo</h2>
        <p>Selecciona tu fuente de audio arriba para comenzar.</p>
    </div>
    """, unsafe_allow_html=True)