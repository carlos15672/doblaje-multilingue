"""Entrada/salida de audio. No contiene reconocimiento ni modelos."""
from pathlib import Path
import wave

import numpy as np


def read_audio(path):
    """Lee WAV PCM con la biblioteca estándar; otros formatos usan soundfile."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as source:
            rate, channels = source.getframerate(), source.getnchannels()
            width = source.getsampwidth()
            raw = source.readframes(source.getnframes())
        if width == 1:
            samples = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) / 128
        elif width in (2, 4):
            samples = np.frombuffer(raw, f"<i{width}").astype(np.float64)
            samples /= 2 ** (width * 8 - 1)
        elif width == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            values = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            samples = ((values ^ 0x800000) - 0x800000) / 8388608.0
        else:
            raise ValueError(f"WAV PCM de {width} bytes no soportado")
        samples = samples.reshape(-1, channels).mean(axis=1)
    except (wave.Error, EOFError):
        try:
            import soundfile as sf
        except ImportError as error:
            raise ValueError("Este formato necesita soundfile; instala requirements_asr.txt") from error
        samples, rate = sf.read(str(path), dtype="float64", always_2d=True)
        samples = samples.mean(axis=1)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Audio vacío o no finito: {path}")
    return samples, int(rate)


def write_wav(path, samples, rate=16000):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"El archivo ya existe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def record(path, seconds=5.0):
    if not 0.5 <= seconds <= 30:
        raise ValueError("La grabación debe durar entre 0.5 y 30 segundos")
    if Path(path).exists():
        raise FileExistsError(f"El archivo ya existe: {path}")
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ValueError("Instala sounddevice con requirements_asr.txt para usar el micrófono") from error
    # Grabar a la frecuencia nativa evita exigir soporte de 16 kHz al dispositivo.
    rate = int(sd.query_devices(kind="input")["default_samplerate"])
    print(f"Grabando {seconds:g} segundos. Habla ahora...", flush=True)
    samples = sd.rec(round(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    write_wav(path, samples[:, 0], rate)
    return {"archivo": str(Path(path).resolve()), "segundos": seconds, "frecuencia": rate}
