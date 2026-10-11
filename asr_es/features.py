"""Remuestreo, FFT radix-2 y filtros log-Mel explícitos, usando solo NumPy.

No utiliza librosa, scipy.signal, np.fft ni transformaciones acústicas de torchaudio.
Las ventanas conservan el orden temporal; no se resume toda la frase en un vector.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    window_ms: int = 25
    hop_ms: int = 10
    n_fft: int = 512
    n_mels: int = 40

    def __post_init__(self):
        win = round(self.sample_rate * self.window_ms / 1000)
        hop = round(self.sample_rate * self.hop_ms / 1000)
        if self.sample_rate < 1000 or win < 2 or hop < 1 or hop > win:
            raise ValueError("Configuración de ventanas no válida")
        if self.n_fft < win or self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft debe ser potencia de dos y cubrir la ventana")
        if not 4 <= self.n_mels <= self.n_fft // 2:
            raise ValueError("Número de bandas Mel no válido")


def resample(samples, source_rate, target_rate):
    """Interpolación sinc con filtro paso bajo y ventana de Hann.

    Al reducir frecuencia, el corte se reduce ANTES del submuestreo para evitar
    aliasing. Se procesa por bloques para no construir una matriz de todo el audio.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Se requiere audio mono, no vacío y finito")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Frecuencia de muestreo no válida")
    if source_rate == target_rate:
        return x.copy()
    cutoff = 0.94 * min(1.0, target_rate / source_rate)
    radius = math.ceil(16 / cutoff)
    offsets = np.arange(-radius, radius + 1)
    result = np.empty(math.ceil(len(x) * target_rate / source_rate))
    for start in range(0, len(result), 2048):
        stop = min(start + 2048, len(result))
        positions = np.arange(start, stop) * source_rate / target_rate
        indices = np.floor(positions).astype(np.int64)[:, None] + offsets
        delta = positions[:, None] - indices
        window = np.where(np.abs(delta) <= radius, 0.5 + 0.5 * np.cos(np.pi * delta / radius), 0)
        weights = cutoff * np.sinc(cutoff * delta) * window
        valid = (indices >= 0) & (indices < len(x))
        weights *= valid
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        result[start:stop] = (x[np.clip(indices, 0, len(x) - 1)] * weights).sum(axis=1)
    return result


@lru_cache(maxsize=8)
def _reverse_bits(size):
    values = np.arange(size)
    reversed_values = np.zeros(size, dtype=np.int64)
    for _ in range(size.bit_length() - 1):
        reversed_values = (reversed_values << 1) | (values & 1)
        values >>= 1
    return reversed_values


def radix2_rfft(frames, n_fft):
    """FFT Cooley-Tukey iterativa; eje final = muestras, restantes = lotes."""
    frames = np.asarray(frames)
    if n_fft < 2 or n_fft & (n_fft - 1) or frames.shape[-1] > n_fft:
        raise ValueError("Tamaño FFT no válido")
    values = np.zeros((*frames.shape[:-1], n_fft), dtype=np.complex128)
    values[..., :frames.shape[-1]] = frames
    values = values[..., _reverse_bits(n_fft)]
    width = 2
    while width <= n_fft:
        blocks = values.reshape(*values.shape[:-1], -1, width)
        even = blocks[..., :width // 2].copy()
        rotation = np.exp(-2j * np.pi * np.arange(width // 2) / width)
        odd = blocks[..., width // 2:].copy() * rotation
        blocks[..., :width // 2] = even + odd
        blocks[..., width // 2:] = even - odd
        width *= 2
    return values[..., :n_fft // 2 + 1]


@lru_cache(maxsize=8)
def mel_filters(config):
    """Triángulos espaciados uniformemente en la escala Mel (fórmula HTK)."""
    maximum = 2595 * np.log10(1 + config.sample_rate / 2 / 700)
    points = 700 * (10 ** (np.linspace(0, maximum, config.n_mels + 2) / 2595) - 1)
    frequencies = np.arange(config.n_fft // 2 + 1) * config.sample_rate / config.n_fft
    filters = []
    for left, middle, right in zip(points[:-2], points[1:-1], points[2:]):
        rising = (frequencies - left) / (middle - left)
        falling = (right - frequencies) / (right - middle)
        filters.append(np.maximum(0, np.minimum(rising, falling)))
    bank = np.asarray(filters)
    if np.any(bank.sum(axis=1) == 0):
        raise ValueError("Hay bandas Mel sin bins; aumenta n_fft o reduce n_mels")
    bank.setflags(write=False)
    return bank


def extract_features(samples, rate, config=FeatureConfig()):
    x = resample(samples, rate, config.sample_rate)
    x -= x.mean()  # Quitar componente continua; no cortar silencios internos.
    win = round(config.sample_rate * config.window_ms / 1000)
    hop = round(config.sample_rate * config.hop_ms / 1000)
    count = 1 + max(0, math.ceil((len(x) - win) / hop))
    padded = np.pad(x, (0, (count - 1) * hop + win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))
    spectrum = radix2_rfft(frames * window, config.n_fft)
    power = np.abs(spectrum) ** 2 / config.n_fft
    features = np.log(np.maximum(power @ mel_filters(config).T, 1e-10))
    # Media/desviación globales de ESTA grabación; no se utilizan datos de prueba.
    features = (features - features.mean()) / max(float(features.std()), 1e-5)
    return features.astype(np.float32)
