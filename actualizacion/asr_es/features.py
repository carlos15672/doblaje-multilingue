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


MAX_PHASES = 4096  # Fases distintas del filtro que vale la pena precalcular.
# Bloques pequeños para que los datos temporales quepan en la caché del procesador:
# con varios procesos en paralelo, los bloques grandes saturan la memoria compartida.
# Medido con 8 procesos (Ryzen 9 5900HS): 8192 salidas/todas las ventanas -> 22 clips/s;
# 256/32 -> 132 clips/s, con resultados idénticos.
RESAMPLE_BLOCK = 256
FRAME_BLOCK = 32


def _filter_shape(source_rate, target_rate):
    cutoff = 0.94 * min(1.0, target_rate / source_rate)
    return cutoff, math.ceil(16 / cutoff)


def _sinc_weights(delta, cutoff, radius):
    window = np.where(np.abs(delta) <= radius, 0.5 + 0.5 * np.cos(np.pi * delta / radius), 0)
    return cutoff * np.sinc(cutoff * delta) * window


def _resample_direct(x, source_rate, target_rate, outputs):
    """Algoritmo de referencia: calcula los pesos de cada salida pedida.

    Se usa en los bordes (donde faltan muestras y los pesos se renormalizan),
    con frecuencias que no forman una razón sencilla y en las pruebas.
    """
    cutoff, radius = _filter_shape(source_rate, target_rate)
    offsets = np.arange(-radius, radius + 1)
    result = np.empty(len(outputs))
    for start in range(0, len(outputs), 2048):
        positions = outputs[start:start + 2048] * source_rate / target_rate
        indices = np.floor(positions).astype(np.int64)[:, None] + offsets
        delta = positions[:, None] - indices
        weights = _sinc_weights(delta, cutoff, radius)
        valid = (indices >= 0) & (indices < len(x))
        weights *= valid
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        result[start:start + 2048] = (x[np.clip(indices, 0, len(x) - 1)] * weights).sum(axis=1)
    return result


@lru_cache(maxsize=16)
def _phase_weights(source_rate, target_rate):
    """Pesos normalizados por fase. Con razón up/down solo existen `up` desfases distintos
    (48 kHz -> 16 kHz: uno solo), así que se calculan una vez y no en cada bloque."""
    common = math.gcd(source_rate, target_rate)
    up, down = target_rate // common, source_rate // common
    cutoff, radius = _filter_shape(source_rate, target_rate)
    positions = np.arange(up) * source_rate / target_rate
    delta = (positions - np.floor(positions))[:, None] - np.arange(-radius, radius + 1)
    weights = _sinc_weights(delta, cutoff, radius)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    weights.setflags(write=False)
    return up, down, radius, weights


def resample(samples, source_rate, target_rate):
    """Interpolación sinc con filtro paso bajo y ventana de Hann.

    Al reducir frecuencia, el corte se reduce ANTES del submuestreo para evitar
    aliasing. En el interior del audio se usan pesos precalculados por fase; en los
    bordes, el cálculo directo. Ambos dan el mismo resultado (ver pruebas).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Se requiere audio mono, no vacío y finito")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Frecuencia de muestreo no válida")
    if source_rate == target_rate:
        return x.copy()
    outputs = np.arange(math.ceil(len(x) * target_rate / source_rate))
    if source_rate != int(source_rate) or target_rate != int(target_rate):
        return _resample_direct(x, source_rate, target_rate, outputs)
    up, down, radius, weights = _phase_weights(int(source_rate), int(target_rate))
    if up > MAX_PHASES or len(x) < 2 * radius + 1:
        return _resample_direct(x, source_rate, target_rate, outputs)
    result = np.empty(len(outputs))
    floors = outputs * down // up
    interior = (floors >= radius) & (floors + radius < len(x))
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * radius + 1)
    selected = np.flatnonzero(interior)
    for start in range(0, len(selected), RESAMPLE_BLOCK):
        chunk = selected[start:start + RESAMPLE_BLOCK]
        block = windows[floors[chunk] - radius]
        result[chunk] = block @ weights[0] if up == 1 else np.einsum("ij,ij->i", block, weights[chunk % up])
    edges = np.flatnonzero(~interior)
    if len(edges):
        result[edges] = _resample_direct(x, source_rate, target_rate, edges)
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
    bank = mel_filters(config).T
    energies = np.empty((len(frames), config.n_mels))
    # Cada ventana es independiente: procesarlas por bloques da el mismo resultado.
    for start in range(0, len(frames), FRAME_BLOCK):
        spectrum = radix2_rfft(frames[start:start + FRAME_BLOCK] * window, config.n_fft)
        power = np.abs(spectrum) ** 2 / config.n_fft
        energies[start:start + FRAME_BLOCK] = power @ bank
    features = np.log(np.maximum(energies, 1e-10))
    # Media/desviación globales de ESTA grabación; no se utilizan datos de prueba.
    features = (features - features.mean()) / max(float(features.std()), 1e-5)
    return features.astype(np.float32)
