"""Detección de voz y segmentación temporal sin servicios ni modelos externos."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 30
    hop_ms: int = 10
    relative_threshold_db: float = 10.0
    absolute_threshold_db: float = -55.0
    min_speech_ms: int = 180
    min_silence_ms: int = 250
    padding_ms: int = 150
    max_segment_seconds: float = 25.0
    overlap_seconds: float = 0.4

    def __post_init__(self):
        if self.frame_ms < 5 or self.hop_ms < 1 or self.hop_ms > self.frame_ms:
            raise ValueError("Configuración temporal de VAD no válida")
        if self.min_speech_ms < 0 or self.min_silence_ms < 0 or self.padding_ms < 0:
            raise ValueError("Duraciones de VAD no válidas")
        if self.max_segment_seconds <= 0 or not 0 <= self.overlap_seconds < self.max_segment_seconds:
            raise ValueError("Segmentación máxima o solapamiento no válidos")


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int

    def seconds(self, sample_rate):
        return self.start_sample / sample_rate, self.end_sample / sample_rate


def _fill_short_gaps(mask, maximum):
    result = mask.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        if index > 0 and end < len(result) and end - index <= maximum:
            result[index:end] = True
        index = end
    return result


def _remove_short_runs(mask, minimum):
    result = mask.copy()
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and result[end]:
            end += 1
        if end - index < minimum:
            result[index:end] = False
        index = end
    return result


def detect_speech(samples, sample_rate, config=VADConfig()):
    """Devuelve segmentos de voz y diagnóstico energético.

    El umbral combina un piso absoluto con el percentil bajo de energía de la
    propia grabación. Así, el ruido estacionario no se interpreta como voz solo
    porque el archivo se haya grabado con mucha ganancia.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or sample_rate <= 0:
        raise ValueError("Se requiere audio mono, no vacío y finito")
    frame = max(1, round(sample_rate * config.frame_ms / 1000))
    hop = max(1, round(sample_rate * config.hop_ms / 1000))
    count = 1 + max(0, math.ceil((len(x) - frame) / hop))
    padded = np.pad(x - np.median(x), (0, (count - 1) * hop + frame - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_db = 20 * np.log10(np.maximum(rms, 1e-10))
    noise_floor_db = float(np.percentile(energy_db, 20))
    threshold_db = max(config.absolute_threshold_db, noise_floor_db + config.relative_threshold_db)
    active = energy_db >= threshold_db
    # Una grabación recortada puede contener voz continua y no tener cuadros de
    # silencio para estimar el ruido. Solo activar el fallback si la señal completa
    # es claramente fuerte; no se aplica a ruido ambiente de nivel bajo.
    if not active.any() and float(np.median(energy_db)) > -30:
        active = energy_db >= config.absolute_threshold_db
    active = _fill_short_gaps(active, math.ceil(config.min_silence_ms / config.hop_ms))
    active = _remove_short_runs(active, math.ceil(config.min_speech_ms / config.hop_ms))

    padding = round(sample_rate * config.padding_ms / 1000)
    regions = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and active[end]:
            end += 1
        start_sample = max(0, index * hop - padding)
        end_sample = min(len(x), (end - 1) * hop + frame + padding)
        if regions and start_sample <= regions[-1].end_sample:
            regions[-1] = SpeechSegment(regions[-1].start_sample, end_sample)
        else:
            regions.append(SpeechSegment(start_sample, end_sample))
        index = end

    maximum = round(config.max_segment_seconds * sample_rate)
    overlap = round(config.overlap_seconds * sample_rate)
    segments = []
    for region in regions:
        start = region.start_sample
        while region.end_sample - start > maximum:
            segments.append(SpeechSegment(start, start + maximum))
            start += maximum - overlap
        if region.end_sample > start:
            segments.append(SpeechSegment(start, region.end_sample))
    diagnostics = {
        "noise_floor_db": noise_floor_db,
        "threshold_db": threshold_db,
        "speech_ratio": float(active.mean()),
        "frames": len(active),
    }
    return segments, diagnostics
