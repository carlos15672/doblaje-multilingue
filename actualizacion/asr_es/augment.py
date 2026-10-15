"""Aumentos reproducibles de audio y log-Mel implementados con NumPy."""
from dataclasses import dataclass

import numpy as np

from .features import resample


@dataclass(frozen=True)
class AugmentConfig:
    probability: float = 0.8
    gain_db: float = 6.0
    noise_probability: float = 0.5
    noise_snr_min_db: float = 8.0
    noise_snr_max_db: float = 25.0
    speed_probability: float = 0.4
    speed_min: float = 0.9
    speed_max: float = 1.1
    time_masks: int = 2
    max_time_mask: int = 20
    frequency_masks: int = 2
    max_frequency_mask: int = 6

    def __post_init__(self):
        probabilities = (self.probability, self.noise_probability, self.speed_probability)
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Probabilidades de aumento no válidas")
        if self.gain_db < 0 or not 0 < self.speed_min <= self.speed_max:
            raise ValueError("Ganancia o velocidad no válidas")
        if self.noise_snr_min_db > self.noise_snr_max_db:
            raise ValueError("Intervalo SNR no válido")


def augment_audio(samples, sample_rate, rng, config=AugmentConfig()):
    x = np.asarray(samples, dtype=np.float64).copy()
    if rng.random() >= config.probability:
        return x
    if config.gain_db:
        gain = rng.uniform(-config.gain_db, config.gain_db)
        x *= 10 ** (gain / 20)
    if rng.random() < config.speed_probability:
        factor = rng.uniform(config.speed_min, config.speed_max)
        x = resample(x, sample_rate * factor, sample_rate)
    if rng.random() < config.noise_probability:
        signal_rms = float(np.sqrt(np.mean(x * x)))
        if signal_rms > 1e-9:
            snr = rng.uniform(config.noise_snr_min_db, config.noise_snr_max_db)
            noise = rng.normal(size=len(x))
            noise *= signal_rms / (10 ** (snr / 20) * max(float(np.sqrt(np.mean(noise * noise))), 1e-12))
            x += noise
    return np.clip(x, -1.0, 1.0)


def mask_features(features, rng, config=AugmentConfig()):
    result = np.asarray(features, dtype=np.float32).copy()
    if rng.random() >= config.probability:
        return result
    for _ in range(config.time_masks):
        width = int(rng.integers(0, min(config.max_time_mask, len(result)) + 1))
        if width:
            start = int(rng.integers(0, len(result) - width + 1))
            result[start:start + width] = 0
    for _ in range(config.frequency_masks):
        width = int(rng.integers(0, min(config.max_frequency_mask, result.shape[1]) + 1))
        if width:
            start = int(rng.integers(0, result.shape[1] - width + 1))
            result[:, start:start + width] = 0
    return result
