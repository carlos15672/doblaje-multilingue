"""CRNN propia, construida con capas genéricas de PyTorch; sin pesos externos."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .text import ALPHABET


@dataclass
class ModelConfig:
    n_mels: int = 40
    conv_channels: int = 16
    hidden_size: int = 128
    recurrent_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_mels < 4 or self.conv_channels < 2:
            raise ValueError("Configuración convolucional no válida")
        if self.hidden_size < 4 or self.recurrent_layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Configuración recurrente no válida")


class CRNN(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        c = config.conv_channels
        # Ejes Conv2D: tiempo, frecuencia. Solo la primera reduce el tiempo.
        self.conv1 = nn.Conv2d(1, c, 3, stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(c, 2 * c, 3, stride=(1, 2), padding=1)
        frequencies = ((config.n_mels + 1) // 2 + 1) // 2
        self.projection = nn.Sequential(
            nn.Linear(2 * c * frequencies, config.hidden_size),
            nn.LayerNorm(config.hidden_size), nn.ReLU(), nn.Dropout(config.dropout),
        )
        self.recurrent = nn.GRU(
            config.hidden_size, config.hidden_size, num_layers=config.recurrent_layers,
            batch_first=True, bidirectional=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0,
        )
        self.output = nn.Linear(2 * config.hidden_size, len(ALPHABET) + 1)

    @staticmethod
    def output_lengths(lengths):
        return (lengths + 1) // 2

    def forward(self, features, lengths):
        if features.ndim != 3 or features.shape[2] != self.config.n_mels:
            raise ValueError("Se esperan características [lote, tiempo, bandas Mel]")
        lengths = lengths.cpu().long()
        if len(lengths) != features.shape[0] or torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("Longitudes de secuencia no válidas")
        mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths.to(features.device)[:, None]
        x = features * mask[:, :, None]
        x = torch.relu(self.conv1(x[:, None, :, :]))
        output_lengths = self.output_lengths(lengths)
        # Impedir que el relleno del lote afecte los bordes de grabaciones cortas.
        mask = torch.arange(x.shape[2], device=x.device)[None, :] < output_lengths.to(x.device)[:, None]
        x = x * mask[:, None, :, None]
        x = torch.relu(self.conv2(x))
        x = self.projection(x.permute(0, 2, 1, 3).flatten(2))
        packed = pack_padded_sequence(x, output_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.recurrent(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        log_probs = self.output(x).log_softmax(-1).transpose(0, 1).contiguous()
        return log_probs, output_lengths
