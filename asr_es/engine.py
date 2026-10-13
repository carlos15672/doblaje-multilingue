"""Entrenamiento supervisado desde cero, evaluación y carga de pesos propios."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import read_audio
from .data import SpeechDataset, audit_partitions, collate, read_manifest
from .features import FeatureConfig, extract_features
from .metrics import error_rates
from .model import CRNN, ModelConfig
from .text import ALPHABET, greedy_decode


def choose_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible; utiliza --device cpu")
    return torch.device(value)


def decode_batch(log_probs, lengths):
    paths = log_probs.detach().argmax(-1).cpu()
    return [greedy_decode(paths[:int(length), i]) for i, length in enumerate(lengths)]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, count, references, hypotheses = 0.0, 0, [], []
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    for features, lengths, targets, target_lengths, texts in loader:
        log_probs, output_lengths = model(features.to(device), lengths)
        loss = criterion(log_probs, targets.to(device), output_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise ValueError("Pérdida CTC no finita en evaluación; revisar datos y longitudes")
        loss_sum += float(loss) * len(texts)
        count += len(texts)
        references.extend(texts)
        hypotheses.extend(decode_batch(log_probs, output_lengths))
    rates = error_rates(references, hypotheses)
    rates["loss"] = loss_sum / count
    return rates, [{"referencia": r, "hipotesis": h} for r, h in zip(references, hypotheses)]


def train(train_path, dev_path, output, epochs=20, batch_size=8, learning_rate=0.001,
          device_name="auto", seed=42, hidden_size=128):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or hidden_size < 4:
        raise ValueError("Parámetros de entrenamiento no válidos")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("La carpeta de entrenamiento debe estar vacía; utiliza otro --output")
    training, development = read_manifest(train_path), read_manifest(dev_path)
    audit = audit_partitions({"train": training, "dev": development})
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reduce variación de cuDNN; no promete identidad entre GPU/versiones distintas.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(device_name)
    config = FeatureConfig()
    model_config = ModelConfig(n_mels=config.n_mels, hidden_size=hidden_size)
    model = CRNN(model_config).to(device)
    output.mkdir(parents=True, exist_ok=True)
    train_data = SpeechDataset(training, config, output / "features")
    dev_data = SpeechDataset(development, config, output / "features")
    # Verificar y preparar características antes de empezar una época costosa.
    for name, dataset in (("train", train_data), ("dev", dev_data)):
        for i in range(len(dataset)):
            dataset[i]
            if (i + 1) % 100 == 0 or i + 1 == len(dataset):
                print(f"Preparando {name}: {i + 1}/{len(dataset)}", flush=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, generator=generator)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    run_info = {"feature_config": asdict(config), "model_config": asdict(model_config),
                "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "device": str(device), "torch_version": str(torch.__version__), "audit": audit,
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_manifest": str(Path(train_path).resolve()), "dev_manifest": str(Path(dev_path).resolve()),
                "pretrained_weights": False}
    (output / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    best = (float("inf"), float("inf"))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for features, lengths, targets, target_lengths, texts in train_loader:
            optimizer.zero_grad(set_to_none=True)
            log_probs, output_lengths = model(features.to(device), lengths)
            loss = criterion(log_probs, targets.to(device), output_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida CTC no finita; entrenamiento detenido")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(texts)
            count += len(texts)
        metrics, examples = evaluate(model, dev_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / count,
                  "dev_loss": metrics["loss"], "dev_wer": metrics["wer"], "dev_cer": metrics["cer"]}
        with (output / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = (metrics["wer"], metrics["loss"])
        if score < best:
            best = score
            checkpoint = {"format_version": 1, "alphabet": ALPHABET,
                          "feature_config": asdict(config), "model_config": asdict(model_config),
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "epoch": epoch, "metrics": record, "pretrained_weights": False}
            torch.save(checkpoint, output / "best.pt")
            (output / "dev_predictions.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"checkpoint": str((output / "best.pt").resolve()), "mejor_dev_wer": best[0],
            "nota": "Resultado de validación; falta evaluación final en test y micrófono"}


def load_model(checkpoint_path, device_name="auto"):
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("alphabet") != ALPHABET:
        raise ValueError("Checkpoint incompatible con esta versión del reconocedor")
    config = FeatureConfig(**checkpoint["feature_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    if config.n_mels != model_config.n_mels:
        raise ValueError("Configuración acústica incompatible con el modelo")
    model = CRNN(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, config, device


@torch.no_grad()
def transcribe(checkpoint_path, audio_path, device_name="auto"):
    model, config, device = load_model(checkpoint_path, device_name)
    audio, rate = read_audio(audio_path)
    if len(audio) / rate > 30:
        raise ValueError("Esta primera versión procesa grabaciones de hasta 30 segundos")
    rms = float(np.sqrt(np.mean((audio - audio.mean()) ** 2)))
    if rms < 1e-7:
        return {"texto": "", "estado": "sin señal", "nota": "No se ejecutó reconocimiento"}
    features = extract_features(audio, rate, config)
    log_probs, lengths = model(torch.from_numpy(features)[None].to(device), torch.tensor([len(features)]))
    return {"texto": decode_batch(log_probs, lengths)[0], "estado": "hipótesis del modelo",
            "nota": "No es una transcripción verificada; comprobar calidad en test independiente"}


def test_checkpoint(checkpoint_path, test_path, train_path, dev_path, device_name="auto"):
    partitions = {"train": read_manifest(train_path), "dev": read_manifest(dev_path), "test": read_manifest(test_path)}
    audit_partitions(partitions)
    model, config, device = load_model(checkpoint_path, device_name)
    dataset = SpeechDataset(partitions["test"], config)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    metrics, predictions = evaluate(model, loader, device)
    return {"metrics": metrics, "predictions": predictions}
