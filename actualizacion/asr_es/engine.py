"""Entrenamiento supervisado desde cero, evaluación y carga de pesos propios."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

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


def _save_atomic(checkpoint, path):
    # Escribir aparte y renombrar: un corte a mitad de la escritura no deja el archivo dañado.
    partial = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, partial)
    os.replace(partial, path)


def train(train_path, dev_path, output, epochs=20, batch_size=8, learning_rate=0.001,
          device_name="auto", seed=42, hidden_size=128, cache_dir=None, workers=0, resume=False):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or hidden_size < 4:
        raise ValueError("Parámetros de entrenamiento no válidos")
    output = Path(output)
    previous = source = None
    if resume:
        # last.pt guarda modelo, optimizador y orden de los datos. best.pt de corridas anteriores
        # a esta opción solo guarda los pesos: en ese caso Adam vuelve a empezar.
        source = next((path for path in (output / "last.pt", output / "best.pt") if path.exists()), None)
        if source is None or not (output / "config.json").exists():
            raise ValueError("No hay una corrida que reanudar en esa carpeta (faltan config.json y last.pt/best.pt)")
        run_info = json.loads((output / "config.json").read_text(encoding="utf-8"))
        requested = {"train_manifest": str(Path(train_path).resolve()), "dev_manifest": str(Path(dev_path).resolve()),
                     "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed}
        changed = [key for key, value in requested.items() if run_info.get(key) != value]
        if run_info["model_config"]["hidden_size"] != hidden_size:
            changed.append("hidden_size")
        if changed:
            raise ValueError(f"Para reanudar usa los mismos parámetros que la corrida original; cambian: {changed}")
        previous = torch.load(source, map_location="cpu", weights_only=True)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("La carpeta de entrenamiento debe estar vacía; utiliza otro --output o --resume")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start_epoch, best = 0, (float("inf"), float("inf"))
    if previous is not None:
        if (previous.get("alphabet") != ALPHABET or previous.get("model_config") != asdict(model_config)
                or previous.get("feature_config") != asdict(config)):
            raise ValueError("El checkpoint no corresponde a esta configuración del modelo")
        model.load_state_dict(previous["state_dict"])
        start_epoch = previous["epoch"]
        if "optimizer" in previous:
            optimizer.load_state_dict(previous["optimizer"])
            generator.set_state(previous["generator"])
            best = tuple(previous["best"])
        else:
            generator.manual_seed(seed + start_epoch)  # no repetir el orden de la primera época
            best = (previous["metrics"]["dev_wer"], previous["metrics"]["dev_loss"])
        print(f"Reanudando después de la época {start_epoch} desde {source.name}"
              f"{'' if 'optimizer' in previous else ' (solo pesos; optimizador nuevo)'}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    # Caché compartida (ver `cache-features`) o, si no se indica, una propia de esta corrida.
    cache = Path(cache_dir).resolve() if cache_dir else output / "features"
    train_data = SpeechDataset(training, config, cache)
    dev_data = SpeechDataset(development, config, cache)
    # Verificar y preparar características antes de empezar una época costosa.
    # Con `workers` > 0 los archivos se leen en procesos paralelos: en Windows abrir un archivo
    # de la caché que no se ha abierto recientemente cuesta ~9 ms, y uno por uno serían ~50 min por época.
    if previous is None:
        for name, dataset in (("train", train_data), ("dev", dev_data)):
            checked, step = 0, 100 if workers == 0 else 10000
            for *_, texts in DataLoader(dataset, batch_size=64, collate_fn=collate, num_workers=workers):
                before, checked = checked, checked + len(texts)
                if checked // step > before // step or checked == len(dataset):
                    print(f"Preparando {name}: {checked}/{len(dataset)}", flush=True)
    else:
        print("Se omite la verificación previa: ya se hizo al iniciar esta corrida con los mismos datos", flush=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=workers, persistent_workers=workers > 0, generator=generator)
    dev_workers = min(workers, 4)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate,
                            num_workers=dev_workers, persistent_workers=dev_workers > 0)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    if previous is None:
        run_info = {"feature_config": asdict(config), "model_config": asdict(model_config),
                    "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "device": str(device), "torch_version": str(torch.__version__), "audit": audit,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_manifest": str(Path(train_path).resolve()), "dev_manifest": str(Path(dev_path).resolve()),
                    "feature_cache": str(cache), "data_workers": workers, "pretrained_weights": False}
    else:
        run_info["epochs"] = epochs
        run_info.setdefault("reanudaciones", []).append({
            "desde": source.name, "despues_de_epoca": start_epoch, "data_workers": workers,
            "optimizador": "restaurado" if "optimizer" in previous else "nuevo"})
    (output / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        epoch_start = time.time()
        for step, (features, lengths, targets, target_lengths, texts) in enumerate(train_loader, 1):
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
            if step % 1000 == 0:
                # Con el corpus completo una época dura varios minutos: avisar del avance.
                print(f"Época {epoch}: {step}/{len(train_loader)} lotes | pérdida media {total_loss / count:.3f} | "
                      f"{(time.time() - epoch_start) / 60:.1f} min", flush=True)
        metrics, examples = evaluate(model, dev_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / count,
                  "dev_loss": metrics["loss"], "dev_wer": metrics["wer"], "dev_cer": metrics["cer"],
                  "minutes": round((time.time() - epoch_start) / 60, 1)}
        with (output / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = (metrics["wer"], metrics["loss"])
        checkpoint = {"format_version": 1, "alphabet": ALPHABET,
                      "feature_config": asdict(config), "model_config": asdict(model_config),
                      "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "epoch": epoch, "metrics": record, "pretrained_weights": False}
        if score < best:
            best = score
            _save_atomic(checkpoint, output / "best.pt")
            (output / "dev_predictions.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        # Estado completo para `--resume`: si se corta, se continúa desde la última época terminada.
        _save_atomic({**checkpoint, "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
                      "best": list(best)}, output / "last.pt")
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


def test_checkpoint(checkpoint_path, test_path, train_path, dev_path, device_name="auto", cache_dir=None):
    partitions = {"train": read_manifest(train_path), "dev": read_manifest(dev_path), "test": read_manifest(test_path)}
    audit_partitions(partitions)
    model, config, device = load_model(checkpoint_path, device_name)
    dataset = SpeechDataset(partitions["test"], config, cache_dir)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    metrics, predictions = evaluate(model, loader, device)
    return {"metrics": metrics, "predictions": predictions}
