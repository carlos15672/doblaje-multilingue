"""Entrenamiento supervisado desde cero, evaluación y carga de pesos propios."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
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
from .decoding import NGramLanguageModel, forced_align_words, prefix_beam_search
from .features import FeatureConfig, extract_features, resample
from .metrics import error_rates
from .model import CRNN, ModelConfig
from .text import ALPHABET, greedy_decode
from .vad import VADConfig, detect_speech


def choose_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible; utiliza --device cpu")
    return torch.device(value)


def decode_batch(log_probs, lengths):
    paths = log_probs.detach().argmax(-1).cpu()
    return [greedy_decode(paths[:int(length), i]) for i, length in enumerate(lengths)]


def _greedy_confidence(log_probs, length, batch_index=0):
    values = log_probs[:int(length), batch_index].detach().cpu()
    path = values.argmax(-1)
    emissions, previous = [], None
    for frame, token in enumerate(path.tolist()):
        if token and token != previous:
            emissions.append(float(values[frame, token]))
        previous = token
    return float(np.exp(np.mean(emissions))) if emissions else 0.0


def _merge_transcripts(left, right, maximum_overlap=6):
    left_words, right_words = left.split(), right.split()
    for size in range(min(maximum_overlap, len(left_words), len(right_words)), 0, -1):
        if left_words[-size:] == right_words[:size]:
            right_words = right_words[size:]
            break
    return " ".join(left_words + right_words)


def _condition_breakdown(references, hypotheses, metadata):
    groups = defaultdict(lambda: ([], []))
    for reference, hypothesis, item in zip(references, hypotheses, metadata):
        duration = item["duration"]
        duration_group = "corta_<3s" if duration < 3 else "media_3-8s" if duration <= 8 else "larga_>8s"
        conditions = {
            "acento": item["entry"].accent,
            "ambiente": item["entry"].environment,
            "microfono": item["entry"].microphone,
            "duracion": duration_group,
        }
        for dimension, value in conditions.items():
            key = (dimension, value or "desconocido")
            groups[key][0].append(reference)
            groups[key][1].append(hypothesis)
    result = defaultdict(dict)
    for (dimension, value), (refs, hyps) in groups.items():
        rates = error_rates(refs, hyps)
        result[dimension][value] = {"grabaciones": len(refs), **rates}
    return dict(result)


@torch.no_grad()
def evaluate(model, loader, device, entries=None, include_breakdown=False,
             beam_width=1, language_model=None, lm_weight=0.25):
    model.eval()
    loss_sum, count, references, hypotheses, metadata = 0.0, 0, [], [], []
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    for features, lengths, targets, target_lengths, texts in loader:
        log_probs, output_lengths = model(features.to(device), lengths)
        loss = criterion(log_probs, targets.to(device), output_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise ValueError("Pérdida CTC no finita en evaluación; revisar datos y longitudes")
        loss_sum += float(loss) * len(texts)
        count += len(texts)
        references.extend(texts)
        if beam_width > 1:
            for index, length in enumerate(output_lengths):
                values = log_probs[:int(length), index].detach().cpu().numpy()
                hypotheses.append(prefix_beam_search(values, beam_width, language_model, lm_weight, alternatives=1)[0]["texto"])
        else:
            hypotheses.extend(decode_batch(log_probs, output_lengths))
        if entries is not None:
            for index, length in enumerate(lengths.tolist()):
                metadata.append({"entry": entries[count - len(texts) + index], "duration": length * 0.01})
    rates = error_rates(references, hypotheses)
    rates["loss"] = loss_sum / count
    if include_breakdown and metadata:
        rates["desglose"] = _condition_breakdown(references, hypotheses, metadata)
    predictions = []
    for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
        item = {"referencia": reference, "hipotesis": hypothesis}
        if metadata:
            item.update(duracion=round(metadata[index]["duration"], 3),
                        acento=metadata[index]["entry"].accent,
                        ambiente=metadata[index]["entry"].environment,
                        microfono=metadata[index]["entry"].microphone)
        predictions.append(item)
    return rates, predictions


def _save_atomic(checkpoint, path):
    # Escribir aparte y renombrar: un corte a mitad de la escritura no deja el archivo dañado.
    partial = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, partial)
    os.replace(partial, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(train_path, dev_path, output, epochs=20, batch_size=8, learning_rate=0.001,
          device_name="auto", seed=42, hidden_size=128, cache_dir=None, workers=0, resume=False,
          augment=False, init_checkpoint=None):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or hidden_size < 4 or workers < 0:
        raise ValueError("Parámetros de entrenamiento no válidos")
    if resume and init_checkpoint:
        raise ValueError("--resume y --init-checkpoint son mutuamente excluyentes")
    config = FeatureConfig()
    model_config = ModelConfig(n_mels=config.n_mels, hidden_size=hidden_size)
    output = Path(output)
    previous = source = None
    warmstart = False
    if resume:
        # last.pt guarda modelo, optimizador y orden de los datos. best.pt de corridas anteriores
        # a esta opción solo guarda los pesos: en ese caso Adam vuelve a empezar.
        source = next((path for path in (output / "last.pt", output / "best.pt") if path.exists()), None)
        if source is None or not (output / "config.json").exists():
            raise ValueError("No hay una corrida que reanudar en esa carpeta (faltan config.json y last.pt/best.pt)")
        run_info = json.loads((output / "config.json").read_text(encoding="utf-8"))
        requested = {"batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
                     "augmentation": bool(augment)}
        changed = [key for key, value in requested.items()
                   if run_info.get(key, False if key == "augmentation" else None) != value]
        for key, path in (("train_manifest", train_path), ("dev_manifest", dev_path)):
            if Path(run_info.get(key, "")).resolve() != Path(path).resolve():
                changed.append(key)
        stored_model = ModelConfig(**run_info["model_config"])
        if stored_model != model_config:
            changed.append("model_config")
        for key, path in (("train_manifest_sha256", train_path), ("dev_manifest_sha256", dev_path)):
            if run_info.get(key) and run_info[key] != _file_sha256(path):
                changed.append(key)
        if changed:
            raise ValueError(f"Para reanudar usa los mismos parámetros que la corrida original; cambian: {changed}")
        previous = torch.load(source, map_location="cpu", weights_only=True)
    elif init_checkpoint:
        source = Path(init_checkpoint).resolve()
        if not source.is_file():
            raise ValueError(f"No existe el checkpoint inicial: {source}")
        if output.exists() and any(output.iterdir()):
            raise ValueError("La carpeta de entrenamiento debe estar vacía para --init-checkpoint")
        previous = torch.load(source, map_location="cpu", weights_only=True)
        source_model = ModelConfig(**previous.get("model_config", {}))
        source_features = previous.get("feature_config")
        if (previous.get("format_version") != 1 or previous.get("alphabet") != ALPHABET
                or source_features != asdict(config)):
            raise ValueError("El checkpoint inicial es incompatible con características o alfabeto actuales")
        if source_model != model_config:
            raise ValueError("El checkpoint inicial no corresponde a la arquitectura solicitada")
        warmstart = True
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
    model = CRNN(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start_epoch, best = 0, (float("inf"), float("inf"))
    if previous is not None:
        previous_model = ModelConfig(**previous.get("model_config", {}))
        if (previous.get("alphabet") != ALPHABET or previous_model != model_config
                or previous.get("feature_config") != asdict(config)):
            raise ValueError("El checkpoint no corresponde a esta configuración del modelo")
        model.load_state_dict(previous["state_dict"])
        if warmstart:
            print(f"Inicializando pesos desde {source}; optimizador nuevo", flush=True)
        else:
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
    if augment:
        from .augment import AugmentConfig
        augmentation = AugmentConfig()
    else:
        augmentation = None
    train_data = SpeechDataset(training, config, cache, augmentation=augmentation, seed=seed)
    dev_data = SpeechDataset(development, config, cache)
    # Verificar y preparar características antes de empezar una época costosa.
    # Con `workers` > 0 los archivos se leen en procesos paralelos: en Windows abrir un archivo
    # de la caché que no se ha abierto recientemente cuesta ~9 ms, y uno por uno serían ~50 min por época.
    if not resume:
        clean_train = SpeechDataset(training, config, cache)
        for name, dataset in (("train", clean_train), ("dev", dev_data)):
            checked, step = 0, 100 if workers == 0 else 10000
            for *_, texts in DataLoader(dataset, batch_size=64, collate_fn=collate, num_workers=workers):
                before, checked = checked, checked + len(texts)
                if checked // step > before // step or checked == len(dataset):
                    print(f"Preparando {name}: {checked}/{len(dataset)}", flush=True)
    else:
        print("Se omite la verificación previa: ya se hizo al iniciar esta corrida con los mismos datos", flush=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=workers, persistent_workers=workers > 0 and not augment,
                              generator=generator)
    dev_workers = min(workers, 4)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate,
                            num_workers=dev_workers, persistent_workers=dev_workers > 0)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    if not resume:
        run_info = {"feature_config": asdict(config), "model_config": asdict(model_config),
                    "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "device": str(device), "torch_version": str(torch.__version__), "audit": audit,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_manifest": str(Path(train_path).resolve()), "dev_manifest": str(Path(dev_path).resolve()),
                    "train_manifest_sha256": _file_sha256(train_path),
                    "dev_manifest_sha256": _file_sha256(dev_path),
                    "feature_cache": str(cache), "data_workers": workers, "augmentation": bool(augment),
                    "pretrained_weights": bool(warmstart)}
        if warmstart:
            run_info["initial_checkpoint"] = {
                "path": str(source), "sha256": _file_sha256(source),
                "epoch": previous.get("epoch"), "metrics": previous.get("metrics"),
            }
    else:
        run_info["epochs"] = epochs
        run_info.setdefault("reanudaciones", []).append({
            "desde": source.name, "despues_de_epoca": start_epoch, "data_workers": workers,
            "optimizador": "restaurado" if "optimizer" in previous else "nuevo"})
    (output / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    for epoch in range(start_epoch + 1, epochs + 1):
        train_data.set_epoch(epoch)
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
                      "epoch": epoch, "metrics": record,
                      "pretrained_weights": bool(run_info.get("pretrained_weights"))}
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
def transcribe(checkpoint_path, audio_path, device_name="auto", beam_width=16, language_model_path=None,
               lm_weight=0.25, min_confidence=0.15, vad_config=None):
    if beam_width < 1 or not 0 <= min_confidence <= 1 or lm_weight < 0:
        raise ValueError("Parámetros de decodificación no válidos")
    model, config, device = load_model(checkpoint_path, device_name)
    audio, rate = read_audio(audio_path)
    if len(audio) / rate > 2 * 60 * 60:
        raise ValueError("Divide archivos de más de dos horas antes de transcribir")
    normalized_audio = resample(audio, rate, config.sample_rate)
    vad_config = vad_config or VADConfig()
    segments, vad = detect_speech(normalized_audio, config.sample_rate, vad_config)
    if not segments:
        return {"texto": "", "segmentos": [], "palabras": [], "confianza": 0.0,
                "estado": "sin habla", "vad": vad,
                "nota": "El VAD no encontró voz; no se ejecutó reconocimiento"}
    language_model = NGramLanguageModel.load(language_model_path) if language_model_path else None
    transcript, accepted, all_words = "", [], []
    for segment in segments:
        chunk = normalized_audio[segment.start_sample:segment.end_sample]
        features = extract_features(chunk, config.sample_rate, config)
        log_probs, lengths = model(torch.from_numpy(features)[None].to(device), torch.tensor([len(features)]))
        valid = log_probs[:int(lengths[0]), 0].detach().cpu().numpy()
        if beam_width == 1:
            text = decode_batch(log_probs, lengths)[0]
            confidence = _greedy_confidence(log_probs, lengths[0])
            alternatives = [{"texto": text, "confianza": confidence}]
        else:
            alternatives = prefix_beam_search(valid, beam_width, language_model, lm_weight)
            text, confidence = alternatives[0]["texto"], alternatives[0]["confianza"]
        start, end = segment.seconds(config.sample_rate)
        rejected = not text or confidence < min_confidence
        words = [] if rejected else forced_align_words(
            valid, text, frame_seconds=config.hop_ms / 1000 * 2, offset_seconds=start)
        item = {"inicio": round(start, 3), "fin": round(end, 3), "texto": text,
                "confianza": round(confidence, 4), "rechazado": rejected,
                "alternativas": alternatives[:3]}
        accepted.append(item)
        if not rejected:
            transcript = _merge_transcripts(transcript, text) if transcript else text
            all_words.extend(words)
    usable = [item for item in accepted if not item["rechazado"]]
    overall_confidence = float(np.mean([item["confianza"] for item in usable])) if usable else 0.0
    return {"texto": transcript, "segmentos": accepted, "palabras": all_words,
            "confianza": round(overall_confidence, 4),
            "estado": "hipótesis del modelo" if usable else "voz rechazada por baja confianza",
            "vad": vad,
            "nota": "Confianza acústica heurística; debe calibrarse con un conjunto independiente"}


def test_checkpoint(checkpoint_path, test_path, train_path, dev_path, device_name="auto", cache_dir=None,
                    beam_width=1, language_model_path=None, lm_weight=0.25):
    if beam_width < 1 or lm_weight < 0:
        raise ValueError("Parámetros de decodificación no válidos")
    partitions = {"train": read_manifest(train_path), "dev": read_manifest(dev_path), "test": read_manifest(test_path)}
    audit = audit_partitions(partitions)
    model, config, device = load_model(checkpoint_path, device_name)
    dataset = SpeechDataset(partitions["test"], config, cache_dir)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    language_model = NGramLanguageModel.load(language_model_path) if language_model_path else None
    metrics, predictions = evaluate(model, loader, device, partitions["test"], include_breakdown=True,
                                    beam_width=beam_width, language_model=language_model, lm_weight=lm_weight)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {"metrics": metrics, "audit": audit,
            "checkpoint": {"path": str(Path(checkpoint_path).resolve()),
                           "sha256": _file_sha256(checkpoint_path),
                           "epoch": checkpoint.get("epoch"),
                           "model_config": checkpoint.get("model_config")},
            "test_manifest": {"path": str(Path(test_path).resolve()), "sha256": _file_sha256(test_path)},
            "decoder": {"beam_width": beam_width, "language_model": language_model_path,
                        "lm_weight": lm_weight},
            "predictions": predictions}
