"""Manifiestos CSV reproducibles y comprobaciones contra contaminación de prueba."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np

from .audio import read_audio
from .features import FeatureConfig, extract_features
from .text import encode, minimum_ctc_steps, normalize_text


@dataclass(frozen=True)
class Entry:
    audio: Path
    text: str
    speaker: str


def read_manifest(path):
    path = Path(path).resolve()
    entries, paths = [], set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"audio", "text", "speaker"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: faltan columnas audio,text,speaker")
        for line, row in enumerate(reader, 2):
            if not row.get("audio") or not row.get("speaker", "").strip():
                raise ValueError(f"{path}:{line}: falta archivo o identificador del hablante")
            audio = (path.parent / row["audio"]).resolve()
            if not audio.is_file():
                raise ValueError(f"{path}:{line}: no existe {audio}")
            if audio in paths:
                raise ValueError(f"{path}:{line}: audio duplicado")
            try:
                text = normalize_text(row.get("text") or "")
            except ValueError as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            paths.add(audio)
            entries.append(Entry(audio, text, row["speaker"].strip()))
    if not entries:
        raise ValueError(f"Manifiesto vacío: {path}")
    return entries


def audit_partitions(partitions):
    """Bloquea archivos y hablantes compartidos; informa textos repetidos."""
    report = {}
    for name, entries in partitions.items():
        report[name] = {"grabaciones": len(entries), "hablantes": len({e.speaker for e in entries})}
    names = list(partitions)
    text_overlap = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = partitions[left], partitions[right]
            if {e.audio for e in a} & {e.audio for e in b}:
                raise ValueError(f"Hay archivos compartidos entre {left} y {right}")
            if {e.speaker for e in a} & {e.speaker for e in b}:
                raise ValueError(f"Hay hablantes compartidos entre {left} y {right}; separa por persona")
            count = len({e.text for e in a} & {e.text for e in b})
            if count:
                text_overlap.append({"particiones": [left, right], "textos_compartidos": count})
    report["coincidencias_textuales"] = text_overlap
    return report


def cache_path(entry, config, cache):
    """Ruta en caché; cambia si cambia el audio o la configuración acústica."""
    stat = entry.audio.stat()
    key = json.dumps([str(entry.audio), stat.st_size, stat.st_mtime_ns, asdict(config), "features-v1"], sort_keys=True)
    return Path(cache) / (hashlib.sha256(key.encode()).hexdigest() + ".npy")


def load_features(entry, config, cache=None):
    """Características de una grabación (desde la caché si existen) y sus tokens, ya validados."""
    path = cache_path(entry, config, cache) if cache else None
    if path and path.exists():
        features = np.load(path, allow_pickle=False)
    else:
        audio, rate = read_audio(entry.audio)
        if len(audio) / rate > 30:
            raise ValueError(f"{entry.audio}: divide grabaciones de más de 30 s en segmentos transcritos")
        if np.sqrt(np.mean((audio - audio.mean()) ** 2)) < 1e-7:
            raise ValueError(f"Audio sin señal útil: {entry.audio}")
        features = extract_features(audio, rate, config)
        if path:
            # Escribir aparte y renombrar: un proceso interrumpido no deja .npy a medias.
            partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
            with partial.open("wb") as target:
                np.save(target, features, allow_pickle=False)
            os.replace(partial, path)
    tokens = encode(entry.text)
    if (len(features) + 1) // 2 < minimum_ctc_steps(tokens):
        raise ValueError(f"{entry.audio}: audio demasiado corto para la transcripción y reducción temporal CTC")
    return features, tokens


class SpeechDataset:
    def __init__(self, entries, config=FeatureConfig(), cache=None):
        self.entries, self.config = entries, config
        self.cache = Path(cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        features, tokens = load_features(entry, self.config, self.cache)
        return features, tokens, entry.text


def _cache_one(job):
    entry, config, cache = job
    try:
        load_features(entry, config, cache)
        return None
    except (ValueError, OSError, RuntimeError) as error:
        return str(error)


def cache_features(manifests, cache, config=FeatureConfig(), workers=0):
    """Calcula en paralelo las características de varios manifiestos en una caché compartida.

    Las grabaciones que no se pueden usar NO se eliminan en silencio: por cada manifiesto
    con problemas se escriben `<nombre>_excluidos.csv` (con el motivo) y
    `<nombre>_utilizables.csv`, que es el que debe pasarse a `train`.
    """
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Por defecto, la mitad de los núcleos lógicos (uno por núcleo físico con dos hilos por núcleo):
    # medido, 8 procesos rinden más que 15 porque el cálculo depende mucho de la memoria.
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    # Un hilo de álgebra lineal por proceso: los procesos ya reparten los núcleos.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    report = {"cache": str(cache), "procesos": workers, "manifiestos": {}}
    pool = multiprocessing.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        for manifest in manifests:
            manifest = Path(manifest).resolve()
            entries = read_manifest(manifest)
            jobs = [(entry, config, cache) for entry in entries]
            results = pool.imap(_cache_one, jobs, chunksize=64) if pool else map(_cache_one, jobs)
            start, errors = time.time(), []
            for done, (entry, error) in enumerate(zip(entries, results), 1):
                if error:
                    errors.append((entry, error))
                if done % 5000 == 0 or done == len(entries):
                    elapsed = time.time() - start
                    remaining = elapsed / done * (len(entries) - done)
                    print(f"{manifest.name}: {done:,}/{len(entries):,} | excluidas {len(errors)} | "
                          f"{elapsed / 60:.1f} min, faltan ~{remaining / 60:.0f} min", flush=True)
            summary = {"grabaciones": len(entries), "excluidas": len(errors)}
            if errors:
                usable = manifest.with_name(f"{manifest.stem}_utilizables.csv")
                excluded = manifest.with_name(f"{manifest.stem}_excluidos.csv")
                for path in (usable, excluded):
                    if path.exists():
                        raise ValueError(f"Ya existe {path}; bórralo o usa otra carpeta para no mezclar resultados")
                bad = {entry.audio for entry, _ in errors}
                with usable.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(["audio", "text", "speaker"])
                    writer.writerows((str(e.audio), e.text, e.speaker) for e in entries if e.audio not in bad)
                with excluded.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(["audio", "text", "speaker", "motivo"])
                    writer.writerows((str(e.audio), e.text, e.speaker, error) for e, error in errors)
                summary.update(utilizables=str(usable), excluidos=str(excluded))
            report["manifiestos"][str(manifest)] = summary
    finally:
        if pool:
            pool.close()
            pool.join()
    return report


def collate(samples):
    import torch
    lengths = torch.tensor([len(features) for features, _, _ in samples], dtype=torch.long)
    features = torch.zeros(len(samples), int(lengths.max()), samples[0][0].shape[1])
    for i, (array, _, _) in enumerate(samples):
        features[i, :len(array)] = torch.from_numpy(array)
    targets = torch.tensor([t for _, tokens, _ in samples for t in tokens], dtype=torch.long)
    target_lengths = torch.tensor([len(tokens) for _, tokens, _ in samples], dtype=torch.long)
    return features, lengths, targets, target_lengths, [text for _, _, text in samples]


def prepare_common_voice(root, output, limit=None):
    """Importa particiones oficiales, sin descargar ni generar transcripciones."""
    roots = root if isinstance(root, (list, tuple)) else [root]
    roots = [Path(item).resolve() for item in roots]
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Usa una carpeta de manifiestos nueva para no sobrescribir datos")
    if limit is not None and limit < 1:
        raise ValueError("El límite debe ser positivo")
    partitions, rejected = {}, []
    for split in ("train", "dev", "test"):
        entries = []
        filenames = set()
        for root in roots:
            path = root / f"{split}.tsv"
            with path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source, delimiter="\t")
                if not {"path", "sentence", "client_id"}.issubset(reader.fieldnames or []):
                    raise ValueError(f"{path}: formato Common Voice no reconocido")
                for line, row in enumerate(reader, 2):
                    audio = (root / "clips" / row["path"]).resolve()
                    try:
                        audio.relative_to(root / "clips")
                        text = normalize_text(row["sentence"])
                        speaker = row["client_id"].strip()
                        if not speaker or not audio.is_file():
                            raise ValueError("Falta archivo de audio o identificador del hablante")
                        if audio.name in filenames:
                            raise ValueError("Archivo Common Voice duplicado entre fuentes; no mezclar versiones del mismo corpus")
                        filenames.add(audio.name)
                        entries.append(Entry(audio, text, speaker))
                    except ValueError as error:
                        rejected.append({"origen": str(root), "particion": split, "linea": line, "motivo": str(error)})
        if not entries:
            raise ValueError(f"No hay grabaciones utilizables en la partición {split}")
        partitions[split] = entries
    # Auditar ANTES de limitar: un subconjunto no debe ocultar fugas del origen.
    audit_partitions(partitions)
    if limit:
        for entries in partitions.values():
            random.Random(42).shuffle(entries)
            del entries[limit:]
    report = audit_partitions(partitions)
    report["rechazos"] = rejected
    report["seleccion"] = "Si hay límite: muestreo de clips sin reemplazo, semilla 42 por partición; no estratificado por hablante"
    output.mkdir(parents=True, exist_ok=True)
    for split, entries in partitions.items():
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["audio", "text", "speaker"])
            writer.writerows((str(e.audio), e.text, e.speaker) for e in entries)
    (output / "importacion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
