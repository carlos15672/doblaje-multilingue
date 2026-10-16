"""Ejecutar desde la raíz: python -m asr_es --help."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np

from .audio import read_audio, record
from .data import audit_partitions, prepare_common_voice, read_manifest
from .features import FeatureConfig, extract_features

# Decodificador con modelo de lenguaje elegido con tune-decoder solo en dev (ver README).
LM_DECODER_DEFAULTS = {"lm_weight": 0.6, "word_bonus": 2.0, "prefix_penalty": 15.0, "phonetic": True}
NO_LM_DECODER = {"lm_weight": 0.0, "word_bonus": 0.0, "prefix_penalty": 0.0, "phonetic": False}


def main():
    parser = argparse.ArgumentParser(description="Reconocedor experimental de español, sin modelos preentrenados")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("diagnose", help="Mostrar equipo, PyTorch y GPU")
    capture = commands.add_parser("record", help="Grabar audio; requiere acción explícita del usuario")
    capture.add_argument("output")
    capture.add_argument("--seconds", type=float, default=5)
    analyze = commands.add_parser("analyze", help="Calcular características propias; NO transcribe")
    analyze.add_argument("audio")
    analyze.add_argument("--output", help="Guardar características en un archivo .npz nuevo")
    prepare = commands.add_parser("prepare-cv", help="Importar Common Voice ya descargado")
    prepare.add_argument("root", nargs="*", help="Una o más carpetas con clips/ y train.tsv, dev.tsv, test.tsv")
    prepare.add_argument("--train-tsv", help="TSV de entrenamiento cuando los metadatos están separados")
    prepare.add_argument("--dev-tsv", help="TSV de validación cuando los metadatos están separados")
    prepare.add_argument("--test-tsv", help="TSV de prueba cuando los metadatos están separados")
    prepare.add_argument("--clips", help="Carpeta clips/ compartida por los tres TSV separados")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int, help="Máximo de ejemplos por partición para desarrollo")
    caching = commands.add_parser("cache-features", help="Calcular en paralelo las características de manifiestos en una caché compartida")
    caching.add_argument("manifests", nargs="+", help="Uno o más CSV audio,text,speaker")
    caching.add_argument("--cache", required=True, help="Carpeta de caché compartida entre entrenamientos")
    caching.add_argument("--workers", type=int, default=0, help="Procesos en paralelo; 0 = la mitad de los núcleos lógicos")
    language = commands.add_parser("build-lm", help="Entrenar un modelo n-gram solo con manifiestos locales")
    language.add_argument("manifests", nargs="+", help="CSV cuyos textos se usarán, normalmente solo train")
    language.add_argument("--output", required=True)
    language.add_argument("--order", type=int, default=3)
    language.add_argument("--smoothing", choices=["add-k", "kneser-ney"], default="kneser-ney",
                          help="Suavizado; kneser-ney requiere --order 2")
    tuning = commands.add_parser("tune-decoder", help="Elegir peso del LM, premio por palabra y suavizado con dev")
    tuning.add_argument("--checkpoint", required=True)
    tuning.add_argument("--dev", required=True, help="Manifiesto de validación (nunca test)")
    tuning.add_argument("--lm", required=True, help="Modelo n-gram creado con build-lm")
    tuning.add_argument("--output", required=True, help="Archivo JSON nuevo con todos los resultados")
    tuning.add_argument("--cache", help="Caché de características creada con cache-features")
    tuning.add_argument("--clips", type=int, default=300, help="Clips de dev, muestra fija según --seed")
    tuning.add_argument("--seed", type=int, default=13)
    tuning.add_argument("--lm-weights", default="0.1,0.2,0.3,0.5")
    tuning.add_argument("--word-bonuses", default="0,1,2,3")
    tuning.add_argument("--smoothings", default="add-k,kneser-ney")
    tuning.add_argument("--prefix-penalties", default="0", help="Penalizaciones por prefijo inexistente a probar")
    tuning.add_argument("--phonetic-variants", action="store_true",
                        help="Medir también cada combinación con corrección de homófonos")
    tuning.add_argument("--beam-width", type=int, default=16)
    tuning.add_argument("--workers", type=int, default=0, help="Procesos de decodificación; 0 = la mitad de los núcleos")
    tuning.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    audit = commands.add_parser("audit", help="Verificar formato y separación de archivos y hablantes")
    audit.add_argument("--train", required=True)
    audit.add_argument("--dev", required=True)
    audit.add_argument("--test")
    training = commands.add_parser("train", help="Entrenar CRNN desde cero o desde un checkpoint propio")
    training.add_argument("--train", required=True)
    training.add_argument("--dev", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--augment", action="store_true", help="Ruido, ganancia, velocidad y máscaras tiempo/frecuencia")
    training.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                          help="cosine baja la tasa de aprendizaje suavemente durante las épocas")
    training.add_argument("--min-lr-factor", type=float, default=0.05,
                          help="Con cosine, la tasa final es learning_rate por este factor")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    training.add_argument("--cache", help="Caché de características creada con cache-features")
    training.add_argument("--workers", type=int,
                          help="Procesos que leen datos en paralelo; 0 = en el proceso principal. "
                               "Por defecto, la mitad de los núcleos lógicos (máximo 8)")
    continuation = training.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true",
                              help="Continuar una corrida interrumpida en la misma --output con los mismos parámetros")
    continuation.add_argument("--init-checkpoint",
                              help="Iniciar una corrida nueva desde pesos propios; usa optimizador nuevo")
    recognition = commands.add_parser("transcribe", help="Transcribir con un checkpoint propio previamente entrenado")
    recognition.add_argument("audio")
    recognition.add_argument("--checkpoint", required=True)
    recognition.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    recognition.add_argument("--beam-width", type=int, default=16, help="1 = greedy; valores mayores usan prefix beam search")
    recognition.add_argument("--lm", help="Modelo n-gram creado con build-lm")
    recognition.add_argument("--lm-weight", type=float, help="Peso del LM (por defecto, el elegido en dev)")
    recognition.add_argument("--word-bonus", type=float,
                             help="Premio por palabra terminada; compensa que el LM castigue cada palabra nueva")
    recognition.add_argument("--lm-smoothing", choices=["add-k", "kneser-ney"], help="Usar los conteos del LM con otro suavizado")
    recognition.add_argument("--prefix-penalty", type=float,
                             help="Penalización mientras la palabra a medias no inicia ninguna palabra del vocabulario")
    recognition.add_argument("--phonetic", action=argparse.BooleanOptionalAction,
                             help="Corregir palabras inexistentes por un homófono real (seseo, b/v, h, ll/y)")
    recognition.add_argument("--min-confidence", type=float, default=0.15)
    recognition.add_argument("--max-segment-seconds", type=float, default=25.0)
    recognition.add_argument("--overlap-seconds", type=float, default=0.4)
    evaluation = commands.add_parser("evaluate", help="Evaluar test; comprobar aislamiento respecto a train y dev")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--train", required=True)
    evaluation.add_argument("--dev", required=True)
    evaluation.add_argument("--test", required=True)
    evaluation.add_argument("--output", required=True, help="Archivo JSON nuevo para métricas y predicciones")
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluation.add_argument("--cache", help="Caché de características creada con cache-features")
    evaluation.add_argument("--beam-width", type=int, default=1)
    evaluation.add_argument("--lm", help="Modelo n-gram creado solo con train")
    evaluation.add_argument("--lm-weight", type=float)
    evaluation.add_argument("--word-bonus", type=float)
    evaluation.add_argument("--lm-smoothing", choices=["add-k", "kneser-ney"])
    evaluation.add_argument("--prefix-penalty", type=float)
    evaluation.add_argument("--phonetic", action=argparse.BooleanOptionalAction)
    evaluation.add_argument("--workers", type=int, default=0, help="Procesos para el beam search; 0 = la mitad de los núcleos")
    args = parser.parse_args()
    # Lo que no se indique toma el valor elegido con tune-decoder en dev (con --lm) o queda apagado (sin --lm).
    for key, value in (LM_DECODER_DEFAULTS if getattr(args, "lm", None) else NO_LM_DECODER).items():
        if getattr(args, key, "") is None:
            setattr(args, key, value)
    if args.command == "diagnose":
        import torch
        from .model import CRNN
        result = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__),
                  "cuda": torch.cuda.is_available(), "parametros_crnn": sum(p.numel() for p in CRNN().parameters()),
                  "pesos_preentrenados": False}
        if result["cuda"]:
            properties = torch.cuda.get_device_properties(0)
            result.update(gpu=properties.name, gpu_memory_gib=round(properties.total_memory / 1024**3, 1))
    elif args.command == "record":
        result = record(args.output, args.seconds)
    elif args.command == "analyze":
        if args.output and (Path(args.output).exists() or Path(args.output).suffix != ".npz"):
            raise ValueError("Usa una ruta .npz nueva")
        audio, rate = read_audio(args.audio)
        if len(audio) / rate > 30:
            raise ValueError("Analiza segmentos de hasta 30 segundos")
        config = FeatureConfig()
        features = extract_features(audio, rate, config)
        result = {"archivo": str(Path(args.audio).resolve()), "frecuencia_original": rate,
                  "segundos": len(audio) / rate, "ventanas": len(features), "bandas_mel": features.shape[1],
                  "valores_finitos": bool(np.isfinite(features).all()),
                  "rms": float(np.sqrt(np.mean(audio**2))), "config": asdict(config),
                  "estado": "Procesamiento acústico; no se ha realizado transcripción"}
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, features=features, metadata=json.dumps(result, ensure_ascii=False))
    elif args.command == "prepare-cv":
        split_files = None
        if any((args.train_tsv, args.dev_tsv, args.test_tsv)):
            split_files = {"train": args.train_tsv, "dev": args.dev_tsv, "test": args.test_tsv}
        result = prepare_common_voice(args.root, args.output, args.limit, split_files, args.clips)
    elif args.command == "cache-features":
        from .data import cache_features
        result = cache_features(args.manifests, args.cache, workers=args.workers)
    elif args.command == "build-lm":
        from .decoding import NGramLanguageModel
        texts = [entry.text for manifest in args.manifests for entry in read_manifest(manifest)]
        model = NGramLanguageModel.train(texts, order=args.order, smoothing=args.smoothing)
        model.save(args.output)
        result = {"modelo": str(Path(args.output).resolve()), "orden": args.order, "frases": len(texts)}
    elif args.command == "tune-decoder":
        from .engine import tune_decoder

        def numbers(text):
            return tuple(float(value) for value in text.split(",") if value.strip())

        result = tune_decoder(args.checkpoint, args.dev, args.lm, args.output, args.cache, args.clips, args.seed,
                              numbers(args.lm_weights), numbers(args.word_bonuses),
                              tuple(value.strip() for value in args.smoothings.split(",") if value.strip()),
                              args.beam_width, args.workers or max(1, (os.cpu_count() or 2) // 2), args.device,
                              numbers(args.prefix_penalties), args.phonetic_variants)
    elif args.command == "audit":
        partitions = {name: read_manifest(getattr(args, name)) for name in ("train", "dev", "test") if getattr(args, name)}
        result = audit_partitions(partitions)
    elif args.command == "train":
        from .engine import train
        workers = args.workers if args.workers is not None else min(8, max(1, (os.cpu_count() or 2) // 2))
        result = train(args.train, args.dev, args.output, epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.learning_rate, device_name=args.device, seed=args.seed,
                       hidden_size=args.hidden_size, cache_dir=args.cache, workers=workers, resume=args.resume,
                       augment=args.augment, init_checkpoint=args.init_checkpoint,
                       lr_schedule=args.lr_schedule, min_lr_factor=args.min_lr_factor)
    elif args.command == "transcribe":
        from .engine import transcribe
        from .vad import VADConfig
        vad = VADConfig(max_segment_seconds=args.max_segment_seconds, overlap_seconds=args.overlap_seconds)
        result = transcribe(args.checkpoint, args.audio, args.device, args.beam_width, args.lm,
                            args.lm_weight, args.min_confidence, vad, args.word_bonus, args.lm_smoothing,
                            args.prefix_penalty, args.phonetic)
    else:
        from .engine import test_checkpoint
        output = Path(args.output)
        if output.exists():
            raise ValueError("Usa un archivo de evaluación nuevo")
        result = test_checkpoint(args.checkpoint, args.test, args.train, args.dev, args.device, args.cache,
                                 args.beam_width, args.lm, args.lm_weight, args.word_bonus, args.lm_smoothing,
                                 args.workers or max(1, (os.cpu_count() or 2) // 2),
                                 args.prefix_penalty, args.phonetic)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"metricas": result["metrics"], "resultados": str(output.resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
