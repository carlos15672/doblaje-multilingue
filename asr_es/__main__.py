"""Ejecutar desde la raíz: python -m asr_es --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from .audio import read_audio, record
from .data import audit_partitions, prepare_common_voice, read_manifest
from .features import FeatureConfig, extract_features


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
    prepare.add_argument("root", nargs="+", help="Una o más carpetas con clips/ y train.tsv, dev.tsv, test.tsv")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int, help="Máximo de ejemplos por partición para desarrollo")
    audit = commands.add_parser("audit", help="Verificar formato y separación de archivos y hablantes")
    audit.add_argument("--train", required=True)
    audit.add_argument("--dev", required=True)
    audit.add_argument("--test")
    training = commands.add_parser("train", help="Entrenar CRNN desde pesos aleatorios")
    training.add_argument("--train", required=True)
    training.add_argument("--dev", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    recognition = commands.add_parser("transcribe", help="Transcribir con un checkpoint propio previamente entrenado")
    recognition.add_argument("audio")
    recognition.add_argument("--checkpoint", required=True)
    recognition.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluation = commands.add_parser("evaluate", help="Evaluar test; comprobar aislamiento respecto a train y dev")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--train", required=True)
    evaluation.add_argument("--dev", required=True)
    evaluation.add_argument("--test", required=True)
    evaluation.add_argument("--output", required=True, help="Archivo JSON nuevo para métricas y predicciones")
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
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
        result = prepare_common_voice(args.root, args.output, args.limit)
    elif args.command == "audit":
        partitions = {name: read_manifest(getattr(args, name)) for name in ("train", "dev", "test") if getattr(args, name)}
        result = audit_partitions(partitions)
    elif args.command == "train":
        from .engine import train
        result = train(args.train, args.dev, args.output, args.epochs, args.batch_size, args.learning_rate,
                       args.device, args.seed, args.hidden_size)
    elif args.command == "transcribe":
        from .engine import transcribe
        result = transcribe(args.checkpoint, args.audio, args.device)
    else:
        from .engine import test_checkpoint
        output = Path(args.output)
        if output.exists():
            raise ValueError("Usa un archivo de evaluación nuevo")
        result = test_checkpoint(args.checkpoint, args.test, args.train, args.dev, args.device)
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
