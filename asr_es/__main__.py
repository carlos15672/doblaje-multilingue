"""Fachada de compatibilidad para ejecutar la implementación ASR canónica."""
import sys

from actualizacion.asr_es.__main__ import main


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
