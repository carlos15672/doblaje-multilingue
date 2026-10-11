"""Alfabeto español y decodificación CTC explícita."""
import unicodedata

ALPHABET = " abcdefghijklmnñopqrstuvwxyzáéíóúü"
BLANK = 0


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).lower()
    result = []
    for character in text:
        if character in ALPHABET:
            result.append(character)
        elif character.isspace() or unicodedata.category(character).startswith("P"):
            result.append(" ")
        else:
            raise ValueError(f"Carácter no soportado {character!r}: escribe números con letras y revisa la transcripción")
    normalized = " ".join("".join(result).split())
    if not normalized:
        raise ValueError("La transcripción no puede estar vacía")
    return normalized


def encode(text):
    return [ALPHABET.index(character) + 1 for character in normalize_text(text)]


def minimum_ctc_steps(tokens):
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def greedy_decode(indices):
    """Colapsar repeticiones ANTES de quitar blank; a,blank,a produce aa."""
    output, previous = [], None
    for index in indices:
        index = int(index)
        if not 0 <= index <= len(ALPHABET):
            raise ValueError("Índice fuera del alfabeto")
        if index != previous and index != BLANK:
            output.append(ALPHABET[index - 1])
        previous = index
    return " ".join("".join(output).split())
