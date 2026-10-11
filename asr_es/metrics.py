"""Distancia de edición y tasas de error, sin bibliotecas de evaluación."""


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def error_rates(references, hypotheses):
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Se requieren referencias e hipótesis emparejadas")
    word_count = sum(len(text.split()) for text in references)
    char_count = sum(len(text) for text in references)
    if not word_count or not char_count:
        raise ValueError("Las referencias no pueden estar vacías")
    return {
        "wer": sum(edit_distance(r.split(), h.split()) for r, h in zip(references, hypotheses)) / word_count,
        "cer": sum(edit_distance(r, h) for r, h in zip(references, hypotheses)) / char_count,
    }
