"""Decodificación CTC, modelo n-gram y alineación temporal implementados localmente."""
from collections import Counter, defaultdict
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from .text import ALPHABET, encode, minimum_ctc_steps


NEGATIVE_INFINITY = -float("inf")


def _log_add(*values):
    finite = [value for value in values if value != NEGATIVE_INFINITY]
    if not finite:
        return NEGATIVE_INFINITY
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def tokens_to_text(tokens):
    return " ".join("".join(ALPHABET[int(token) - 1] for token in tokens).split())


def _tokens_to_raw_text(tokens):
    return "".join(ALPHABET[int(token) - 1] for token in tokens)


SMOOTHINGS = ("add-k", "kneser-ney")


class NGramLanguageModel:
    """Modelo de palabras entrenado solo con texto local.

    Suavizados:
    - ``add-k``: suma k a todos los conteos. Con vocabularios grandes reparte casi toda la
      masa entre palabras no vistas y castiga por igual cada palabra nueva.
    - ``kneser-ney``: bigrama interpolado con descuento absoluto. La probabilidad de respaldo
      de una palabra depende de en cuántos contextos distintos aparece (continuación), no de su
      frecuencia. Se calcula con los mismos conteos guardados, sin releer el texto.
    """

    def __init__(self, order=2, add_k=0.5, counts=None, context_counts=None, vocabulary=None,
                 smoothing="add-k", discount=0.75):
        if order < 1 or add_k <= 0 or smoothing not in SMOOTHINGS or not 0 < discount < 1:
            raise ValueError("Configuración n-gram no válida")
        if smoothing == "kneser-ney" and order != 2:
            raise ValueError("Kneser-Ney está implementado para bigramas (--order 2)")
        self.order = order
        self.add_k = float(add_k)
        self.smoothing = smoothing
        self.discount = float(discount)
        self.counts = Counter(counts or {})
        self.context_counts = Counter(context_counts or {})
        self.vocabulary = set(vocabulary or [])
        self._kneser_ney = None
        self._word_cache, self._score_cache = {}, {}

    def _prepare_kneser_ney(self):
        followers, predecessors = Counter(), Counter()
        for (previous, word), count in self.counts.items():
            if count > 0:
                followers[previous] += 1
                predecessors[word] += 1
        # +1 de continuación para <unk>: así la masa de respaldo suma 1 incluyendo palabras nuevas.
        self._kneser_ney = (followers, predecessors, len(self.counts) + 1)

    def _continuation(self, word):
        _, predecessors, total = self._kneser_ney
        return (predecessors[word] if word != "<unk>" else 1) / total

    @classmethod
    def train(cls, texts, order=2, add_k=0.5, smoothing="add-k", discount=0.75):
        model = cls(order, add_k, smoothing=smoothing, discount=discount)
        for text in texts:
            words = text.split()
            model.vocabulary.update(words)
            padded = ["<s>"] * (order - 1) + words + ["</s>"]
            for index in range(order - 1, len(padded)):
                context = tuple(padded[index - order + 1:index])
                model.counts[context + (padded[index],)] += 1
                model.context_counts[context] += 1
        if not model.vocabulary:
            raise ValueError("No hay texto para entrenar el modelo n-gram")
        model.vocabulary.update(("</s>", "<unk>"))
        return model

    def _word_score(self, context, word):
        word = word if word in self.vocabulary else "<unk>"
        context = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        key = context + (word,)
        if key in self._word_cache:
            return self._word_cache[key]
        if self.smoothing == "kneser-ney":
            if self._kneser_ney is None:
                self._prepare_kneser_ney()
            followers = self._kneser_ney[0]
            context_count = self.context_counts[context]
            continuation = self._continuation(word)
            if context_count:
                discounted = max(self.counts[key] - self.discount, 0) / context_count
                backoff = self.discount * followers[context[0]] / context_count
                probability = discounted + backoff * continuation
            else:
                probability = continuation
        else:
            vocabulary_size = max(1, len(self.vocabulary))
            probability = (self.counts[key] + self.add_k) / (self.context_counts[context] + self.add_k * vocabulary_size)
        result = math.log(probability)
        if len(self._word_cache) > 2_000_000:
            self._word_cache.clear()
        self._word_cache[key] = result
        return result

    def score(self, text, final=False):
        cache_key = (text, final)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        words = text.split()
        # Durante búsqueda, la última palabra aún puede crecer si no termina en espacio.
        complete = words if final or text.endswith(" ") else words[:-1]
        context = ["<s>"] * (self.order - 1)
        score = 0.0
        for word in complete:
            score += self._word_score(context, word)
            context.append(word if word in self.vocabulary else "<unk>")
        if final:
            score += self._word_score(context, "</s>")
        if len(self._score_cache) > 500_000:
            self._score_cache.clear()
        self._score_cache[cache_key] = score
        return score

    def save(self, path):
        payload = {
            "format_version": 1,
            "order": self.order,
            "add_k": self.add_k,
            "smoothing": self.smoothing,
            "discount": self.discount,
            "vocabulary": sorted(self.vocabulary),
            "counts": [[list(key), value] for key, value in self.counts.items()],
            "context_counts": [[list(key), value] for key, value in self.context_counts.items()],
        }
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"El archivo ya existe: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, smoothing=None):
        """``smoothing`` permite usar los conteos de un archivo con otro suavizado (p. ej. uno add-k como Kneser-Ney)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Modelo n-gram incompatible")
        return cls(
            payload["order"], payload["add_k"],
            {tuple(key): value for key, value in payload["counts"]},
            {tuple(key): value for key, value in payload["context_counts"]},
            payload["vocabulary"],
            smoothing=smoothing or payload.get("smoothing", "add-k"),
            discount=payload.get("discount", 0.75),
        )


def prefix_beam_search(log_probs, beam_width=16, language_model=None, lm_weight=0.25,
                       word_bonus=0.0, token_prune=20, alternatives=3):
    """Prefix beam search CTC en espacio logarítmico."""
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ALPHABET) + 1 or beam_width < 1:
        raise ValueError("Probabilidades o ancho de búsqueda no válidos")
    beams = {(): (0.0, NEGATIVE_INFINITY)}

    def ranking(prefix, probabilities, final=False):
        acoustic = _log_add(*probabilities)
        text = _tokens_to_raw_text(prefix)
        lm_score = language_model.score(text, final=final) if language_model else 0.0
        return acoustic + lm_weight * lm_score + word_bonus * len(text.split())

    for frame in values:
        if token_prune and token_prune < len(frame):
            candidates = np.argpartition(frame, -token_prune)[-token_prune:]
        else:
            candidates = range(len(frame))
        next_beams = defaultdict(lambda: [NEGATIVE_INFINITY, NEGATIVE_INFINITY])
        for prefix, (prob_blank, prob_nonblank) in beams.items():
            total = _log_add(prob_blank, prob_nonblank)
            for token in candidates:
                token = int(token)
                probability = float(frame[token])
                if token == 0:
                    next_beams[prefix][0] = _log_add(next_beams[prefix][0], total + probability)
                elif prefix and token == prefix[-1]:
                    next_beams[prefix][1] = _log_add(next_beams[prefix][1], prob_nonblank + probability)
                    extended = prefix + (token,)
                    next_beams[extended][1] = _log_add(next_beams[extended][1], prob_blank + probability)
                else:
                    extended = prefix + (token,)
                    next_beams[extended][1] = _log_add(next_beams[extended][1], total + probability)
        best = sorted(next_beams.items(), key=lambda item: ranking(item[0], item[1]), reverse=True)[:beam_width]
        beams = {prefix: tuple(probabilities) for prefix, probabilities in best}

    ordered = sorted(beams.items(), key=lambda item: ranking(item[0], item[1], final=True), reverse=True)
    results = []
    for prefix, probabilities in ordered[:max(1, alternatives)]:
        acoustic = _log_add(*probabilities)
        results.append({
            "texto": tokens_to_text(prefix),
            "score": ranking(prefix, probabilities, final=True),
            "acoustic_score": acoustic,
            "confianza": float(math.exp(min(0.0, acoustic / max(1, len(values))))),
        })
    return results


_WORKER_LANGUAGE_MODEL = None


def _init_decoder_worker(lm_path, smoothing):
    global _WORKER_LANGUAGE_MODEL
    _WORKER_LANGUAGE_MODEL = NGramLanguageModel.load(lm_path, smoothing) if lm_path else None


def _decode_one(job):
    values, beam_width, lm_weight, word_bonus = job
    model = _WORKER_LANGUAGE_MODEL if lm_weight > 0 else None
    return prefix_beam_search(values, beam_width, model, lm_weight, word_bonus, alternatives=1)[0]["texto"]


class DecoderPool:
    """Decodifica muchas secuencias con prefix beam search, en paralelo si ``workers`` > 1.

    Cada proceso carga el modelo de lenguaje una sola vez y se reutiliza para varias
    combinaciones de peso y premio por palabra.
    """

    def __init__(self, lm_path=None, smoothing=None, workers=1):
        self.lm_path, self.smoothing, self.workers = lm_path, smoothing, max(1, int(workers))
        self._pool = None

    def __enter__(self):
        if self.workers > 1:
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(self.workers, _init_decoder_worker, (self.lm_path, self.smoothing))
        else:
            _init_decoder_worker(self.lm_path, self.smoothing)
        return self

    def __exit__(self, *error):
        if self._pool:
            self._pool.terminate() if error[0] else self._pool.close()
            self._pool.join()

    def decode(self, arrays, beam_width=16, lm_weight=0.25, word_bonus=0.0):
        jobs = [(np.asarray(values, dtype=np.float32), beam_width, lm_weight, word_bonus) for values in arrays]
        if self._pool:
            return self._pool.map(_decode_one, jobs, chunksize=4)
        return [_decode_one(job) for job in jobs]


def forced_align_words(log_probs, text, frame_seconds=0.02, offset_seconds=0.0):
    """Alinea la hipótesis CTC y devuelve timestamps aproximados por palabra."""
    values = np.asarray(log_probs, dtype=np.float64)
    tokens = encode(text)
    if not tokens:
        return []
    states = [0]
    for token in tokens:
        states.extend((token, 0))
    time_steps, state_count = len(values), len(states)
    if time_steps < minimum_ctc_steps(tokens):
        return []
    previous = np.full(state_count, NEGATIVE_INFINITY)
    previous[0] = values[0, 0]
    if state_count > 1:
        previous[1] = values[0, states[1]]
    back = np.full((time_steps, state_count), -1, dtype=np.int16)
    for time_index in range(1, time_steps):
        current = np.full(state_count, NEGATIVE_INFINITY)
        for state_index, label in enumerate(states):
            choices = [(previous[state_index], state_index)]
            if state_index > 0:
                choices.append((previous[state_index - 1], state_index - 1))
            if state_index > 1 and label != 0 and label != states[state_index - 2]:
                choices.append((previous[state_index - 2], state_index - 2))
            best_score, best_state = max(choices)
            current[state_index] = best_score + values[time_index, label]
            back[time_index, state_index] = best_state
        previous = current
    final_candidates = [state_count - 1]
    if state_count > 1:
        final_candidates.append(state_count - 2)
    state = max(final_candidates, key=lambda item: previous[item])
    path = [state]
    for time_index in range(time_steps - 1, 0, -1):
        state = int(back[time_index, state])
        path.append(state)
    path.reverse()

    character_times = []
    for token_index in range(len(tokens)):
        target_state = 2 * token_index + 1
        frames = [index for index, state_index in enumerate(path) if state_index == target_state]
        if not frames:
            character_times.append(None)
        else:
            character_times.append((frames[0] * frame_seconds, (frames[-1] + 1) * frame_seconds))
    words, current_chars, current_times = [], [], []
    for character, timing in zip(text, character_times):
        if character == " ":
            if current_chars and current_times:
                words.append({"texto": "".join(current_chars),
                              "inicio": round(offset_seconds + current_times[0][0], 3),
                              "fin": round(offset_seconds + current_times[-1][1], 3)})
            current_chars, current_times = [], []
        else:
            current_chars.append(character)
            if timing:
                current_times.append(timing)
    if current_chars and current_times:
        words.append({"texto": "".join(current_chars),
                      "inicio": round(offset_seconds + current_times[0][0], 3),
                      "fin": round(offset_seconds + current_times[-1][1], 3)})
    return words
