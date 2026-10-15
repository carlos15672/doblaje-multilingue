import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from asr_es.audio import read_audio, write_wav
from asr_es.augment import AugmentConfig, augment_audio, mask_features
from asr_es.data import (Entry, SpeechDataset, audit_partitions, cache_features, cache_path,
                         prepare_common_voice, read_manifest)
from asr_es.decoding import NGramLanguageModel, forced_align_words, prefix_beam_search
from asr_es.engine import load_model, train, transcribe
from asr_es.features import FeatureConfig, _resample_direct, extract_features, radix2_rfft, resample
from asr_es.metrics import error_rates
from asr_es.model import CRNN, ModelConfig
from asr_es.text import encode, greedy_decode, minimum_ctc_steps, normalize_text
from asr_es.vad import VADConfig, detect_speech

torch.set_num_threads(2)


class AcousticTests(unittest.TestCase):
    def test_fft_matches_independent_numpy_reference(self):
        x = np.random.default_rng(7).normal(size=(7, 400))
        np.testing.assert_allclose(radix2_rfft(x, 512), np.fft.rfft(x, n=512), atol=1e-11)

    def test_resampling_suppresses_aliasing(self):
        t = np.arange(48000) / 48000
        low = resample(np.sin(2 * np.pi * 1000 * t), 48000, 16000)[200:-200]
        high = resample(np.sin(2 * np.pi * 10000 * t), 48000, 16000)[200:-200]
        self.assertGreater(np.sqrt(np.mean(low**2)), 0.65)
        self.assertLess(np.sqrt(np.mean(high**2)), 0.015)

    def test_fast_resampling_matches_direct_reference(self):
        # Pesos precalculados por fase frente al cálculo directo de cada salida, bordes incluidos.
        rng = np.random.default_rng(3)
        for rate in (48000, 44100, 32000, 22050, 8000):
            for length in (1, 50, 300, 20000):
                x = rng.normal(size=length)
                outputs = np.arange(len(resample(x, rate, 16000)))
                np.testing.assert_allclose(resample(x, rate, 16000), _resample_direct(x, rate, 16000, outputs), atol=1e-9)

    def test_features_preserve_temporal_changes(self):
        t = np.arange(16000) / 16000
        x = np.r_[np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 2000 * t)]
        f = extract_features(x, 16000)
        self.assertEqual(f.shape, (199, 40))
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(np.linalg.norm(f[10:80].mean(0) - f[110:180].mean(0)), 1)

    def test_short_and_silent_audio_is_finite(self):
        for x in (np.zeros(1), np.zeros(16000), np.ones(400)):
            self.assertTrue(np.isfinite(extract_features(x, 16000)).all())
        with self.assertRaises(ValueError):
            extract_features(np.array([]), 16000)

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            samples = np.linspace(-0.9, 0.9, 1000)
            write_wav(path, samples)
            restored, rate = read_audio(path)
            self.assertEqual(rate, 16000)
            np.testing.assert_allclose(restored, samples, atol=5e-5)
            with self.assertRaises(FileExistsError):
                write_wav(path, samples)

    def test_vad_rejects_silence_and_segments_long_speech(self):
        rate = 16000
        silence, diagnostics = detect_speech(np.zeros(rate), rate)
        self.assertEqual(silence, [])
        self.assertEqual(diagnostics["speech_ratio"], 0)
        t = np.arange(2 * rate) / rate
        audio = 0.5 * np.sin(2 * np.pi * 220 * t)
        config = VADConfig(max_segment_seconds=0.8, overlap_seconds=0.2)
        segments, _ = detect_speech(audio, rate, config)
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(all((s.end_sample - s.start_sample) / rate <= 0.8 for s in segments))

    def test_augmentations_are_reproducible_and_preserve_shape_contract(self):
        x = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
        config = AugmentConfig(probability=1, speed_probability=0, noise_probability=1)
        first = augment_audio(x, 16000, np.random.default_rng(9), config)
        second = augment_audio(x, 16000, np.random.default_rng(9), config)
        np.testing.assert_array_equal(first, second)
        features = np.ones((50, 40), dtype=np.float32)
        masked = mask_features(features, np.random.default_rng(5), config)
        self.assertEqual(masked.shape, features.shape)
        self.assertTrue(np.any(masked == 0))


class TextAndDataTests(unittest.TestCase):
    def test_ctc_repetitions_keep_blank_separation(self):
        a, b = encode("ab")
        self.assertEqual(greedy_decode([a, a, 0, a, b, b, 0]), "aab")
        self.assertEqual(minimum_ctc_steps(encode("llama")), 6)

    def test_spanish_accents_and_explicit_number_policy(self):
        self.assertEqual(normalize_text("¡El NIÑO pidió café!"), "el niño pidió café")
        with self.assertRaises(ValueError):
            normalize_text("tengo 23 años")

    def test_error_rates_use_corpus_denominators(self):
        rates = error_rates(["mi casa", "hola"], ["mi taza", ""])
        self.assertAlmostEqual(rates["wer"], 2 / 3)
        self.assertEqual(error_rates(["hola"], ["hola"])["cer"], 0)

    def test_blocks_speaker_and_file_leakage(self):
        a = Entry(Path("a.wav"), "hola", "persona1")
        b = Entry(Path("b.wav"), "adiós", "persona1")
        with self.assertRaisesRegex(ValueError, "hablantes"):
            audit_partitions({"train": [a], "dev": [b]})
        with self.assertRaisesRegex(ValueError, "archivos"):
            audit_partitions({"train": [a], "dev": [a]})

    def test_common_voice_import_reports_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            for split in ("train", "dev", "test"):
                (root / "clips" / f"{split}.wav").touch()
                with (root / f"{split}.tsv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["path", "sentence", "client_id"])
                    writer.writerow([f"{split}.wav", "tengo 2", split])
                    writer.writerow([f"{split}.wav", "hola", split])
            report = prepare_common_voice(root, root / "manifests")
            self.assertEqual(len(report["rechazos"]), 3)
            self.assertEqual(len(read_manifest(root / "manifests" / "train.csv")), 1)

    def test_common_voice_import_accepts_separate_tsv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "audio" / "clips"
            metadata = root / "metadata"
            clips.mkdir(parents=True)
            metadata.mkdir()
            split_files = {}
            for split in ("train", "dev", "test"):
                (clips / f"{split}.wav").touch()
                tsv = metadata / f"{split}.tsv"
                split_files[split] = tsv
                with tsv.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["path", "sentence", "client_id", "accents"])
                    writer.writerow([f"{split}.wav", "¡Hola!", split, "México"])
            report = prepare_common_voice(None, root / "manifests", split_files=split_files, clips=clips)
            self.assertEqual(report["train"]["grabaciones"], 1)
            imported = read_manifest(root / "manifests" / "train.csv")[0]
            self.assertEqual((imported.text, imported.accent), ("hola", "México"))

    def test_common_voice_import_requires_audio_for_separate_tsv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_files = {}
            for split in ("train", "dev", "test"):
                tsv = root / f"{split}.tsv"
                tsv.write_text("path\tsentence\tclient_id\na.wav\thola\tpersona\n", encoding="utf-8")
                split_files[split] = tsv
            with self.assertRaisesRegex(ValueError, "No existe la carpeta de audio"):
                prepare_common_voice(None, root / "manifests", split_files=split_files, clips=root / "clips")

    def test_shared_cache_reports_unusable_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(8000) / 16000
            write_wav(root / "voz.wav", 0.5 * np.sin(2 * np.pi * 300 * t))
            write_wav(root / "silencio.wav", np.zeros(8000))
            with (root / "train.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["audio", "text", "speaker"])
                writer.writerow(["voz.wav", "hola", "uno"])
                writer.writerow(["silencio.wav", "casa", "dos"])
            for workers, cache in ((1, root / "cache_1"), (2, root / "cache_2")):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = cache_features([root / "train.csv"], cache, workers=workers)
                summary = report["manifiestos"][str((root / "train.csv").resolve())]
                self.assertEqual((summary["grabaciones"], summary["excluidas"]), (2, 1))
                usable = read_manifest(summary["utilizables"])
                self.assertEqual([entry.text for entry in usable], ["hola"])
                self.assertTrue(cache_path(usable[0], FeatureConfig(), cache).exists())
                self.assertEqual(SpeechDataset(usable, FeatureConfig(), cache)[0][2], "hola")
                for path in (summary["utilizables"], summary["excluidos"]):
                    Path(path).unlink()

    def test_prefix_beam_search_ngram_and_word_timestamps(self):
        a, space, b = encode("a b")
        classes = len(" abcdefghijklmnñopqrstuvwxyzáéíóúü") + 1
        path = [0, a, 0, space, 0, b, 0]
        probabilities = np.full((len(path), classes), -12.0)
        for frame, token in enumerate(path):
            probabilities[frame, token] = -0.01
        result = prefix_beam_search(probabilities, beam_width=4)
        self.assertEqual(result[0]["texto"], "a b")
        words = forced_align_words(probabilities, "a b")
        self.assertEqual([item["texto"] for item in words], ["a", "b"])
        model = NGramLanguageModel.train(["a b", "a b", "a casa"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lm.json"
            model.save(path)
            restored = NGramLanguageModel.load(path)
            self.assertGreater(restored.score("a b", final=True), restored.score("b a", final=True))


class NetworkTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "No hay GPU CUDA")
    def test_cuda_training_step_with_default_architecture(self):
        model = CRNN().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        previous = model.conv1.weight.detach().clone()
        probabilities, lengths = model(torch.randn(2, 61, 40, device="cuda"), torch.tensor([61, 49]))
        targets = torch.tensor(encode("hola") + encode("casa"), device="cuda")
        loss = torch.nn.CTCLoss()(probabilities, targets, lengths, torch.tensor([4, 4]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        self.assertFalse(torch.equal(previous, model.conv1.weight))

    def test_padding_does_not_change_valid_predictions(self):
        torch.manual_seed(1)
        model = CRNN(ModelConfig(hidden_size=8, conv_channels=4, dropout=0)).eval()
        short = torch.randn(1, 17, 40)
        single, _ = model(short, torch.tensor([17]))
        batch = torch.randn(2, 38, 40)
        batch[0, :17] = short[0]
        mixed, lengths = model(batch, torch.tensor([17, 38]))
        self.assertEqual(lengths.tolist(), [9, 19])
        torch.testing.assert_close(single[:, 0], mixed[:9, 0], atol=2e-6, rtol=1e-5)

    def test_ctc_reaches_both_convolution_and_recurrent_weights(self):
        model = CRNN(ModelConfig(hidden_size=8, conv_channels=4))
        predictions, lengths = model(torch.randn(2, 31, 40), torch.tensor([31, 25]))
        loss = torch.nn.CTCLoss()(predictions, torch.tensor(encode("hola") + encode("sol")), lengths, torch.tensor([4, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.conv1.weight, model.recurrent.weight_ih_l0):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_training_checkpoint_reload_and_transcription(self):
        # Señales sintéticas: verifica infraestructura, NO precisión del español.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(6400) / 16000
            for split in ("train", "dev"):
                with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["audio", "text", "speaker"])
                    for i, letter in enumerate(("a", "e")):
                        filename = f"{split}{i}.wav"
                        write_wav(root / filename, 0.5 * np.sin(2 * np.pi * (300 + 200 * i) * t))
                        writer.writerow([filename, letter, f"synthetic-{split}-{i}"])
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(root / "train.csv", root / "dev.csv", root / "run", epochs=1,
                               batch_size=2, hidden_size=8, device_name="cpu")
            model, config, _ = load_model(result["checkpoint"], "cpu")
            self.assertEqual(config.sample_rate, 16000)
            self.assertEqual(model.config.hidden_size, 8)
            result = transcribe(result["checkpoint"], root / "dev0.wav", "cpu")
            self.assertIn("texto", result)
            self.assertIn(result["estado"], ("hipótesis del modelo", "voz rechazada por baja confianza"))
            self.assertTrue(result["segmentos"])
            self.assertIn("confianza", result)

    def test_resume_continues_interrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(6400) / 16000
            for split in ("train", "dev"):
                with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["audio", "text", "speaker"])
                    for i, letter in enumerate(("a", "e")):
                        write_wav(root / f"{split}{i}.wav", 0.5 * np.sin(2 * np.pi * (300 + 200 * i) * t))
                        writer.writerow([f"{split}{i}.wav", letter, f"synthetic-{split}-{i}"])
            options = dict(batch_size=2, hidden_size=8, device_name="cpu")
            run = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                train(root / "train.csv", root / "dev.csv", run, epochs=1, **options)
                with self.assertRaises(ValueError):  # carpeta ocupada sin --resume
                    train(root / "train.csv", root / "dev.csv", run, epochs=2, **options)
                with self.assertRaises(ValueError):  # parámetros distintos a los originales
                    train(root / "train.csv", root / "dev.csv", run, epochs=2, resume=True,
                          batch_size=1, hidden_size=8, device_name="cpu")
                train(root / "train.csv", root / "dev.csv", run, epochs=2, resume=True, **options)
                best_epoch = torch.load(run / "best.pt", weights_only=True)["epoch"]
                warm = root / "warm"
                train(root / "train.csv", root / "dev.csv", warm, epochs=1,
                      init_checkpoint=run / "best.pt", **options)
                warm_config = json.loads((warm / "config.json").read_text(encoding="utf-8"))
                self.assertTrue(warm_config["pretrained_weights"])
                self.assertEqual(warm_config["initial_checkpoint"]["path"], str((run / "best.pt").resolve()))
                self.assertEqual(torch.load(warm / "best.pt", weights_only=True)["epoch"], 1)
                (run / "last.pt").unlink()  # corridas viejas: solo best.pt, sin optimizador
                train(root / "train.csv", root / "dev.csv", run, epochs=3, resume=True, **options)
            epochs = [json.loads(line)["epoch"] for line in (run / "history.jsonl").read_text().splitlines()]
            self.assertEqual(epochs, [1, 2] + list(range(best_epoch + 1, 4)))
            resumes = json.loads((run / "config.json").read_text(encoding="utf-8"))["reanudaciones"]
            self.assertEqual([r["optimizador"] for r in resumes], ["restaurado", "nuevo"])
            self.assertEqual(torch.load(run / "last.pt", weights_only=True)["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
