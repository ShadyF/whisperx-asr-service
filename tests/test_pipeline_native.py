"""Focused unit tests for native faster-whisper decoding without ML dependencies."""

import importlib
import os
from pathlib import Path
import re
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


class Segment:
    """Small stand-in for faster-whisper's native segment object."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class Counter:
    """Record metric labels and increments without a Prometheus registry."""

    def __init__(self):
        self.labels_seen = []
        self.count = 0

    def labels(self, **labels):
        self.labels_seen.append(labels)
        return self

    def inc(self):
        self.count += 1


class TestPipelineNativeDecode(unittest.TestCase):
    """Load a fresh native pipeline module for every isolated test."""

    def setUp(self):
        # Remove prior stubs so each test imports isolated startup configuration.
        self.saved_modules = {
            name: sys.modules.pop(name)
            for name in (
                "app.pipeline", "app.metrics", "numpy", "torch", "whisperx",
                "whisperx.audio", "whisperx.diarize", "whisperx.vads",
            )
            if name in sys.modules
        }
        self.saved_pipeline = getattr(sys.modules.get("app"), "pipeline", None)

        # Install only the interfaces used by the pipeline unit tests.
        self._install_stubs()

    def tearDown(self):
        # Restore the original module state for other test modules.
        for name in (
            "app.pipeline", "app.metrics", "numpy", "torch", "whisperx",
            "whisperx.audio", "whisperx.diarize", "whisperx.vads",
        ):
            sys.modules.pop(name, None)
        sys.modules.update(self.saved_modules)
        if self.saved_pipeline is not None:
            sys.modules["app"].pipeline = self.saved_pipeline
        else:
            sys.modules["app"].__dict__.pop("pipeline", None)

    def _install_stubs(self):
        # Provide lightweight numerical and Torch interfaces without GPU setup.
        numpy = types.ModuleType("numpy")
        numpy.ndarray = list
        numpy.floating = float
        numpy.integer = int
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch.device = lambda value: value

        # Provide the WhisperX modules used while importing the pipeline.
        whisperx = types.ModuleType("whisperx")
        whisperx.__path__ = []
        whisperx.load_model = Mock()
        whisperx.load_align_model = Mock()
        whisperx.align = Mock()
        whisperx.assign_word_speakers = Mock(side_effect=lambda _, result, **__: result)
        audio = types.ModuleType("whisperx.audio")
        audio.SAMPLE_RATE = 10
        diarize = types.ModuleType("whisperx.diarize")
        diarize.DiarizationPipeline = Mock()

        # Make a local VAD base class for the native VAD branch check.
        class Vad:
            pass

        vads = types.ModuleType("whisperx.vads")
        vads.Vad = Vad
        vads.Pyannote = types.SimpleNamespace()
        metrics = types.ModuleType("app.metrics")
        metrics.WHISPERX_DECODE_CHUNKS_TOTAL = Counter()
        metrics.WHISPERX_DECODE_FAILURES_TOTAL = Counter()

        # Register all stubs before importing the pipeline under test.
        sys.modules.update(
            {
                "numpy": numpy,
                "torch": torch,
                "whisperx": whisperx,
                "whisperx.audio": audio,
                "whisperx.diarize": diarize,
                "whisperx.vads": vads,
                "app.metrics": metrics,
            }
        )

    def _pipeline(self, **environment):
        # Re-import after applying only the requested process-wide settings.
        sys.modules.pop("app.pipeline", None)
        with patch.dict(os.environ, environment, clear=True):
            return importlib.import_module("app.pipeline")

    def _wrapper(self, pipeline, chunks, outputs, preset_language=None):
        # Model the VAD interface exposed by the installed WhisperX wrapper.
        class NativeVad(pipeline.Vad):
            def preprocess_audio(self, audio):
                self.preprocessed = audio
                return "waveform"

            def __call__(self, value):
                self.called_with = value
                return "vad-output"

            def merge_chunks(self, value, size, onset, offset):
                self.merged_with = (value, size, onset, offset)
                return chunks

        vad = NativeVad()
        native = Mock()
        native.transcribe.side_effect = outputs
        # Return one cached wrapper with one native model target.
        return types.SimpleNamespace(
            vad_model=vad,
            model=native,
            options=types.SimpleNamespace(hotwords="saved-hotwords", initial_prompt="saved-prompt"),
            preset_language=preset_language,
            tokenizer=types.SimpleNamespace(language_code=None),
            detect_language=Mock(return_value="ar"),
        )

    def _worker(self, action):
        errors = []

        # Capture failures so background assertions reach the test thread.
        def run():
            try:
                action()
            except BaseException as error:
                errors.append(error)

        # Bound cleanup even when a regression leaves a worker blocked.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        return thread, errors

    def test_waiting_leases_protect_cache_and_serialize_request_options(self):
        pipeline = self._pipeline(MAX_LOADED_MODELS="1")
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        wrapper = self._wrapper(pipeline, [], [])
        observed = []

        # Hold the first request inside the decoder until the second owns a lease.
        def decode(audio, **kwargs):
            observed.append((wrapper.options.hotwords, wrapper.options.initial_prompt))
            if len(observed) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            return {"segments": [], "language": "en"}

        # Use the production loader while observing the waiter's cache acquisition.
        wrapper.transcribe = decode
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        original_take = pipeline._take_cached

        def take(name, acquire):
            model = original_take(name, acquire)
            if pipeline._whisper_models_in_use.get(name) == 2:
                waiting.set()
            return model

        pipeline._take_cached = take
        self.addCleanup(release.set)
        first, first_errors = self._worker(lambda: pipeline.transcribe([], "whisper-tiny", hotwords="one", initial_prompt="first"))
        self.assertTrue(entered.wait(3))
        second, second_errors = self._worker(lambda: pipeline.transcribe([], "tiny", hotwords="two", initial_prompt="second"))
        self.assertTrue(waiting.wait(3))

        # Active and waiting leases survive both the cap and the idle sweep.
        pipeline.load_whisper_model("base")
        self.assertIn("tiny", pipeline._whisper_models)
        pipeline.MODEL_KEEP_ALIVE_SECONDS = 1
        pipeline._whisper_models_last_used.update(tiny=0, base=0)
        pipeline._evict_from_cache(pipeline._whisper_models, pipeline._whisper_models_last_used, "model", 10, in_use=pipeline._whisper_models_in_use, lock=pipeline._whisper_state_lock)
        self.assertIn("tiny", pipeline._whisper_models)
        self.assertNotIn("base", pipeline._whisper_models)
        self.assertEqual(observed, [("one", "first")])

        # Finish both requests and check restored options and canonical cache keys.
        release.set()
        first.join(3)
        second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(first_errors + second_errors, [])
        self.assertEqual(observed, [("one", "first"), ("two", "second")])
        self.assertEqual((wrapper.options.hotwords, wrapper.options.initial_prompt), ("saved-hotwords", "saved-prompt"))
        self.assertEqual(pipeline._whisper_models_in_use, {})
        self.assertNotIn("whisper-tiny", pipeline._whisper_models)

    def test_cached_lease_does_not_wait_for_unrelated_loading(self):
        pipeline = self._pipeline()
        entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
        wrapper = self._wrapper(pipeline, [], [])

        # Block only the unrelated model's slow loader.
        def load(name, **kwargs):
            if name == "base":
                entered.set()
                self.assertTrue(release.wait(3))
            return wrapper

        sys.modules["whisperx"].load_model = load
        pipeline.load_whisper_model("tiny")
        self.addCleanup(release.set)
        loader, load_errors = self._worker(lambda: pipeline.load_whisper_model("base"))
        self.assertTrue(entered.wait(3))

        # Acquire through the real cache fast path while the load lock is held.
        def cached():
            with pipeline.use_whisper_model("tiny"):
                acquired.set()

        reader, read_errors = self._worker(cached)
        self.assertTrue(acquired.wait(3))
        release.set()
        loader.join(3)
        reader.join(3)
        self.assertEqual(load_errors + read_errors, [])
        self.assertEqual(pipeline._whisper_models_in_use, {})

    def test_failures_release_leases_and_restore_batched_options(self):
        pipeline = self._pipeline()
        wrapper = self._wrapper(pipeline, [], [])
        wrapper.transcribe = Mock(side_effect=RuntimeError("decode failed"))

        # A decode failure must restore original options and release its lease.
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        with self.assertRaises(RuntimeError):
            pipeline.transcribe([], "tiny", hotwords="request", initial_prompt="request")
        self.assertEqual(pipeline._whisper_models_in_use, {})
        self.assertEqual((wrapper.options.hotwords, wrapper.options.initial_prompt), ("saved-hotwords", "saved-prompt"))

        # A failed load must never publish a model or an in-use count.
        sys.modules["whisperx"].load_model = Mock(side_effect=RuntimeError("load failed"))
        with self.assertRaises(RuntimeError):
            pipeline.transcribe([], "base")
        self.assertEqual(pipeline._whisper_models_in_use, {})
        self.assertNotIn("base", pipeline._whisper_models)

    def test_native_lazy_generator_runs_inside_decode_lock_and_lease(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(pipeline, [{"start": 0, "end": 1}], [])

        # Check the lazy execution boundary rather than only generator creation.
        def segments():
            self.assertTrue(pipeline._transcription_lock.locked())
            self.assertEqual(pipeline._whisper_models_in_use, {"tiny": 1})
            yield Segment(0, 1, " speech")

        wrapper.model.transcribe = Mock(return_value=(segments(), None))
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        pipeline.transcribe([0] * 10, "tiny")
        self.assertEqual(pipeline._whisper_models_in_use, {})

    def test_default_mode_is_batched_and_preserves_batched_result_and_options(self):
        pipeline = self._pipeline()
        wrapper = self._wrapper(pipeline, [], [])
        expected = {"segments": [{"start": 0, "end": 1, "text": " kept"}], "language": "en"}
        wrapper.transcribe = Mock(return_value=expected)
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        result = pipeline.transcribe(
            [0] * 10,
            hotwords="request-word",
            initial_prompt="request-prompt",
        )

        self.assertEqual(pipeline.WHISPER_DECODE_MODE, "batched")
        self.assertIs(result, expected)
        wrapper.transcribe.assert_called_once_with(
            [0] * 10,
            batch_size=2,
            language=None,
            task="transcribe",
        )
        self.assertEqual(wrapper.options.hotwords, "saved-hotwords")
        self.assertEqual(wrapper.options.initial_prompt, "saved-prompt")

    def test_invalid_mode_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "WHISPER_DECODE_MODE must be either"):
            self._pipeline(WHISPER_DECODE_MODE="invalid")

    def test_silero_batched_dispatch_preserves_settings_and_options(self):
        # Select Silero without changing the established batched decoder path.
        pipeline = self._pipeline(VAD_METHOD="silero", VAD_CHUNK_SIZE="20")
        wrapper = self._wrapper(pipeline, [], [])
        expected = {"segments": [], "language": "en"}
        wrapper.transcribe = Mock(return_value=expected)
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        # Dispatch a request through the shared loader and restore request options.
        result = pipeline.transcribe([0] * 10, "tiny", language="en", hotwords="term")
        self.assertIs(result, expected)
        wrapper.transcribe.assert_called_once_with(
            [0] * 10, batch_size=2, language="en", task="transcribe"
        )
        wrapper.model.transcribe.assert_not_called()
        self.assertEqual(wrapper.options.hotwords, "saved-hotwords")
        self.assertEqual(sys.modules["whisperx"].load_model.call_args.kwargs["vad_method"], "silero")
        self.assertEqual(
            sys.modules["whisperx"].load_model.call_args.kwargs["vad_options"],
            {"chunk_size": 20, "vad_onset": .5, "vad_offset": .363},
        )

    def test_native_uses_vad_chunks_and_flattens_clamped_monotonic_segments(self):
        # Arrange two merged VAD chunks and multiple local native segments.
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native", VAD_CHUNK_SIZE="20")
        wrapper = self._wrapper(
            pipeline,
            [
                {"start": -1, "end": 1.2, "segments": []},
                {"start": 1.2, "end": 3, "segments": []},
            ],
            [
                (
                    iter([Segment(-1, .8, " first"), Segment(.8, 3, " second")]),
                    object(),
                ),
                (iter([Segment(0, .5, " third")]), object()),
            ],
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        pipeline.clear_gpu_memory = Mock()
        native_model = wrapper.model

        # Act through the public transcription dispatcher.
        result = pipeline.transcribe(
            [0] * 25,
            language="ar",
            task="translate",
            hotwords="term",
            initial_prompt="prompt",
        )

        # Assert flattened timestamps, shared model use, and one final cleanup.
        self.assertEqual(result["language"], "ar")
        self.assertEqual(result["segments"], [
            {"start": 0.0, "end": 0.8, "text": " first"},
            {"start": 0.8, "end": 1.2, "text": " second"},
            {"start": 1.2, "end": 1.7, "text": " third"},
        ])
        self.assertEqual(
            wrapper.vad_model.called_with,
            {"waveform": "waveform", "sample_rate": 10},
        )
        self.assertEqual(wrapper.vad_model.merged_with, ("vad-output", 20, .5, .363))
        self.assertEqual(wrapper.model.transcribe.call_count, 2)
        self.assertIs(wrapper.model, native_model)
        pipeline.clear_gpu_memory.assert_called_once_with()
        first_kwargs = wrapper.model.transcribe.call_args_list[0].kwargs
        second_kwargs = wrapper.model.transcribe.call_args_list[1].kwargs
        self.assertEqual(
            first_kwargs,
            {
                "language": "ar",
                "task": "translate",
                "hotwords": "term",
                "initial_prompt": "prompt",
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "beam_size": 5,
                "best_of": 5,
                "vad_filter": False,
                "word_timestamps": False,
            },
        )
        self.assertEqual(second_kwargs["language"], "ar")
        self.assertEqual(second_kwargs["task"], "translate")
        self.assertEqual(second_kwargs["hotwords"], "term")
        self.assertIsNone(second_kwargs["initial_prompt"])

    def test_readme_egyptian_recipe_reaches_native_decoder_without_forcing_language(self):
        # Apply the copyable README environment just as an opt-in deployment does.
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        section = readme.split("### Egyptian Turbo (opt-in)", 1)[1]
        block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
        environment = dict(re.findall(r"^([A-Z_]+)=(.*)$", block, re.M))
        pipeline = self._pipeline(**environment)
        wrapper = self._wrapper(pipeline, [{"start": 0, "end": 1}], [(iter([]), None), (iter([]), None)])
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        # Substitute preparation only; observe the real loader and decode boundaries.
        with patch.object(pipeline, "prepare_egyptian_model", return_value="/prepared/egyptian"):
            result = pipeline.transcribe([0] * 10)
            pipeline.transcribe([0] * 10, language="ar", initial_prompt="client context", hotwords="term")
        wrapper.detect_language.assert_called_once_with([0] * 10)
        self.assertEqual(result, {"segments": [], "language": "ar"})
        loaded = sys.modules["whisperx"].load_model.call_args
        self.assertEqual(loaded.args[0], "/prepared/egyptian")
        self.assertEqual(loaded.kwargs["device"], "cuda")
        self.assertEqual(loaded.kwargs["compute_type"], "int8_float16")
        self.assertEqual(loaded.kwargs["vad_method"], "silero")

        # Recipe decoding controls coexist with automatic language and explicit context.
        first, explicit = wrapper.model.transcribe.call_args_list
        self.assertIsNone(first.kwargs["initial_prompt"])
        self.assertEqual(explicit.kwargs["initial_prompt"], "client context")
        self.assertEqual(explicit.kwargs["hotwords"], "term")
        for call in (first, explicit):
            self.assertEqual(call.kwargs["language"], "ar")
            self.assertEqual(call.kwargs["beam_size"], 8)
            self.assertEqual(call.kwargs["best_of"], 5)
            self.assertEqual(call.kwargs["temperature"], [0, .2, .4, .6, .8, 1])
            self.assertFalse(call.kwargs["condition_on_previous_text"])
            self.assertEqual(call.kwargs["no_speech_threshold"], .8)
            self.assertEqual(call.kwargs["log_prob_threshold"], -.8)
            self.assertEqual(call.kwargs["compression_ratio_threshold"], 2.4)
        self.assertEqual(wrapper.vad_model.merged_with[1:3], (20, .5))

    def test_native_passes_configured_decoder_thresholds_directly(self):
        # Configure the native decoder thresholds at process startup.
        pipeline = self._pipeline(
            WHISPER_DECODE_MODE="native",
            VAD_METHOD=" SiLeRo ",
            VAD_CHUNK_SIZE="20",
            VAD_ONSET="0.7",
            VAD_OFFSET="0.4",
            NO_SPEECH_THRESHOLD="0.2",
            COMPRESSION_RATIO_THRESHOLD="3.3",
            LOG_PROB_THRESHOLD="-0.2",
        )
        wrapper = self._wrapper(
            pipeline,
            [{"start": 0, "end": 1, "segments": []}],
            [(iter([]), object())],
            preset_language="en",
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        # Decode one VAD chunk through the public native dispatcher.
        pipeline.transcribe([0] * 10)

        # Confirm native decoding receives the configured threshold values directly.
        self.assertIsInstance(wrapper.vad_model, pipeline.Vad)
        self.assertEqual(wrapper.vad_model.preprocessed, [0] * 10)
        self.assertEqual(wrapper.vad_model.called_with, {"waveform": "waveform", "sample_rate": 10})
        self.assertEqual(wrapper.vad_model.merged_with, ("vad-output", 20, .7, .4))
        self.assertEqual(sys.modules["whisperx"].load_model.call_args.kwargs["vad_method"], "silero")
        self.assertEqual(
            sys.modules["whisperx"].load_model.call_args.kwargs["vad_options"],
            {"chunk_size": 20, "vad_onset": .7, "vad_offset": .4},
        )
        self.assertEqual(
            {
                key: wrapper.model.transcribe.call_args.kwargs[key]
                for key in (
                    "no_speech_threshold",
                    "compression_ratio_threshold",
                    "log_prob_threshold",
                )
            },
            {
                "no_speech_threshold": 0.2,
                "compression_ratio_threshold": 3.3,
                "log_prob_threshold": -0.2,
            },
        )

    def test_native_detects_once_and_reuses_language_for_each_chunk(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(
            pipeline,
            [
                {"start": 0, "end": 1, "segments": []},
                {"start": 1, "end": 2, "segments": []},
            ],
            [(iter([]), object()), (iter([]), object())],
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        result = pipeline.transcribe([0] * 20)

        self.assertEqual(result, {"segments": [], "language": "ar"})
        wrapper.detect_language.assert_called_once_with([0] * 20)
        self.assertEqual(
            [call.kwargs["language"] for call in wrapper.model.transcribe.call_args_list],
            ["ar", "ar"],
        )

    def test_native_uses_tokenizer_language_code_without_detection(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(
            pipeline,
            [{"start": 0, "end": 1, "segments": []}],
            [(iter([Segment(0, .5, " English stays English")]), object())],
        )
        wrapper.tokenizer.language_code = "ar"
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        result = pipeline.transcribe([0] * 10)

        self.assertEqual(result["language"], "ar")
        self.assertEqual(result["segments"][0]["text"], " English stays English")
        wrapper.detect_language.assert_not_called()
        self.assertEqual(wrapper.model.transcribe.call_args.kwargs["language"], "ar")

    def test_native_rejects_invalid_vad_chunks_before_decoder(self):
        cases = (
            {"start": 0, "end": 0, "segments": []},
            {"start": 2, "end": 1, "segments": []},
            {"start": float("nan"), "end": 1, "segments": []},
        )
        for chunk in cases:
            with self.subTest(chunk=chunk):
                pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
                wrapper = self._wrapper(pipeline, [chunk], [])
                sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

                with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(
                    pipeline.NativeDecodeError
                ) as raised:
                    pipeline.transcribe([0] * 20)

                wrapper.model.transcribe.assert_not_called()
                self.assertNotIn("nan", str(raised.exception))
                self.assertIn("exception=ValueError", str(raised.exception))
                self.assertNotIn("nan", "\n".join(logs.output).lower())

    def test_native_vad_receives_16000_sample_rate_and_skips_silent_audio(self):
        # Model the VAD interface without invoking an actual Silero model.
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native", VAD_METHOD="silero")
        wrapper = self._wrapper(pipeline, [], [], preset_language="en")
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        audio = [0] * 16000

        # Use the deployed sample rate rather than the fixture's shorter test scale.
        with patch.object(pipeline, "SAMPLE_RATE", 16000):
            result = pipeline.transcribe(audio)

        # Empty VAD output must preserve the supplied waveform and skip decoding.
        self.assertIs(wrapper.vad_model.preprocessed, audio)
        self.assertEqual(
            wrapper.vad_model.called_with,
            {"waveform": "waveform", "sample_rate": 16000},
        )
        self.assertEqual(result, {"segments": [], "language": "en"})
        wrapper.model.transcribe.assert_not_called()

    def test_native_uses_pyannote_vad_fallback_flow(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        preprocessor = Mock(return_value="pyannote-waveform")
        merger = Mock(return_value=[])
        pipeline.Pyannote = types.SimpleNamespace(
            preprocess_audio=preprocessor,
            merge_chunks=merger,
        )
        vad = Mock(return_value="pyannote-vad-output")
        wrapper = types.SimpleNamespace(
            vad_model=vad,
            model=Mock(),
            options=types.SimpleNamespace(hotwords=None, initial_prompt=None),
            preset_language="en",
            tokenizer=types.SimpleNamespace(language_code=None),
            detect_language=Mock(),
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        self.assertEqual(pipeline.transcribe([0] * 10), {"segments": [], "language": "en"})
        preprocessor.assert_called_once_with([0] * 10)
        vad.assert_called_once_with({"waveform": "pyannote-waveform", "sample_rate": 10})
        merger.assert_called_once_with("pyannote-vad-output", 30, onset=.5, offset=.363)

    def test_native_exhausts_generator_and_logs_no_transcript_on_failure(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        consumed = []

        def generator():
            consumed.append("done")
            yield Segment(0, .5, "secret transcript")

        wrapper = self._wrapper(
            pipeline,
            [
                {"start": 0, "end": 1, "segments": []},
                {"start": 1, "end": 2, "segments": []},
            ],
            [(generator(), object()), RuntimeError("secret transcript")],
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(
            pipeline.NativeDecodeError
        ) as raised:
            pipeline.transcribe([0] * 20)

        self.assertEqual(consumed, ["done"])
        self.assertNotIn("secret transcript", "\n".join(logs.output))
        self.assertNotIn("secret transcript", str(raised.exception))
        self.assertIn("stage=decoder_creation chunk_index=1 segment_index=-1 reason=decoder_error", str(raised.exception))
        metrics = sys.modules["app.metrics"]
        self.assertEqual(metrics.WHISPERX_DECODE_CHUNKS_TOTAL.count, 2)
        self.assertEqual(metrics.WHISPERX_DECODE_FAILURES_TOTAL.count, 1)
        self.assertEqual(metrics.WHISPERX_DECODE_FAILURES_TOTAL.labels_seen, [{"mode": "native"}])

    def test_native_normalization_error_is_sanitized_and_counted_once(self):
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(
            pipeline,
            [{"start": 0, "end": 1, "segments": []}],
            [(iter([Segment("bad transcript", .5, "hidden text")]), object())],
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)

        with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(
            pipeline.NativeDecodeError
        ) as raised:
            pipeline.transcribe([0] * 10)

        self.assertNotIn("hidden text", "\n".join(logs.output))
        self.assertNotIn("bad transcript", str(raised.exception))
        self.assertIn("exception=ValueError", str(raised.exception))
        metrics = sys.modules["app.metrics"]
        self.assertEqual(metrics.WHISPERX_DECODE_CHUNKS_TOTAL.count, 1)
        self.assertEqual(metrics.WHISPERX_DECODE_FAILURES_TOTAL.count, 1)

    def test_decoder_controls_validate_at_startup(self):
        # Reject invalid values even when batched decoding is selected.
        cases = {
            "WHISPER_BEAM_SIZE": ("0", "-1", "1.5", "bad"),
            "WHISPER_NATIVE_BEST_OF": ("0", "-2", "nan", "2.0"),
            "WHISPER_NATIVE_TEMPERATURES": (",", "0,", ",0", "0,,1", "nan", "inf", "-inf", "-0.1", "1.1", "bad"),
            "WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT": ("maybe", "2", "truthy"),
            "WHISPER_NATIVE_PROMPT_SCOPE": ("last", "first,all"),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self._pipeline(**{name: value})

    def test_decoder_controls_defaults_and_blank_values(self):
        # Unset, empty, and whitespace values share the same startup defaults.
        names = (
            "WHISPER_BEAM_SIZE", "WHISPER_NATIVE_BEST_OF", "WHISPER_NATIVE_TEMPERATURES",
            "WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT", "WHISPER_NATIVE_PROMPT_SCOPE",
        )
        for environment in ({}, dict.fromkeys(names, ""), dict.fromkeys(names, "  ")):
            with self.subTest(environment=environment):
                pipeline = self._pipeline(**environment)
                self.assertEqual(pipeline.WHISPER_BEAM_SIZE, 5)
                self.assertEqual(pipeline.WHISPER_NATIVE_BEST_OF, 5)
                self.assertEqual(pipeline.WHISPER_NATIVE_TEMPERATURES, (0, .2, .4, .6, .8, 1))
                self.assertFalse(pipeline.WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT)
                self.assertEqual(pipeline.WHISPER_NATIVE_PROMPT_SCOPE, "first")

    def test_native_condition_boolean_spellings(self):
        # Accept only established boolean spellings after trimming and folding case.
        for value in ("true", "1", "yes", "on", " TRUE ", "false", "0", "no", "off", " OFF "):
            with self.subTest(value=value):
                pipeline = self._pipeline(WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT=value)
                self.assertEqual(pipeline.WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT, value.strip().lower() in ("true", "1", "yes", "on"))

    def test_native_overrides_prompt_scope_and_request_isolation(self):
        # Decode two requests on one cached model without carrying decoded history.
        for scope in ("first", " ALL "):
            with self.subTest(scope=scope):
                pipeline = self._pipeline(
                    WHISPER_DECODE_MODE="native", WHISPER_BEAM_SIZE="3",
                    WHISPER_NATIVE_BEST_OF="2", WHISPER_NATIVE_TEMPERATURES=" 0, .7, 1 ",
                    WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT="yes", WHISPER_NATIVE_PROMPT_SCOPE=scope,
                )
                wrapper = self._wrapper(pipeline, [{"start": 0, "end": 1}, {"start": 1, "end": 2}], [
                    (iter([Segment(0, .5, "decoded history")]), None), (iter([]), None),
                    (iter([]), None), (iter([]), None),
                ])
                setattr(sys.modules["whisperx"], "load_model", Mock(return_value=wrapper))

                # Environment changes after import must not alter effective controls.
                with patch.dict(os.environ, {"WHISPER_BEAM_SIZE": "99", "WHISPER_NATIVE_PROMPT_SCOPE": "invalid"}):
                    pipeline.transcribe([0] * 20, "tiny", initial_prompt="first-request", hotwords="one")
                    pipeline.transcribe([0] * 20, "tiny", initial_prompt="second-request", hotwords="two")

                # Repeat only the request prompt, never prior decoded text.
                calls = wrapper.model.transcribe.call_args_list
                self.assertEqual([call.kwargs["initial_prompt"] for call in calls], [
                    "first-request", "first-request" if scope.strip().lower() == "all" else None,
                    "second-request", "second-request" if scope.strip().lower() == "all" else None,
                ])
                self.assertEqual([call.kwargs["hotwords"] for call in calls], ["one", "one", "two", "two"])
                for call in calls:
                    self.assertEqual(call.kwargs["beam_size"], 3)
                    self.assertEqual(call.kwargs["best_of"], 2)
                    self.assertEqual(call.kwargs["temperature"], [0, .7, 1])
                    self.assertTrue(call.kwargs["condition_on_previous_text"])
                self.assertEqual((wrapper.options.hotwords, wrapper.options.initial_prompt), ("saved-hotwords", "saved-prompt"))
                self.assertEqual(pipeline._whisper_models_in_use, {})
                self.assertEqual(sys.modules["whisperx"].load_model.call_count, 1)

    def test_shared_beam_is_only_new_batched_constructor_option(self):
        # Native-only settings are validated but never forwarded to WhisperX construction.
        pipeline = self._pipeline(WHISPER_BEAM_SIZE="7", WHISPER_NATIVE_TEMPERATURES="0", WHISPER_NATIVE_BEST_OF="2", WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT="true", WHISPER_NATIVE_PROMPT_SCOPE="all")
        pipeline.load_whisper_model("tiny")
        self.assertEqual(sys.modules["whisperx"].load_model.call_args.kwargs["asr_options"], {
            "beam_size": 7, "no_speech_threshold": .6, "compression_ratio_threshold": 2.4, "log_prob_threshold": -1,
        })
        self.assertEqual(pipeline.WHISPER_NATIVE_TEMPERATURES, (0,))

    def test_single_zero_temperature_and_effective_configuration_log(self):
        # Configure a zero-temperature decode without changing compute settings.
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native", WHISPER_BEAM_SIZE="1", WHISPER_NATIVE_BEST_OF="3", WHISPER_NATIVE_TEMPERATURES="0", WHISPER_NATIVE_PROMPT_SCOPE="all")
        wrapper = self._wrapper(pipeline, [{"start": 0, "end": 1}], [(iter([]), None)])
        setattr(sys.modules["whisperx"], "load_model", Mock(return_value=wrapper))

        # The once-only summary must omit all request prompt and hotword contents.
        with self.assertLogs("app.pipeline", "INFO") as logs:
            pipeline.transcribe([0] * 10, "tiny", initial_prompt="secret prompt", hotwords="secret hotwords")
            pipeline.log_decoder_configuration_once()
        summaries = [line for line in logs.output if "Whisper decoder configuration" in line]
        self.assertEqual(len(summaries), 1)
        self.assertIn("beam_size=1 read_once=true native_only_temperatures=(0.0,) native_only_best_of=3", summaries[0])
        self.assertIn("native_only_condition_on_previous_text=False native_only_prompt_scope=all", summaries[0])
        self.assertNotIn("secret", "".join(logs.output))
        self.assertEqual(wrapper.model.transcribe.call_args.kwargs["temperature"], [0])
        self.assertEqual(sys.modules["whisperx"].load_model.call_args.kwargs["compute_type"], "int8")

    def test_native_failures_have_controlled_stage_and_reason(self):
        # Model a generator that fails after producing speech without returning partial output.
        def late_failure():
            yield Segment(0, .2, "secret decoded text")
            raise ValueError("secret dependency failure")

        # Keep all timestamp behavior fail-closed, including mixed valid and invalid segments.
        cases = (
            ({"start": "secret chunk", "end": 1}, [], "chunk_validation", "invalid_chunk", -1),
            ({"start": 0, "end": 1}, [ValueError("secret immediate failure")], "decoder_creation", "decoder_error", -1),
            ({"start": 0, "end": 1}, [(late_failure(), None)], "lazy_iteration", "iteration_error", 1),
            ({"start": 0, "end": 1}, [(iter([Segment("secret timestamp", .5, "secret text")]), None)], "timestamp_validation", "malformed_timestamp", 0),
            ({"start": 0, "end": 1}, [(iter([Segment(0, float("nan"), "secret text")]), None)], "timestamp_validation", "nonfinite_timestamp", 0),
            ({"start": 0, "end": 1}, [(iter([Segment(0, float("inf"), "secret text")]), None)], "timestamp_validation", "nonfinite_timestamp", 0),
            ({"start": 0, "end": 1}, [(iter([Segment(.5, .2, "secret text")]), None)], "timestamp_validation", "reversed_timestamp", 0),
            ({"start": 0, "end": 1}, [(iter([Segment(2, 3, "secret text")]), None)], "timestamp_clamping", "collapsed_timestamp", 0),
            ({"start": 0, "end": 1}, [(iter([Segment(0, .5, "secret valid"), Segment(.1, .2, "secret invalid")]), None)], "timestamp_clamping", "collapsed_timestamp", 1),
        )
        for chunk, outputs, stage, reason, index in cases:
            with self.subTest(stage=stage, reason=reason, index=index):
                pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
                setattr(sys.modules["app.metrics"], "WHISPERX_DECODE_FAILURES_TOTAL", Counter())
                wrapper = self._wrapper(pipeline, [chunk], outputs)
                setattr(sys.modules["whisperx"], "load_model", Mock(return_value=wrapper))
                with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(pipeline.NativeDecodeError) as raised:
                    pipeline.transcribe([0] * 10, initial_prompt="secret prompt", hotwords="secret hotwords")

                # Diagnostics contain only bounded stages, indices, reasons, and exception classes.
                message = str(raised.exception)
                self.assertIn(f"stage={stage} chunk_index=0 segment_index={index} reason={reason}", message)
                self.assertNotIn("secret", message + "".join(logs.output))
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertEqual(sys.modules["app.metrics"].WHISPERX_DECODE_FAILURES_TOTAL.labels_seen, [{"mode": "native"}])
                self.assertEqual(pipeline._whisper_models_in_use, {})

    def test_transcription_refreshes_the_current_cached_wrapper_timestamp(self):
        pipeline = self._pipeline()
        wrapper = self._wrapper(pipeline, [], [])
        wrapper.transcribe = Mock(return_value={"segments": [], "language": "en"})
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        pipeline._whisper_models["small"] = wrapper
        pipeline._whisper_models_last_used["small"] = 0

        with patch.object(pipeline.time, "time", return_value=123.0):
            pipeline.transcribe([0] * 10, model_name="small")

        self.assertEqual(pipeline._whisper_models_last_used["small"], 123.0)

    def test_native_result_flows_unchanged_through_alignment_and_diarization(self):
        # Arrange empty and blank native outputs without loading real models.
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(pipeline, [], [])
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        self.assertEqual(pipeline.transcribe([0] * 10), {"segments": [], "language": "ar"})

        # Start the next scenario with an empty cache so the real loader sees its wrapper.
        pipeline._whisper_models.clear()
        wrapper = self._wrapper(
            pipeline,
            [{"start": 0, "end": 1, "segments": []}],
            [(iter([Segment(0, .5, "   ")]), object())],
        )
        sys.modules["whisperx"].load_model = Mock(return_value=wrapper)
        self.assertEqual(pipeline.transcribe([0] * 10)["segments"], [])

        # Arrange a native-shaped result and distinct downstream stage outputs.
        native_result = {
            "segments": [{"start": 0, "end": 1, "text": " x"}],
            "language": "en",
        }
        aligned_result = {
            "segments": native_result["segments"],
            "language": "en",
            "word_segments": [{"word": "x", "start": 0, "end": 1}],
        }
        diarized_result = {
            "segments": [{"start": 0, "end": 1, "text": " x", "speaker": "S0"}],
            "language": "en",
        }
        pipeline.transcribe = Mock(return_value=native_result)
        pipeline.align = Mock(return_value=aligned_result)
        pipeline.diarize = Mock(return_value=(diarized_result, {"S0": [1.0]}))

        # Act with both downstream stages enabled.
        result, embeddings = pipeline.run_pipeline(
            [0] * 10,
            should_diarize=True,
            return_speaker_embeddings=True,
        )

        # Assert each stage receives its predecessor's exact result shape.
        self.assertIs(pipeline.align.call_args.args[1], native_result)
        self.assertIs(pipeline.diarize.call_args.args[1], aligned_result)
        self.assertEqual(result, diarized_result)
        self.assertEqual(embeddings, {"S0": [1.0]})


if __name__ == "__main__":
    unittest.main()
