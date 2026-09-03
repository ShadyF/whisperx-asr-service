"""Focused unit tests for native faster-whisper decoding without ML dependencies."""

import importlib
import os
import sys
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

    def test_default_mode_is_batched_and_preserves_batched_result_and_options(self):
        pipeline = self._pipeline()
        wrapper = self._wrapper(pipeline, [], [])
        expected = {"segments": [{"start": 0, "end": 1, "text": " kept"}], "language": "en"}
        wrapper.transcribe = Mock(return_value=expected)
        pipeline.load_whisper_model = Mock(return_value=wrapper)

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
        pipeline.load_whisper_model = Mock(return_value=wrapper)
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
                "vad_filter": False,
                "word_timestamps": False,
            },
        )
        self.assertEqual(second_kwargs["language"], "ar")
        self.assertEqual(second_kwargs["task"], "translate")
        self.assertEqual(second_kwargs["hotwords"], "term")
        self.assertIsNone(second_kwargs["initial_prompt"])

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
        pipeline.load_whisper_model = Mock(return_value=wrapper)

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
        pipeline.load_whisper_model = Mock(return_value=wrapper)

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
                pipeline.load_whisper_model = Mock(return_value=wrapper)

                with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(
                    pipeline.NativeDecodeError
                ) as raised:
                    pipeline.transcribe([0] * 20)

                wrapper.model.transcribe.assert_not_called()
                self.assertNotIn("nan", str(raised.exception))
                self.assertIn("exception=ValueError", str(raised.exception))
                self.assertNotIn("nan", "\n".join(logs.output).lower())

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
        pipeline.load_whisper_model = Mock(return_value=wrapper)

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
        pipeline.load_whisper_model = Mock(return_value=wrapper)

        with self.assertLogs("app.pipeline", "ERROR") as logs, self.assertRaises(
            pipeline.NativeDecodeError
        ) as raised:
            pipeline.transcribe([0] * 20)

        self.assertEqual(consumed, ["done"])
        self.assertNotIn("secret transcript", "\n".join(logs.output))
        self.assertNotIn("secret transcript", str(raised.exception))
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
        pipeline.load_whisper_model = Mock(return_value=wrapper)

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

    def test_transcription_refreshes_the_current_cached_wrapper_timestamp(self):
        pipeline = self._pipeline()
        wrapper = self._wrapper(pipeline, [], [])
        wrapper.transcribe = Mock(return_value={"segments": [], "language": "en"})
        pipeline.load_whisper_model = Mock(return_value=wrapper)
        pipeline._whisper_models["small"] = wrapper
        pipeline._whisper_models_last_used["small"] = 0

        with patch.object(pipeline.time, "time", return_value=123.0):
            pipeline.transcribe([0] * 10, model_name="small")

        self.assertEqual(pipeline._whisper_models_last_used["small"], 123.0)

    def test_native_result_flows_unchanged_through_alignment_and_diarization(self):
        # Arrange empty and blank native outputs without loading real models.
        pipeline = self._pipeline(WHISPER_DECODE_MODE="native")
        wrapper = self._wrapper(pipeline, [], [])
        pipeline.load_whisper_model = Mock(return_value=wrapper)
        self.assertEqual(pipeline.transcribe([0] * 10), {"segments": [], "language": "ar"})

        wrapper = self._wrapper(
            pipeline,
            [{"start": 0, "end": 1, "segments": []}],
            [(iter([Segment(0, .5, "   ")]), object())],
        )
        pipeline.load_whisper_model = Mock(return_value=wrapper)
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
