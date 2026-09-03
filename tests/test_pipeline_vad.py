"""Unit tests for process-wide WhisperX ASR VAD configuration."""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

_MISSING = object()


class TestPipelineVadConfiguration(unittest.TestCase):
    """Load the pipeline with mocked ML dependencies and isolated environments."""

    def setUp(self):
        # Preserve the app package attribute as well as its registered submodule.
        self.original_app_pipeline_attribute = getattr(
            sys.modules.get("app"), "pipeline", _MISSING
        )

        # Remove prior imports so each test gets a fresh process-wide configuration.
        self.original_modules = {
            name: sys.modules.pop(name)
            for name in ("app.pipeline", "numpy", "torch", "whisperx", "whisperx.audio", "whisperx.diarize", "whisperx.vads")
            if name in sys.modules
        }

        # Use a minimal environment so host settings cannot affect assertions.
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()
        self._install_dependency_stubs()

    def tearDown(self):
        # Restore imports and environment state for unrelated tests.
        self.env_patcher.stop()
        for name in ("app.pipeline", "numpy", "torch", "whisperx", "whisperx.audio", "whisperx.diarize", "whisperx.vads"):
            sys.modules.pop(name, None)
        sys.modules.update(self.original_modules)

        # Restore the package attribute that Python sets when importing a submodule.
        app_package = sys.modules.get("app")
        if app_package is not None:
            if self.original_app_pipeline_attribute is _MISSING:
                app_package.__dict__.pop("pipeline", None)
            else:
                setattr(app_package, "pipeline", self.original_app_pipeline_attribute)

    def _install_dependency_stubs(self):
        """Install only the dependency interfaces pipeline import needs."""
        numpy = types.ModuleType("numpy")
        setattr(numpy, "ndarray", object)
        setattr(numpy, "floating", float)
        setattr(numpy, "integer", int)

        # Mimic CPU-only Torch without initializing CUDA.
        torch = types.ModuleType("torch")
        setattr(torch, "cuda", types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
        setattr(torch, "device", lambda value: value)

        # Provide mock WhisperX loading and diarization interfaces without downloads.
        whisperx = types.ModuleType("whisperx")
        whisperx.__path__ = []
        setattr(whisperx, "load_model", Mock(return_value=object()))
        setattr(whisperx, "load_align_model", Mock())
        setattr(whisperx, "align", Mock())
        setattr(whisperx, "assign_word_speakers", Mock(side_effect=lambda _, result, **__: result))

        diarize = types.ModuleType("whisperx.diarize")
        setattr(diarize, "DiarizationPipeline", Mock())
        audio = types.ModuleType("whisperx.audio")
        setattr(audio, "SAMPLE_RATE", 16000)
        vads = types.ModuleType("whisperx.vads")
        setattr(vads, "Vad", object)
        setattr(vads, "Pyannote", object)
        sys.modules.update({
            "numpy": numpy,
            "torch": torch,
            "whisperx": whisperx,
            "whisperx.audio": audio,
            "whisperx.diarize": diarize,
            "whisperx.vads": vads,
        })

    def _load_pipeline(self, **environment):
        """Import a fresh pipeline module after setting its startup environment."""
        # Rebuild the module while only the requested startup values are visible.
        sys.modules.pop("app.pipeline", None)
        with patch.dict(os.environ, environment, clear=True):
            return importlib.import_module("app.pipeline")

    def test_vad_defaults_are_passed_when_model_is_constructed(self):
        pipeline = self._load_pipeline()

        # Confirm absent variables preserve the existing application defaults.
        self.assertEqual(pipeline.VAD_CHUNK_SIZE, 30)
        self.assertEqual(pipeline.VAD_ONSET, 0.500)
        self.assertEqual(pipeline.VAD_OFFSET, 0.363)

        # Construct a model to verify defaults reach WhisperX unchanged.
        with self.assertLogs("app.pipeline", "INFO") as logs:
            pipeline.load_whisper_model("small")

        # Confirm WhisperX receives the defaults and the effective values are logged.
        pipeline.whisperx.load_model.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
            download_root="/.cache",
            vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
        )
        self.assertIn(
            "INFO:app.pipeline:ASR VAD configuration: chunk_size=30 onset=0.500 offset=0.363",
            logs.output,
        )
        self.assertIn("INFO:app.pipeline:Whisper decode mode: batched", logs.output)

    def test_custom_vad_environment_is_passed_when_cached_model_is_constructed(self):
        pipeline = self._load_pipeline(
            VAD_CHUNK_SIZE="20", VAD_ONSET="0.700", VAD_OFFSET="0.400"
        )

        # Load twice to prove VAD options are supplied only at cache construction.
        with self.assertLogs("app.pipeline", "INFO") as logs:
            pipeline.load_whisper_model("small")
            pipeline.load_whisper_model("small")

        # Confirm custom values are passed and model construction is cached.
        pipeline.whisperx.load_model.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
            download_root="/.cache",
            vad_options={"chunk_size": 20, "vad_onset": 0.7, "vad_offset": 0.4},
        )
        self.assertEqual(
            [message for message in logs.output if "ASR VAD" in message],
            ["INFO:app.pipeline:ASR VAD configuration: chunk_size=20 onset=0.700 offset=0.400"],
        )

    def test_invalid_vad_values_fail_at_import(self):
        # Cover malformed values, bounds, non-finite floats, and threshold order.
        cases = (
            ({"VAD_CHUNK_SIZE": "nope"}, "VAD_CHUNK_SIZE must be an integer"),
            ({"VAD_CHUNK_SIZE": "61"}, "VAD_CHUNK_SIZE must be between 5 and 60"),
            ({"VAD_ONSET": "nope"}, "VAD_ONSET must be a float"),
            ({"VAD_OFFSET": "nope"}, "VAD_OFFSET must be a float"),
            ({"VAD_OFFSET": "nan"}, "VAD_OFFSET must be a finite float"),
            ({"VAD_ONSET": "0.3", "VAD_OFFSET": "0.4"}, "VAD thresholds must satisfy"),
        )

        # Import each case in a fresh environment and check its startup error.
        for environment, message in cases:
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, message):
                self._load_pipeline(**environment)

    def test_diarization_overrides_are_isolated_from_asr_vad(self):
        pipeline = self._load_pipeline(
            HF_TOKEN="test-token",
            DIARIZE_PARAM_OVERRIDES='{"clustering": {"Fb": 1.0}}',
        )

        # Prepare a Pyannote wrapper that exposes its existing parameter tree.
        pyannote = Mock()
        pyannote.parameters.return_value = {"clustering": {"Fb": 0.8}}
        wrapper = Mock(model=pyannote)
        pipeline.DiarizationPipeline.return_value = wrapper

        # Apply a Pyannote override without constructing a Whisper model.
        pipeline.load_diarize_pipeline()

        # Confirm only the diarization pipeline receives the Pyannote override.
        pyannote.instantiate.assert_called_once_with({"clustering": {"Fb": 1.0}})
        pipeline.DiarizationPipeline.assert_called_once_with(
            model_name="pyannote/speaker-diarization-community-1",
            use_auth_token="test-token",
            device="cpu",
        )
        pipeline.whisperx.load_model.assert_not_called()
        self.assertEqual((pipeline.VAD_CHUNK_SIZE, pipeline.VAD_ONSET, pipeline.VAD_OFFSET), (30, 0.5, 0.363))


if __name__ == "__main__":
    unittest.main()
