"""Explicit VAD interfaces for import-only stand-in compatibility."""


class Vad:
    """Base type used to select the service's VAD path."""


class Pyannote(Vad):
    """Native decoding requires a test-specific VAD implementation."""

    @staticmethod
    def preprocess_audio(audio):
        raise NotImplementedError("The stub server does not implement native VAD.")

    @staticmethod
    def merge_chunks(*args, **kwargs):
        raise NotImplementedError("The stub server does not implement native VAD.")
