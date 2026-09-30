"""Stand-in for faster_whisper: the model list and name check of 1.2.1."""

from faster_whisper.utils import _MODELS, download_model  # noqa: F401

__version__ = "1.2.1+stub"


def available_models():
    return list(_MODELS)
