"""Model names and the name check of faster-whisper 1.2.1 (utils.download_model)."""

import os
import re

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def download_model(size_or_id, output_dir=None, local_files_only=False, cache_dir=None,
                   revision=None, use_auth_token=None):
    """Same acceptance rules as the real function; downloads nothing."""
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        repo_id = _MODELS.get(size_or_id)
        if repo_id is None:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s" % (size_or_id, ", ".join(_MODELS.keys()))
            )
    return os.path.join(output_dir or cache_dir or "/tmp/faster-whisper-stub", repo_id.replace("/", "--"))
