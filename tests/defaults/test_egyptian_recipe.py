"""Check the public, copyable opt-in deployment recipe."""

import re
from pathlib import Path


def test_readme_has_complete_egyptian_environment_recipe():
    # Limit the check to the opt-in section, not unrelated deployment examples.
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    section = re.search(
        r"^### Egyptian Turbo \(opt-in\)\s*\n(.*?)(?=^#{1,3} |\Z)",
        readme,
        re.M | re.S,
    )
    assert section, "README needs the Egyptian Turbo (opt-in) section"

    # One copyable block must contain every setting needed for the selected case.
    expected = {
        "ASR_BACKEND": "whisper",
        "DEFAULT_MODEL": "mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching",
        "PRELOAD_MODEL": "",
        "DEVICE": "cuda",
        "COMPUTE_TYPE": "int8_float16",
        "WHISPER_DECODE_MODE": "native",
        "VAD_METHOD": "silero",
        "VAD_CHUNK_SIZE": "20",
        "VAD_ONSET": "0.5",
        "WHISPER_BEAM_SIZE": "8",
        "WHISPER_NATIVE_BEST_OF": "5",
        "WHISPER_NATIVE_TEMPERATURES": "0.0,0.2,0.4,0.6,0.8,1.0",
        "WHISPER_NATIVE_CONDITION_ON_PREVIOUS_TEXT": "false",
        "NO_SPEECH_THRESHOLD": "0.8",
        "LOG_PROB_THRESHOLD": "-0.8",
        "COMPRESSION_RATIO_THRESHOLD": "2.4",
        "CACHE_DIR": "/.cache",
    }
    blocks = re.findall(r"```bash\n(.*?)```", section.group(1), re.S)
    environments = [dict(re.findall(r"^([A-Z_]+)=(.*)$", block, re.M)) for block in blocks]
    assert any(expected.items() <= environment.items() for environment in environments), (
        "One bash environment block must contain the complete Egyptian recipe"
    )
