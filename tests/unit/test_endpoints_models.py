"""The /asr and OpenAI endpoints resolve models the same way.

Audio decoding and the pipeline are faked, so no model is loaded.
"""

import io
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_model_selection import MODEL_ENV, _drop_app_modules


@pytest.fixture
def client_for(monkeypatch):
    def _make(**env):
        for key in MODEL_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _drop_app_modules()
        import whisperx
        monkeypatch.setattr(whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
        import app.main as main
        import app.openai_compat as oc
        seen = {}

        def fake_pipeline(audio, model_name=None, **kwargs):
            # Capture public request options without loading a model.
            seen["asr"] = model_name
            seen["asr_options"] = kwargs
            return {"language": "en", "segments": [], "word_segments": []}, None

        def fake_transcribe_and_align(audio, model, *args, **kwargs):
            # Capture OpenAI options before the substitute returns its response.
            seen["openai"] = model
            seen["openai_args"] = args
            seen["openai_options"] = kwargs
            return {"language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

        monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(oc, "_run_transcribe_and_align", fake_transcribe_and_align)
        return TestClient(main.app), seen   # no `with`: the startup preload does not run
    yield _make
    _drop_app_modules()


def _audio():
    return {"audio_file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")}


def test_asr_without_a_model_uses_the_default_even_with_an_empty_preload(client_for):
    client, seen = client_for(PRELOAD_MODEL="")
    r = client.post("/asr", files=_audio())
    assert r.status_code == 200, r.text
    assert seen["asr"] == "large-v3"

    r = client.post("/asr?model=", files=_audio())
    assert r.status_code == 200 and seen["asr"] == "large-v3"


def test_asr_resolves_aliases_and_rejects_unknown_models(client_for):
    client, seen = client_for(PRELOAD_MODEL="large-v3")
    assert client.post("/asr?model=whisper-medium", files=_audio()).status_code == 200
    assert seen["asr"] == "medium"
    r = client.post("/asr?model=large-v4", files=_audio())
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


@pytest.mark.parametrize("model,expected", [
    ("turbo", "turbo"),
    ("distil-large-v3.5", "distil-large-v3.5"),
    ("medium.en", "medium.en"),
    ("whisper-1", "large-v3"),
])
def test_openai_endpoint_accepts_every_listed_model(client_for, model, expected):
    client, seen = client_for(PRELOAD_MODEL="")
    r = client.post("/v1/audio/transcriptions", data={"model": model},
                    files={"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")})
    assert r.status_code == 200, r.text
    assert seen["openai"] == expected


def test_openai_endpoint_rejects_unknown_models(client_for):
    client, _ = client_for()
    r = client.post("/v1/audio/transcriptions", data={"model": "large-v4"},
                    files={"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_models_list_matches_what_is_accepted(client_for):
    client, _ = client_for(ALLOWED_MODELS="turbo")
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["whisper-1", "large-v3", "turbo"]


@pytest.mark.parametrize("language", [None, "ar"])
def test_egyptian_asr_keeps_language_context_and_feature_requests(client_for, language):
    # Opt in through the existing model environment without selecting a language.
    name = "mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching"
    client, seen = client_for(DEFAULT_MODEL=name, PRELOAD_MODEL="")
    params = {"initial_prompt": "technical context", "hotwords": "Kubernetes",
              "word_timestamps": "true", "diarize": "true",
              "return_speaker_embeddings": "true", "num_speakers": "2"}
    if language is not None:
        params["language"] = language

    # Forward explicit context and feature controls while leaving detection automatic.
    response = client.post("/asr", params=params, files=_audio())
    assert response.status_code == 200, response.text
    assert seen["asr"] == name
    assert seen["asr_options"] == {
        "language": language, "task": "transcribe", "initial_prompt": "technical context",
        "hotwords": "Kubernetes", "word_timestamps": True, "should_diarize": True,
        "num_speakers": 2, "min_speakers": None, "max_speakers": None,
        "return_speaker_embeddings": True,
    }
    assert set(response.json()) == {"text", "language", "segments", "word_segments"}

    # An explicit existing model still overrides the Egyptian deployment default.
    response = client.post("/asr", params={"model": "large-v3", "diarize": "false"}, files=_audio())
    assert response.status_code == 200, response.text
    assert seen["asr"] == "large-v3"
    assert seen["asr_options"]["should_diarize"] is False
    assert seen["asr_options"]["initial_prompt"] is None


@pytest.mark.parametrize("model,expected", [("whisper-1", None), ("large-v3", "large-v3")])
def test_egyptian_openai_model_selection_keeps_existing_precedence(client_for, model, expected):
    # The OpenAI alias follows the opt-in default unless the request names a model.
    name = "mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching"
    client, seen = client_for(DEFAULT_MODEL=name, PRELOAD_MODEL="")
    response = client.post(
        "/v1/audio/transcriptions", data={"model": model, "language": "ar", "prompt": "context"},
        files={"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert seen["openai"] == (expected or name)
    assert seen["openai_args"][:2] == ("ar", "transcribe")
    assert seen["openai_options"]["hotwords"] == "context"
    assert response.json()["text"] == "hi"
