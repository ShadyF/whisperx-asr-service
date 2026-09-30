"""End to end on stand-in ML packages: the documented setup answers requests.

The launcher (stub_server.py) is started exactly as the Speakr CI starts it.
The first request is the one Speakr's startup voice-embedding check sends: a
short WAV with no model, diarization on, one speaker, embeddings requested.
"""

import io
import math
import os
import socket
import struct
import subprocess
import sys
import time
import wave

import httpx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
MINIMAL_ENV = os.path.join(REPO_ROOT, "tests", "defaults", "minimal.env")


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wav(seconds=3.0, freq=440.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return buf.getvalue()


def _no_preload_env(tmp_dir):
    path = os.path.join(tmp_dir, "no-preload.env")
    with open(MINIMAL_ENV, encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
        dst.writelines(line for line in src if not line.startswith("PRELOAD_MODEL="))
    return path


@pytest.fixture(scope="module", params=["documented", "no-preload"])
def server(request, tmp_path_factory):
    env_file = MINIMAL_ENV if request.param == "documented" else _no_preload_env(
        str(tmp_path_factory.mktemp("env")))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "stub_server.py"), "--env-file", env_file,
         "--port", str(port), "--timeout", "60"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT)
    deadline, log = time.time() + 90, []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        log.append(line)
        if line.startswith("READY "):
            break
    else:
        proc.kill()
    if proc.poll() is not None:
        pytest.fail("stub server exited:\n" + "".join(log[-60:]))
    url = f"http://127.0.0.1:{port}"
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health(server):
    assert httpx.get(server + "/health", timeout=10).status_code == 200


def test_the_request_speakr_sends_at_startup(server):
    params = {"encode": "true", "task": "transcribe", "output": "json", "language": "en",
              "diarize": "true", "enable_diarization": "true", "return_speaker_embeddings": "true",
              "min_speakers": "1", "max_speakers": "1"}
    r = httpx.post(server + "/asr", params=params,
                   files={"audio_file": ("canary.wav", _wav(), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["segments"], body
    assert "large-v3" in body["segments"][0]["text"]
    embeddings = body["speaker_embeddings"]
    assert list(embeddings) == ["SPEAKER_00"]
    assert len(embeddings["SPEAKER_00"]) == 256


def test_a_named_model(server):
    r = httpx.post(server + "/asr", params={"model": "distil-large-v3.5", "output": "json"},
                   files={"audio_file": ("a.wav", _wav(2.0), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    assert "distil-large-v3.5" in r.json()["segments"][0]["text"]


def test_an_unknown_model_is_a_400(server):
    r = httpx.post(server + "/asr", params={"model": "large-v4", "output": "json"},
                   files={"audio_file": ("a.wav", _wav(1.0), "audio/wav")}, timeout=60)
    assert r.status_code == 400, r.text


def test_v1_models_lists_whisper_1_and_turbo(server):
    r = httpx.get(server + "/v1/models", timeout=10)
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert {"whisper-1", "turbo", "large-v3"} <= ids


@pytest.mark.parametrize("model", ["whisper-1", "turbo"])
def test_v1_transcriptions(server, model):
    r = httpx.post(server + "/v1/audio/transcriptions", data={"model": model, "response_format": "json"},
                   files={"file": ("a.wav", _wav(2.0), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    assert r.json()["text"].strip()
