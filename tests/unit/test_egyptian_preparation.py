"""Tiny native-tool substitutes exercise the filesystem preparation seam."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import multiprocessing
from queue import Empty

import pytest


@pytest.fixture
def tools(monkeypatch):
    # Replace only external tool boundaries; keep real cache filesystem operations.
    from app import egyptian_model as model
    calls = []
    state = {"failure": None, "version": "test-1"}
    monkeypatch.setattr(model, "WEIGHT_SHA256", hashlib.sha256(b"checkpoint").hexdigest())
    monkeypatch.setattr(model.metadata, "version", lambda name: state["version"])

    def download(repo_id, filename, **kwargs):
        # Write tiny source assets under the requested isolated source directory.
        calls.append(("download", repo_id, filename, kwargs))
        if state["failure"] == "download":
            raise OSError("offline")
        root = Path(kwargs["local_dir"])
        root.mkdir(parents=True, exist_ok=True)
        data = b"checkpoint" if filename == "model.safetensors" else b"{}"
        if filename == "config.json":
            data = b'{"model_type":"whisper","num_mel_bins":128}'
        target = root / filename
        target.write_bytes(data)
        return str(target)

    class Converter:
        def __init__(self, source, **kwargs):
            # Record the public converter configuration, not its internals.
            calls.append(("converter", source, kwargs))

        def convert(self, output, **kwargs):
            # Leave a partial artifact on failure to exercise publication safety.
            calls.append(("convert", output, kwargs))
            root = Path(output)
            root.mkdir()
            (root / "model.bin").write_bytes(b"converted")
            if state["failure"] == "convert":
                raise ValueError("conversion failed")
            for name in ("config.json", "vocabulary.json"):
                (root / name).write_text("{}")

    class Asset:
        feature_size = 128
        sampling_rate = 16000
        chunk_length = 30

        def __init__(self, name):
            # Keep the export filename for this processor or tokenizer substitute.
            self.name = name

        def save_pretrained(self, output):
            # Represent the native processor/tokenizer export with tiny JSON files.
            calls.append(("export", self.name))
            (Path(output) / self.name).write_text("{}")

    def processor(source, **kwargs):
        # Capture local-only loading and remote-code protection.
        calls.append(("processor", source, kwargs))
        return SimpleNamespace(feature_extractor=Asset("preprocessor_config.json"), tokenizer=Asset("tokenizer.json"))

    def device(name):
        # Observe explicit CPU placement without initializing a real torch runtime.
        calls.append(("device", name))
        return nullcontext()

    # Make all model tool imports resolve to controlled, download-free substitutes.
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", SimpleNamespace(TransformersConverter=Converter))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=processor)))
    import torch
    monkeypatch.setattr(torch, "device", device, raising=False)
    return model, calls, state


def test_first_preparation_pins_source_and_uses_native_cpu_conversion(tools, tmp_path):
    # Prepare through the filesystem seam and inspect the published artifact.
    model, calls, _ = tools
    output = Path(model.prepare_egyptian_model(str(tmp_path)))
    manifest = json.loads((output / "provenance.json").read_text())
    assert manifest["identity"]["revision"] == "2d9f6140a9a4e340541a3183e1700bafb336fd75"
    assert manifest["identity"]["source"] == model.MODEL_ID
    assert manifest["license"]["declaration"] == "MIT"
    assert "publisher" in manifest["license"]["notice"]
    assert manifest["files"]["model.bin"]["sha256"] == hashlib.sha256(b"converted").hexdigest()
    downloads = [call for call in calls if call[0] == "download"]
    assert downloads and all(call[3]["revision"] == manifest["identity"]["revision"] for call in downloads)
    assert all(call[3]["token"] is False for call in downloads)
    assert ("device", "cpu") in calls
    assert next(call[2] for call in calls if call[0] == "converter") == {
        "load_as_float16": True, "low_cpu_mem_usage": True, "trust_remote_code": False,
    }
    assert next(call[2] for call in calls if call[0] == "convert") == {"quantization": "int8_float16", "force": False}
    assert next(call[2] for call in calls if call[0] == "processor") == {
        "local_files_only": True, "trust_remote_code": False, "use_fast": True,
    }
    assert {call[1] for call in calls if call[0] == "export"} == {"preprocessor_config.json", "tokenizer.json"}


def test_completed_cache_reuses_offline_but_rejects_changed_integrity(tools, tmp_path):
    # A checked completed artifact needs no source access on subsequent loads.
    model, calls, state = tools
    output = Path(model.prepare_egyptian_model(str(tmp_path)))
    calls.clear()
    state["failure"] = "download"
    assert model.prepare_egyptian_model(str(tmp_path)) == str(output)
    assert calls == []

    # Corruption must never be returned or deleted along with unrelated cache data.
    unrelated = tmp_path / "ordinary-model"
    unrelated.write_text("retain")
    (output / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        model.prepare_egyptian_model(str(tmp_path))
    assert unrelated.read_text() == "retain"
    assert (output / "model.bin").read_bytes() == b"corrupt"


def test_processes_publish_one_complete_artifact(tools, monkeypatch, tmp_path):
    # Pause native conversion so another process tries the same cache concurrently.
    model, _, _ = tools
    context = multiprocessing.get_context("fork")
    events = context.Queue()
    results = context.Queue()
    release = context.Event()
    converter = sys.modules["ctranslate2.converters"].TransformersConverter
    original = converter.convert

    def convert(self, output, **kwargs):
        # Publication must remain invisible until conversion and export finish.
        events.put("converting")
        assert release.wait(5)
        return original(self, output, **kwargs)

    def worker():
        # Capture process failures as observable preparation results.
        events.put("started")
        try:
            results.put(("ok", model.prepare_egyptian_model(str(tmp_path))))
        except Exception as exc:
            results.put(("error", str(exc)))

    # Start one blocked converter before letting a second process request the artifact.
    monkeypatch.setattr(converter, "convert", convert)
    first = context.Process(target=worker)
    second = context.Process(target=worker)
    first.start()
    assert events.get(timeout=5) == "started"
    assert events.get(timeout=5) == "converting"
    second.start()
    try:
        assert events.get(timeout=5) == "started"
        with pytest.raises(Empty):
            events.get(timeout=0.3)
        assert not list((tmp_path / "egyptian-turbo").glob("*/provenance.json"))
    finally:
        # Release and collect child processes even when an assertion fails.
        release.set()
        first.join(5)
        second.join(5)
        for process in (first, second):
            if process.is_alive():
                process.kill()
                process.join()

    # Both callers receive one published, checked artifact.
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    assert all(status == "ok" for status, _ in outcomes), outcomes
    assert outcomes[0][1] == outcomes[1][1]


@pytest.mark.parametrize("change", ["version", "revision", "source_hash"])
def test_changed_conversion_identity_never_reuses_previous_output(tools, monkeypatch, tmp_path, change):
    # Keep older artifacts intact while using a distinct identity for new preparation.
    model, calls, state = tools
    previous = Path(model.prepare_egyptian_model(str(tmp_path)))
    calls.clear()
    if change == "version":
        state["version"] = "test-2"
    elif change == "revision":
        monkeypatch.setattr(model, "REVISION", "different-pinned-revision")
    else:
        monkeypatch.setattr(model, "WEIGHT_SHA256", "0" * 64)

    # Offline access must fail instead of returning incompatible cached output.
    state["failure"] = "download"
    with pytest.raises(RuntimeError, match="offline"):
        model.prepare_egyptian_model(str(tmp_path))
    assert calls
    assert (previous / "provenance.json").is_file()


@pytest.mark.parametrize("failure", ["download", "convert", "hash", "permissions"])
def test_failed_preparation_never_publishes_or_removes_other_data(tools, monkeypatch, tmp_path, failure):
    # Keep failure checks on the same filesystem seam without invoking model tools.
    model, calls, state = tools
    unrelated = tmp_path / "ordinary-model"
    unrelated.write_text("retain")
    state["failure"] = failure
    expected = {"download": "offline", "convert": "conversion failed", "hash": "SHA-256", "permissions": "Permission denied"}[failure]
    if failure == "hash":
        monkeypatch.setattr(model, "WEIGHT_SHA256", "0" * 64)
    elif failure == "permissions":
        monkeypatch.setattr(model.os, "open", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Permission denied")))

    # A failed attempt leaves neither a completed artifact nor owned staging data.
    with pytest.raises(RuntimeError, match=expected):
        model.prepare_egyptian_model(str(tmp_path))
    assert not list(tmp_path.glob("egyptian-turbo/*/provenance.json"))
    assert not list(tmp_path.glob("egyptian-turbo/.prepare-*"))
    assert unrelated.read_text() == "retain"
    if failure == "hash":
        assert not any(call[0] == "converter" for call in calls)


@pytest.mark.parametrize("damage", ["missing_manifest", "missing_binary", "provenance", "symlink", "extra_file"])
def test_incomplete_or_changed_completed_entry_is_rejected(tools, tmp_path, damage):
    # Treat incomplete and altered completed directories as errors, not reusable models.
    model, _, _ = tools
    output = Path(model.prepare_egyptian_model(str(tmp_path)))
    if damage == "missing_manifest":
        (output / "provenance.json").unlink()
    elif damage == "missing_binary":
        (output / "model.bin").unlink()
    elif damage == "provenance":
        manifest = json.loads((output / "provenance.json").read_text())
        manifest["identity"]["revision"] = "unverified"
        (output / "provenance.json").write_text(json.dumps(manifest))
    elif damage == "symlink":
        (output / "model.bin").unlink()
        (output / "model.bin").symlink_to(output / "config.json")
    else:
        (output / "unexpected.txt").write_text("not part of the artifact")

    # Never silently repair or substitute a damaged completed model.
    with pytest.raises(RuntimeError, match="integrity"):
        model.prepare_egyptian_model(str(tmp_path))


def test_killed_preparation_is_not_a_completed_artifact(tools, monkeypatch, tmp_path):
    # Kill a process with a partial converter output and a held cache lock.
    model, _, _ = tools
    context = multiprocessing.get_context("fork")
    entered = context.Event()
    release = context.Event()
    converter = sys.modules["ctranslate2.converters"].TransformersConverter
    original = converter.convert

    def interrupted(self, output, **kwargs):
        # Leave an actual partial output behind to represent abrupt termination.
        Path(output).mkdir()
        (Path(output) / "model.bin").write_bytes(b"partial")
        entered.set()
        release.wait(10)

    # Confirm no completion marker exists while the partial output is being written.
    monkeypatch.setattr(converter, "convert", interrupted)
    process = context.Process(target=model.prepare_egyptian_model, args=(str(tmp_path),))
    process.start()
    try:
        assert entered.wait(5)
        assert not list(tmp_path.glob("egyptian-turbo/*/provenance.json"))
    finally:
        # Stop and collect the interrupted child even when an assertion fails.
        process.kill()
        process.join(5)

    # A fresh caller gets the released lock and builds independently of abandoned staging.
    monkeypatch.setattr(converter, "convert", original)
    output = Path(model.prepare_egyptian_model(str(tmp_path)))
    assert (output / "model.bin").read_bytes() == b"converted"
    assert (output / "provenance.json").is_file()
