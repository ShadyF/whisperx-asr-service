"""Pinned Egyptian Whisper preparation for the existing CTranslate2 backend."""

import hashlib
import fcntl
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

MODEL_ID = "mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching"
REVISION = "2d9f6140a9a4e340541a3183e1700bafb336fd75"
WEIGHT_SHA256 = "50a9a1cba2ca8c4889b2cc4fcfb7b6b294aecc4d900868da5a462923ff420364"
SOURCE_FILES = (
    "config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "model.safetensors", "README.md",
)
REQUIRED_FILES = {
    "model.bin", "config.json", "vocabulary.json", "tokenizer.json",
    "preprocessor_config.json", "generation_config.json", "SOURCE_README.md",
}
OPTIONS = {"load_as_float16": True, "low_cpu_mem_usage": True, "trust_remote_code": False}


def _digest(path: Path) -> dict:
    # Hash large checkpoints with bounded memory.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _identity() -> dict:
    # Exact installed versions separate conversions whose compatibility is unproven.
    versions = {name: metadata.version(name) for name in (
        "ctranslate2", "transformers", "torch", "tokenizers", "safetensors", "huggingface-hub", "numpy",
    )}
    return {
        "schema": 1, "source": MODEL_ID, "revision": REVISION,
        "checkpoint_sha256": WEIGHT_SHA256,
        "converter": "ctranslate2.converters.TransformersConverter",
        "versions": versions, "device": "cpu", "options": OPTIONS,
        "quantization": "int8_float16", "processor_export": "native-v1",
    }


def _prepare(source: Path, output: Path) -> dict:
    # Import heavy tools only when this opt-in artifact needs preparation.
    from huggingface_hub import hf_hub_download
    from ctranslate2.converters import TransformersConverter
    from transformers import AutoProcessor
    import torch

    # Fetch only pinned public source assets without storing credentials.
    source.mkdir()
    for name in SOURCE_FILES:
        downloaded = Path(hf_hub_download(
            MODEL_ID, name, revision=REVISION, local_dir=str(source), token=False,
        ))
        if downloaded != source / name or downloaded.is_symlink() or not downloaded.is_file():
            raise ValueError(f"Unsafe Egyptian source download: {name}")

    # Check the original checkpoint before handing it to the converter.
    source_files = {name: _digest(source / name) for name in SOURCE_FILES}
    if source_files["model.safetensors"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("Egyptian source checkpoint SHA-256 mismatch")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "whisper" or config.get("num_mel_bins") != 128:
        raise ValueError("Egyptian source must be the pinned 128-bin Whisper model")

    # Native conversion loads on CPU even when the serving process uses CUDA.
    with torch.device("cpu"):
        converter = TransformersConverter(str(source), **OPTIONS)
        converter.convert(str(output), quantization="int8_float16", force=False)

    # Export processor assets through their supported native APIs.
    processor = AutoProcessor.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    extractor = processor.feature_extractor
    if (extractor.feature_size, extractor.sampling_rate, extractor.chunk_length) != (128, 16000, 30):
        raise ValueError("Egyptian processor configuration mismatch")
    extractor.save_pretrained(str(output))
    processor.tokenizer.save_pretrained(str(output))
    shutil.copyfile(source / "generation_config.json", output / "generation_config.json")
    shutil.copyfile(source / "README.md", output / "SOURCE_README.md")
    return source_files


def _output_files(output: Path) -> dict:
    # Refuse incomplete, empty or linked output before publication.
    paths = list(output.iterdir())
    if not REQUIRED_FILES <= {path.name for path in paths}:
        raise ValueError("Egyptian conversion is incomplete")
    if any(path.is_symlink() or not path.is_file() or not path.stat().st_size for path in paths):
        raise ValueError("Egyptian conversion has unsafe or empty output")
    return {path.name: _digest(path) for path in paths if path.name != "provenance.json"}


def _check_completed(target: Path, identity: dict) -> None:
    # A manifest is the completion marker; verify every file before cache reuse.
    try:
        if target.is_symlink() or not target.is_dir():
            raise ValueError("unsafe artifact directory")
        manifest_path = target / "provenance.json"
        if manifest_path.is_symlink():
            raise ValueError("unsafe provenance file")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("incompatible provenance")
        if manifest["source_files"]["model.safetensors"]["sha256"] != WEIGHT_SHA256:
            raise ValueError("source provenance mismatch")
        if manifest["license"]["declaration"] != "MIT" or manifest["license"]["attribution"] != MODEL_ID:
            raise ValueError("missing publisher attribution")
        if manifest["files"] != _output_files(target):
            raise ValueError("output hashes differ")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Egyptian cache integrity check failed at {target}: {exc}") from exc


def prepare_egyptian_model(cache_dir: str) -> str:
    """Return a checked local CTranslate2 artifact, preparing it when needed."""
    # Keep all preparation data inside a dedicated namespace in the existing cache.
    try:
        identity = _identity()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(cache_dir).resolve() / "egyptian-turbo"
        root.mkdir(parents=True, exist_ok=True)
        target = root / key

        # Keep the lock file stable so all processes coordinate on the same inode.
        descriptor = os.open(root / f"{key}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            # Recheck after locking; another process may have completed preparation.
            if target.exists() or target.is_symlink():
                _check_completed(target, identity)
                return str(target)

            # Delete only this attempt's staging data on ordinary failure or interruption.
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
                stage = Path(temporary)
                output = stage / "output"
                source_files = _prepare(stage / "source", output)
                manifest = {
                    "identity": identity, "source_files": source_files,
                    "license": {
                        "declaration": "MIT", "attribution": MODEL_ID,
                        "notice": "MIT is the publisher's declaration; no independent legal audit is claimed.",
                        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
                    },
                    "files": _output_files(output),
                }
                (output / "provenance.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))

                # Verify completion metadata before an atomic same-filesystem publication.
                _check_completed(output, identity)
                output.rename(target)
        return str(target)
    except Exception as exc:
        raise RuntimeError(f"Egyptian model preparation failed in {cache_dir}: {exc}") from exc
