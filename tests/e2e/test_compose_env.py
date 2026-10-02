"""compose_env builds the container environment the way docker compose does."""

import os
import shutil
import subprocess

import pytest

from compose_env import (
    DEFAULT_COMPOSE_FILE, compose_environment, docker_compose_environment, interpolate,
    parse_compose_environment, parse_env_file,
)


def _docker_compose_available():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


def test_env_file_rules(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n"
        "PLAIN=large-v3\n"
        "EMPTY=\n"
        "COMMENTED=large-v3   # Leave empty to disable\n"
        "HASH_INSIDE=a#b\n"
        'DOUBLE="  spaced  "\n'
        "SINGLE='$NOT_EXPANDED'\n"
        "export EXPORTED=yes\n"
        "\n", encoding="utf-8")
    values = parse_env_file(str(path))
    assert values == {
        "PLAIN": "large-v3", "EMPTY": "", "COMMENTED": "large-v3", "HASH_INSIDE": "a#b",
        "DOUBLE": "  spaced  ", "SINGLE": "$NOT_EXPANDED", "EXPORTED": "yes",
    }


@pytest.mark.parametrize("text, variables, expected", [
    ("${A:-x}", {}, "x"),
    ("${A:-x}", {"A": ""}, "x"),
    ("${A:-x}", {"A": "y"}, "y"),
    ("${A-x}", {"A": ""}, ""),
    ("${A-x}", {}, "x"),
    ("${A:-}", {}, ""),
    ("${A}", {}, ""),
    ("$A/b", {"A": "a"}, "a/b"),
    ("$$A", {"A": "a"}, "$A"),
    ("${A:+set}", {"A": "1"}, "set"),
    ("${A:+set}", {"A": ""}, ""),
    ("${A:-${B:-z}}", {}, "z"),
])
def test_interpolation(text, variables, expected):
    assert interpolate(text, variables) == expected


def test_a_key_missing_from_env_arrives_empty(tmp_path):
    """The mechanism behind Speakr #409."""
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=x\n", encoding="utf-8")
    env = parse_compose_environment(env_file=str(path))
    assert env["PRELOAD_MODEL"] == ""
    assert env["DEVICE"] == "cuda"          # ${DEVICE:-cuda}
    assert env["BATCH_SIZE"] == "16"        # ${BATCH_SIZE:-16}


@pytest.mark.skipif(not _docker_compose_available(), reason="docker compose not available")
@pytest.mark.parametrize("contents", [
    "HF_TOKEN=hf_x\nDEVICE=cuda\nCOMPUTE_TYPE=float16\nBATCH_SIZE=16\nPRELOAD_MODEL=large-v3\n",
    "HF_TOKEN=hf_x\n",
    "HF_TOKEN=hf_x\nPRELOAD_MODEL=\nDEFAULT_MODEL=small\nBATCH_SIZE=  8  # comment\n",
    'HF_TOKEN=hf_x\nPRELOAD_MODEL="   "\nASR_BACKEND=qwen3\n',
])
def test_parser_matches_docker_compose(tmp_path, contents):
    path = tmp_path / ".env"
    path.write_text(contents, encoding="utf-8")
    assert parse_compose_environment(env_file=str(path)) == docker_compose_environment(env_file=str(path))


def test_auto_resolver_returns_the_compose_environment():
    env = compose_environment()
    assert env["PRELOAD_MODEL"] == "large-v3"
    assert os.path.exists(DEFAULT_COMPOSE_FILE)
