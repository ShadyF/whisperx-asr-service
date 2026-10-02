"""The environment a container gets from docker-compose.yml plus a .env file.

Users configure the service through `.env` and the shipped compose file, which
forwards most variables as `${VAR:-}`: a key missing from `.env` arrives in the
container as an empty string, not unset. Tests that describe "the documented
setup" must therefore see the environment exactly as compose builds it.

`compose_environment()` asks `docker compose config` when Docker is available
and otherwise uses `parse_compose_environment()`, a small parser that follows
compose's rules for .env files and variable interpolation. Tests check that the
two agree whenever Docker is present.
"""

import json
import os
import re
import shutil
import subprocess
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yml")
DEFAULT_ENV_FILE = os.path.join(REPO_ROOT, "tests", "defaults", "minimal.env")
DEFAULT_SERVICE = "whisperx-asr"


# ---------------------------------------------------------------------------
# .env files (compose semantics)
# ---------------------------------------------------------------------------
def parse_env_file(path: str) -> Dict[str, str]:
    """Parse a .env file the way docker compose does.

    Blank lines and lines starting with # are skipped; `export KEY=...` is
    allowed; single-quoted values are literal; double-quoted values keep
    everything inside the quotes; in unquoted values a ` #` starts a comment
    and surrounding whitespace is trimmed.
    """
    values: Dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value[:1] in ("'", '"'):
                quote = value[0]
                closing = value.find(quote, 1)
                value = value[1:closing] if closing != -1 else value[1:]
            else:
                comment = re.search(r"\s#", value)
                if comment:
                    value = value[:comment.start()]
                value = value.strip()
            values[key] = value
    return values


# ---------------------------------------------------------------------------
# Interpolation (compose semantics)
# ---------------------------------------------------------------------------
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _apply(name, op, arg, variables):
    value = variables.get(name)
    if op in (":-", "-"):
        unset = value is None or (op == ":-" and value == "")
        return interpolate(arg, variables) if unset else value
    if op in (":+", "+"):
        present = value is not None and (op == "+" or value != "")
        return interpolate(arg, variables) if present else ""
    if op in (":?", "?"):
        if value is None or (op == ":?" and value == ""):
            raise ValueError(f"required variable {name} is missing: {arg}")
        return value
    return value if value is not None else ""


def interpolate(text: str, variables: Dict[str, str]) -> str:
    """Compose variable interpolation: $VAR, ${VAR}, ${VAR:-default},
    ${VAR-default}, ${VAR:+alt}, ${VAR+alt}, ${VAR:?err}, ${VAR?err}, and $$
    for a literal $. Defaults may themselves contain ${...}."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch != "$":
            out.append(ch)
            i += 1
            continue
        if text.startswith("$$", i):
            out.append("$")
            i += 2
            continue
        if text.startswith("${", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text.startswith("${", j):
                    depth += 1
                    j += 2
                    continue
                if text[j] == "}":
                    depth -= 1
                j += 1
            inner = text[i + 2:j - 1]
            match = _NAME.match(inner)
            if not match:
                out.append(text[i:j])
            else:
                name, rest = match.group(0), inner[match.end():]
                op = next((o for o in (":-", ":+", ":?", "-", "+", "?") if rest.startswith(o)), "")
                out.append(_apply(name, op, rest[len(op):], variables))
            i = j
            continue
        match = _NAME.match(text, i + 1)
        if match:
            out.append(_apply(match.group(0), "", "", variables))
            i = match.end()
        else:
            out.append("$")
            i += 1
    return "".join(out)


def _service_environment_lines(compose_file: str, service: str):
    """The raw `environment:` entries of one service, without needing PyYAML."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        with open(compose_file, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        env = (data.get("services", {}).get(service, {}) or {}).get("environment", [])
        if isinstance(env, dict):
            return [f"{k}={'' if v is None else v}" for k, v in env.items()]
        return [str(item) for item in env]

    lines, in_service, in_env, service_indent, env_indent = [], False, False, None, None
    with open(compose_file, encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip())
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(rf"^{re.escape(service)}:\s*$", stripped):
                in_service, service_indent = True, indent
                continue
            if in_service and indent <= service_indent:
                break
            if in_service and stripped == "environment:":
                in_env, env_indent = True, indent
                continue
            if in_env:
                if indent <= env_indent:
                    in_env = False
                    continue
                if stripped.startswith("- "):
                    item = stripped[2:].strip()
                    if item[:1] in ("'", '"') and item[-1:] == item[:1]:
                        item = item[1:-1]
                    lines.append(item)
    return lines


def parse_compose_environment(compose_file: str = DEFAULT_COMPOSE_FILE,
                              env_file: str = DEFAULT_ENV_FILE,
                              service: str = DEFAULT_SERVICE,
                              shell_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Pure-Python equivalent of `docker compose config` for `environment:`."""
    variables = dict(parse_env_file(env_file)) if env_file else {}
    # Shell variables take precedence over the .env file, as in compose.
    variables.update(shell_env or {})
    result: Dict[str, str] = {}
    for entry in _service_environment_lines(compose_file, service):
        if "=" in entry:
            key, value = entry.split("=", 1)
            result[key] = interpolate(value, variables)
        elif entry in variables:
            # `- KEY` without a value passes the variable through only if set.
            result[entry] = variables[entry]
    return result


def docker_compose_environment(compose_file: str = DEFAULT_COMPOSE_FILE,
                               env_file: str = DEFAULT_ENV_FILE,
                               service: str = DEFAULT_SERVICE,
                               shell_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Ask docker compose itself. Raises RuntimeError when it is unavailable."""
    if not shutil.which("docker"):
        raise RuntimeError("docker is not installed")
    env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG") if k in os.environ}
    env.update(shell_env or {})
    cmd = ["docker", "compose", "-f", os.path.abspath(compose_file), "--project-directory",
           os.path.dirname(os.path.abspath(compose_file))]
    if env_file:
        cmd += ["--env-file", os.path.abspath(env_file)]
    cmd += ["config", "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose config failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    environment = data["services"][service].get("environment") or {}
    return {k: ("" if v is None else str(v)) for k, v in environment.items()}


def compose_environment(compose_file: str = DEFAULT_COMPOSE_FILE,
                        env_file: str = DEFAULT_ENV_FILE,
                        service: str = DEFAULT_SERVICE,
                        resolver: str = "auto",
                        shell_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The container environment for the documented setup.

    resolver: "docker", "parser", or "auto" (docker when available).
    """
    if resolver in ("auto", "docker"):
        try:
            return docker_compose_environment(compose_file, env_file, service, shell_env)
        except RuntimeError:
            if resolver == "docker":
                raise
    return parse_compose_environment(compose_file, env_file, service, shell_env)
