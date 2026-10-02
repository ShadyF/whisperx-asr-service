# Documented-defaults tests

These tests start from the configuration the documentation gives users, not
from a developer's `.env`. `minimal.env` is the `.env` block of
`SETUP_GUIDE.md` (a test fails if the two drift apart), and each case builds
the container environment the way `docker compose` does from
`docker-compose.yml` and that file (`tests/e2e/compose_env.py`). The service
settings are then imported in a fresh interpreter on the stand-in ML packages
in `tests/e2e/stubs`, so no GPU or model is needed.

Cases cover the documented setup and its documented variations: no
`PRELOAD_MODEL` line (Speakr #409), an empty or blank value, an alias, another
model, `DEFAULT_MODEL`, `OPENAI_WHISPER1_MODEL`, a misspelt model, the qwen3
and external backends, and malformed integer settings.

```bash
pip install -r tests/requirements-ci.txt
PYTHONPATH=tests/e2e/stubs python -m pytest -q tests/defaults
```

## Regression check against an older version

`WX_APP_ROOT` points the probe at another checkout of `app/`. On v0.4.1
(`cac0f15`) the #409 cases fail, which is what these tests are for:

```bash
mkdir -p /tmp/wx-041 && git archive cac0f15 app | tar -x -C /tmp/wx-041
WX_APP_ROOT=/tmp/wx-041 PYTHONPATH=tests/e2e/stubs python -m pytest -q tests/defaults
```
