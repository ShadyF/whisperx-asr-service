# Stub-server end-to-end tests

`stub_server.py` runs the real service (`app/`, unmodified) on the stand-in ML
packages in `stubs/`, configured the way a user's setup is configured: the
container environment comes from `docker-compose.yml` plus an env file, as
`docker compose` builds it (`compose_env.py`). No GPU, CUDA or model download
is needed, so it runs on a plain CI runner in a few seconds.

## Starting it

```bash
pip install -r tests/requirements-ci.txt
tests/e2e/run_stub_server.sh --port 9000
# prints "READY http://127.0.0.1:9000" when /health answers
```

Options (all optional):

| Option | Default | Meaning |
|---|---|---|
| `--env-file PATH` | `tests/defaults/minimal.env` | the user's `.env`; the default is the `SETUP_GUIDE.md` block |
| `--port N` / `--host H` | `9000` / `127.0.0.1` | where to listen |
| `--set KEY=VALUE` | none | override one container variable after compose (repeatable), e.g. `--set PRELOAD_MODEL=` |
| `--resolver auto\|docker\|parser` | `auto` | `docker compose config` when Docker is available, else the built-in parser |
| `--compose-file PATH` | `docker-compose.yml` | another compose file |
| `--print-env` | | print the container environment and exit |
| `--timeout S` | `60` | how long to wait for `/health` |

It exits non-zero if the service does not become healthy, and stops the
service on SIGTERM or SIGINT.

## Using it from Speakr's CI

```bash
git clone --depth 1 https://github.com/murtaza-nasir/whisperx-asr-service.git /tmp/wx
pip install -r /tmp/wx/tests/requirements-ci.txt
/tmp/wx/tests/e2e/run_stub_server.sh --port 9000 > /tmp/wx.log 2>&1 &
for i in $(seq 1 60); do curl -fs http://127.0.0.1:9000/health && break; sleep 1; done
# ASR_BASE_URL=http://127.0.0.1:9000 for Speakr
```

To test the setup behind Speakr #409 (no `PRELOAD_MODEL` line in `.env`),
write an env file without that line, or pass `--set PRELOAD_MODEL=`.

## What the stand-ins return

Transcripts are deterministic: one segment per 5 seconds of audio, with text
naming the model that produced it ("Stub segment 1 transcribed by large-v3."),
so a test can see which model a request used. Speakers rotate every 5 seconds
(one speaker for audio under 10 seconds unless a count is given), and each
speaker has a fixed 256-dimensional embedding. Unknown model names fail with
faster-whisper's own "Invalid model size" error, which the service returns as
a 400. See `stubs/README.md`.

## Tests

```bash
PYTHONPATH=tests/e2e/stubs python -m pytest -q tests/e2e
```

`test_stub_server.py` starts the launcher twice, with the documented `.env`
and with `PRELOAD_MODEL` left out, and sends the request Speakr's startup
voice-embedding check sends, a named model, an unknown model, and the
OpenAI-compatible endpoints. `test_compose_env.py` checks the env-file and
interpolation rules and, where Docker is available, that the parser agrees
with `docker compose config`.
