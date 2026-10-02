#!/usr/bin/env python3
"""Run the real service on stand-in ML packages, configured like a user's setup.

The container environment is built from docker-compose.yml and an env file
(default: tests/defaults/minimal.env, the .env block of SETUP_GUIDE.md), then
`uvicorn app.main:app` starts with tests/e2e/stubs first on PYTHONPATH. No GPU,
CUDA or model download is involved. See tests/e2e/README.md.

    python3 tests/e2e/stub_server.py --port 9000
    python3 tests/e2e/stub_server.py --env-file my.env --set PRELOAD_MODEL= --print-env

Prints "READY http://HOST:PORT" once /health answers, then keeps running until
interrupted. Exits non-zero if the service does not come up.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compose_env import (  # noqa: E402
    DEFAULT_COMPOSE_FILE, DEFAULT_ENV_FILE, DEFAULT_SERVICE, REPO_ROOT, compose_environment,
)

STUBS = os.path.join(HERE, "stubs")
_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV", "SYSTEMROOT")


def build_environment(env_file, compose_file, service, resolver, overrides):
    container_env = compose_environment(compose_file, env_file, service, resolver)
    container_env.update(overrides)
    child = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    child.update(container_env)
    child["PYTHONPATH"] = os.pathsep.join([STUBS, REPO_ROOT])
    child["PYTHONUNBUFFERED"] = "1"
    return container_env, child


def wait_for_health(url, timeout, proc):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--resolver", choices=("auto", "docker", "parser"), default="auto")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one container variable after compose (repeatable)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--print-env", action="store_true", help="print the container environment and exit")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = value
    container_env, child_env = build_environment(
        args.env_file, args.compose_file, args.service, args.resolver, overrides)

    if args.print_env:
        for key in sorted(container_env):
            print(f"{key}={container_env[key]}")
        return 0

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(args.port)],
        cwd=REPO_ROOT, env=child_env,
    )

    def stop(*_):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    url = f"http://{args.host}:{args.port}"
    if not wait_for_health(url, args.timeout, proc):
        print(f"stub server did not become healthy at {url}", file=sys.stderr)
        stop()
        return 1
    print(f"READY {url}", flush=True)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
