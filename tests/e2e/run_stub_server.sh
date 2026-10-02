#!/usr/bin/env sh
# Run the service on stand-in ML packages with the documented configuration.
# All arguments go to tests/e2e/stub_server.py (see tests/e2e/README.md).
cd "$(dirname "$0")/../.." || exit 1
exec python3 tests/e2e/stub_server.py "$@"
