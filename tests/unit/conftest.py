"""Unit tests run without a GPU or model downloads; see tests/unit/README.md."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("DEVICE", "cpu")
