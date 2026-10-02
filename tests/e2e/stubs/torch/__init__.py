"""Stand-in for torch: only what the service touches, and no GPU."""

import contextlib

__version__ = "0.0.0+stub"
float16 = "float16"
float32 = "float32"
int8 = "int8"


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def synchronize(*args, **kwargs):
        return None

    @staticmethod
    def memory_allocated(*args, **kwargs):
        return 0

    @staticmethod
    def memory_reserved(*args, **kwargs):
        return 0


cuda = _Cuda()


class device:  # noqa: N801 - mirrors torch.device
    def __init__(self, spec="cpu"):
        self.type = str(spec).split(":")[0]
        self._spec = str(spec)

    def __str__(self):
        return self._spec

    def __repr__(self):
        return f"device(type='{self.type}')"


@contextlib.contextmanager
def no_grad():
    yield


@contextlib.contextmanager
def inference_mode():
    yield
