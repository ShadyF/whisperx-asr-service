# Unit tests

Fast tests for model selection and loading. They fake model loading and audio
decoding, so they need neither a GPU nor downloaded models, only the Python
dependencies of the service image.

Run them in the image without a GPU and without publishing a port, so they
never touch a running service:

```bash
docker run --rm -v "$PWD":/workspace:ro -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh \
  learnedmachine/whisperx-asr-service:latest \
  -c "pip install -q pytest 'httpx<0.28' && python3 -m pytest -q -p no:cacheprovider tests/unit"
```
