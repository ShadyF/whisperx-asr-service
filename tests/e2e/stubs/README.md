# Stand-in ML packages

These packages replace `torch`, `whisperx` and `faster_whisper` so the real
service code in `app/` runs on a machine without a GPU, CUDA or downloaded
models. Put this directory first on `PYTHONPATH`; nothing in `app/` knows the
stubs exist.

What they do:

- `faster_whisper.available_models()` returns the model names of
  faster-whisper 1.2.1, and `faster_whisper.utils.download_model()` raises the
  same `ValueError("Invalid model size '...', expected one of: ...")` for an
  unknown name. Hugging Face ids (`org/repo`) and existing directories are
  accepted, as in the real engine.
- `whisperx.load_model()` validates the name the same way and returns a model
  whose `transcribe()` produces one segment per 5 seconds of audio, with text
  naming the model that was used ("Stub segment 1 transcribed by large-v3.").
- `whisperx.load_audio()` decodes WAV itself, uses `ffmpeg` for other formats
  when it is installed, and otherwise estimates a length from the file size.
- `whisperx.align()` spreads word timestamps evenly over each segment.
- `whisperx.diarize.DiarizationPipeline` (where the app imports diarization)
  assigns speakers to 5-second turns in rotation, honours `num_speakers`,
  `min_speakers` and `max_speakers`, and with `return_embeddings=True` returns
  a fixed, normalised 256-dimensional embedding per speaker. The same speaker
  index always gets the same embedding.
- `torch` provides only what the app touches: `torch.cuda.*` (no GPU) and
  `torch.device`.

`whisperx.LOADED` lists the model names loaded in the process, for tests.
