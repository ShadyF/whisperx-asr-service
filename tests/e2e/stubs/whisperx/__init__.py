"""Stand-in for whisperx: deterministic transcription, alignment and speakers.

See tests/e2e/stubs/README.md. The real service code calls these functions
exactly as it calls whisperx; nothing here needs a GPU or a model download.
"""

import os
import shutil
import subprocess
import wave

import numpy as np

from faster_whisper.utils import download_model as _check_model_name

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 5.0

# Model names loaded in this process, in order (for tests).
LOADED = []


def load_audio(file, sr=SAMPLE_RATE):
    try:
        with wave.open(file, "rb") as w:
            frames = w.readframes(w.getnframes())
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if width == 1:
            data = (data - 128.0) / 128.0
        else:
            data /= float(np.iinfo(dtype).max)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        if rate != sr and len(data):
            count = int(len(data) * sr / rate)
            data = np.interp(np.linspace(0, len(data), count, endpoint=False),
                             np.arange(len(data)), data).astype(np.float32)
        return data.astype(np.float32)
    except (wave.Error, EOFError, KeyError):
        pass
    if shutil.which("ffmpeg"):
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1",
             "-acodec", "pcm_s16le", "-ar", str(sr), "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    # No decoder available: about one second per 16 kB keeps the length plausible.
    size = os.path.getsize(file)
    return np.zeros(max(sr, int(size / 16000 * sr)), dtype=np.float32)


class _Options:
    def __init__(self):
        self.hotwords = None
        self.initial_prompt = None


class _StubWhisperModel:
    def __init__(self, name):
        self.name = name
        self.options = _Options()

    def transcribe(self, audio, batch_size=None, language=None, task="transcribe", **kwargs):
        duration = len(audio) / SAMPLE_RATE
        count = max(1, int(np.ceil(duration / SEGMENT_SECONDS))) if duration > 0 else 1
        segments = []
        for i in range(count):
            start = i * SEGMENT_SECONDS
            end = min(duration, start + SEGMENT_SECONDS) if duration > 0 else 1.0
            if end <= start:
                end = start + 0.5
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": f" Stub segment {i + 1} transcribed by {self.name}.",
            })
        return {"segments": segments, "language": language or "en"}


def load_model(whisper_arch, device="cuda", compute_type="float16", download_root=None, **kwargs):
    if not os.path.isdir(str(whisper_arch)):
        _check_model_name(whisper_arch, output_dir=download_root)
    LOADED.append(whisper_arch)
    return _StubWhisperModel(whisper_arch)


def load_align_model(language_code=None, device="cuda", model_name=None, model_dir=None, **kwargs):
    return object(), {"language": language_code or "en"}


def align(transcript, model, align_model_metadata, audio, device, return_char_alignments=False, **kwargs):
    segments, all_words = [], []
    for seg in transcript:
        words = seg.get("text", "").split()
        span = max(seg["end"] - seg["start"], 0.001)
        step = span / max(len(words), 1)
        entries = [
            {"word": w, "start": round(seg["start"] + k * step, 3),
             "end": round(seg["start"] + (k + 1) * step, 3), "score": 0.99}
            for k, w in enumerate(words)
        ]
        aligned = dict(seg)
        aligned["words"] = entries
        segments.append(aligned)
        all_words.extend(entries)
    return {"segments": segments, "word_segments": all_words}


def _speaker_at(turns, t0, t1):
    mid = (t0 + t1) / 2
    for turn in turns:
        if turn["start"] <= mid < turn["end"]:
            return turn["speaker"]
    return turns[-1]["speaker"] if turns else None


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    turns = list(diarize_df)
    for seg in transcript_result.get("segments", []):
        speaker = _speaker_at(turns, seg["start"], seg["end"])
        if speaker is not None:
            seg["speaker"] = speaker
        for word in seg.get("words", []):
            if "start" in word and "end" in word:
                word_speaker = _speaker_at(turns, word["start"], word["end"])
                if word_speaker is not None:
                    word["speaker"] = word_speaker
    return transcript_result
