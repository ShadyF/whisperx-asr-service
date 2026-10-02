"""Stand-in for whisperx.diarize.DiarizationPipeline (pyannote)."""

import numpy as np

SAMPLE_RATE = 16000
TURN_SECONDS = 5.0
EMBEDDING_DIM = 256


def speaker_embedding(index):
    """The fixed, normalised embedding of speaker `index` (same on every run)."""
    rng = np.random.RandomState(1000 + index)
    vector = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class DiarizationPipeline:
    def __init__(self, model_name=None, use_auth_token=None, device="cpu", token=None, **kwargs):
        self.model_name = model_name
        self.device = device
        # The real wrapper exposes the pyannote pipeline as .model; None makes
        # the service skip hyperparameter tuning, as it does when unavailable.
        self.model = None

    def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None,
                 return_embeddings=False, **kwargs):
        duration = len(audio) / SAMPLE_RATE
        if num_speakers:
            count = int(num_speakers)
        else:
            count = 2 if duration >= 2 * TURN_SECONDS else 1
            if min_speakers:
                count = max(count, int(min_speakers))
            if max_speakers:
                count = min(count, int(max_speakers))
        count = max(1, count)

        turns, start, slot = [], 0.0, 0
        end_of_audio = max(duration, 0.5)
        while start < end_of_audio:
            end = min(end_of_audio, start + TURN_SECONDS)
            turns.append({"start": round(start, 3), "end": round(end, 3),
                          "speaker": f"SPEAKER_{slot % count:02d}"})
            start, slot = end, slot + 1

        if not return_embeddings:
            return turns
        used = sorted({t["speaker"] for t in turns})
        embeddings = {name: speaker_embedding(int(name.split("_")[1])) for name in used}
        return turns, embeddings
