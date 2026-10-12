# Upstream v0.4.3: fork comparison and migration

Research checked: 2026-10-12. Source, registry metadata, deployment configuration, and the live image/environment were checked. No representative audio test, image replacement, or deployment change was performed.

## Recommendation

Keep the deployed fork while evaluating upstream v0.4.3 on representative recordings. Upstream now exposes VAD settings that overlap with the fork, but it does not implement the native decoder used by this deployment. Replacing the image would change transcription behavior even if the environment variables remained unchanged.

| Option | Result | Benefit | Cost or limitation |
| --- | --- | --- | --- |
| 🟡 ✅ Evaluate upstream's built-in VAD settings, then migrate if accepted (Recommendation) | Compare upstream Pyannote and Silero at 30-second chunks with the current native 20-second configuration | Can remove the maintained decoder fork using upstream capabilities | Requires audio, API, resource, and deployment checks; exact decoder semantics change |
| ⚪ ⚠️ Retain the existing fork | Keep native decoding and the current configuration | Preserves the active runtime path while validation remains incomplete | Continues custom-code and dependency maintenance |
| 🟠 ⚠️ Seek upstream acceptance of native decoding | Propose the fork's decoder path for upstream review | Could preserve the required semantics under upstream maintenance | Acceptance is uncertain; requires implementation review and real-audio evidence. This work has not been requested |

The preferred route is upstream's existing configuration if its output meets the deployment's needs. If exact native-decoder semantics are required, retain the fork. Keep the current image and configuration available for rollback until a replacement passes acceptance checks.

## Versions and ancestry

The latest upstream release checked was [`v0.4.3`](https://github.com/murtaza-nasir/whisperx-asr-service/releases/tag/v0.4.3), published on 2026-10-11. This is the service-wrapper version, not the WhisperX library version. The deployed fork image is `ghcr.io/shadyf/whisperx-asr-service:0.4.5`; the reviewed fork HEAD is `1537c02`. The higher fork version does not imply that it contains all upstream v0.4.3 changes.

The [official comparison API](https://api.github.com/repos/murtaza-nasir/whisperx-asr-service/compare/v0.4.3...ShadyF:main) reports divergence: the fork is eight commits ahead and two behind, with merge base `4d6acc6` (v0.4.2). It lists 21 changed files. Its file statistics describe the fork-side changes from the merge base; upstream's two commits must also be examined when comparing the resulting behavior.

## Runtime differences at the assessed snapshot

| Area | Upstream v0.4.3 | Reviewed fork at 1537c02 |
| --- | --- | --- |
| Transcription | Batched WhisperX decoding | Selectable batched or native decoding; native calls faster-whisper separately for each VAD chunk |
| VAD selection | `VAD_METHOD=pyannote` or `silero` | Pyannote only at this snapshot; local reconciliation adds method selection below |
| VAD thresholds | Configurable onset and offset; Pyannote defaults are `0.5` and `0.363` | Configurable onset and offset with startup validation, including `offset <= onset` |
| Chunk duration | 30 seconds; not configurable | `VAD_CHUNK_SIZE` configurable; deployed value is 20 seconds |
| Decoder thresholds | Not exposed because the batched pipeline does not apply them | Native path applies no-speech, compression-ratio, and log-probability thresholds |
| Invalid VAD configuration | Warns and falls back to defaults | Fails startup for invalid configuration |
| Shared model state | Existing batched path | Shared transcription lock and restoration of temporary options |
| Observability | General Prometheus metrics | Adds decode-chunk and decode-failure counters and decoder logging |

The main fork change is `app/pipeline.py` (333 additions, 22 deletions in the comparison). In the [assessed snapshot](https://github.com/ShadyF/whisperx-asr-service/blob/1537c02/app/pipeline.py), configuration begins at line 92, native decoding at line 913, and the shared lock guards transcription at line 1053. `app/metrics.py` adds 14 lines for decoder counters. The compared changes in `app/main.py` and `app/serve_deployments.py` add decoder logging.

The fork's Dockerfiles pin WhisperX to `741f8defa185c41a0075876d6e3a6dbeaf342acf` and explicitly install `faster-whisper==1.2.1`. Upstream installs the floating `feat/pyannote-audio-4` branch. The fork publishes to GHCR; upstream publishes to Docker Hub. These differences affect reproducibility and image selection independently of VAD tuning.

Both already provide `/asr` and OpenAI-compatible APIs, diarization, speaker embeddings, speaker-based resegmentation and diarization tuning, CPU alignment, model default/alias/preload fixes, cache eviction, general Prometheus metrics, and simple serving. External transcription, Qwen alignment, and DGX-related support also exist upstream; they are not used by this deployment.

## Current local reconciliation

The local reconciliation adds `VAD_METHOD=pyannote|silero` to both decoder modes,
with `pyannote` as the default. Values are trimmed and lowercased; unset, empty,
or whitespace-only values select Pyannote. Invalid values fail startup under
the fork's strict configuration convention. This differs from upstream's
warning-and-fallback behavior.

Local startup validation requires `0 <= VAD_OFFSET <= VAD_ONSET` and
`0 < VAD_ONSET < 1`, matching the pinned WhisperX VAD's strict onset bounds.
Silero uses onset as its threshold and chunk size as its maximum speech
duration; offset is accepted and validated but has no effect on Silero
detection or merging. Offset tuning applies to Pyannote.

Silero first loads `snakers4/silero-vad` through `torch.hub`, using the Torch
Hub cache rather than the Hugging Face model download directory. Offline use
requires a preseeded cache and a writable, persisted Torch cache. The current
cluster configuration already sets `TORCH_HOME=/.cache/torch`.

Native decoding uses the selected WhisperX VAD followed by the cached
faster-whisper model. Method selection affects speech detection, not speaker
diarization. Existing chunk sizes, onset/offset values, decoder modes, and native
thresholds remain unchanged. Both Compose files pass through `VAD_METHOD` with
a `pyannote` default; their chunk default remains 20 seconds, while the
application default remains 30 seconds.

This local change has not been deployed: the verified live image remains
`ghcr.io/shadyf/whisperx-asr-service:0.4.5`. Silero is opt-in. No audio tests were
performed, and native/GPU quality remains unverified. The ancestry statistics
and runtime comparison above describe the 2026-10-12 assessment of `1537c02`,
not the reconciled working tree.

## What the upstream audio tests establish

The release author reports 36 GPU-tested files covering speech with gaps containing silence, room tone, hum, clicks, distant voices, or a noise floor raised by recorder automatic gain control, plus meeting excerpts. Default output matched v0.4.2 on all 36 files.

In those tests, invented gap text disappeared with `VAD_ONSET=0.8` or `VAD_METHOD=silero`, but not onset `0.6` or `0.7`. Onset `0.8` also lost a few percent of quiet real speech. Silero transcribed more distant voices. About a tenth of the wording elsewhere changed under either setting as speech was divided at different points.

The author reports more hallucinations with chunks shorter than 30 seconds in the upstream batched pipeline. That result does not establish that the fork's native 20-second path is worse: the decoder paths differ, and no equivalent A/B audio test has been performed here. The fork's focused unit tests use mocked components and establish code behavior, not transcription quality on real audio.

## Current deployment and consumer

The live image and environment were verified read-only against the homelab configuration. The service image and runtime settings are in `../k8s-homelab/cluster/apps/default/whisperx-asr-service/whisperx-asr-service-helm-release.yaml:57-88`; the init-container image is at lines 136-137.

| Setting | Verified value |
| --- | --- |
| Image | `ghcr.io/shadyf/whisperx-asr-service:0.4.5` |
| Decoder | `WHISPER_DECODE_MODE=native` |
| VAD | `VAD_CHUNK_SIZE=20`, `VAD_ONSET=0.500`, `VAD_OFFSET=0.363` |
| Decoder thresholds | `NO_SPEECH_THRESHOLD=0.80`, `COMPRESSION_RATIO_THRESHOLD=2.4`, `LOG_PROB_THRESHOLD=-0.8` |
| Devices and precision | `DEVICE=cuda`, `COMPUTE_TYPE=int8_float16`, `ALIGN_DEVICE=cpu` |
| Work scheduling | `BATCH_SIZE=2`, `GPU_CONCURRENCY=1`, `SERVE_MODE=simple` |
| Model loading | `PRELOAD_MODEL=large-v3` |
| Speaker processing | `RESEGMENT_BY_SPEAKER=true`, `DIARIZE_CLUSTERING_THRESHOLD=0.5` |

Speakr is configured to use the internal service URL with diarization and speaker embeddings enabled in `../k8s-homelab/cluster/apps/default/speakr/speakr-helm-release.yaml:68-77`. This establishes the configured consumer and required API behavior; it does not establish observed request traffic or successful audio processing.

The documented GPU is a GTX 1660 SUPER with 6 GiB VRAM on `k8-w5`. Upstream's default image uses the same Torch 2.7.1/CUDA 12.6 base choice as the fork, but that does not prove the upstream image works within this production GPU's capacity.

Docker Hub reports an active `linux/amd64` image for `learnedmachine/whisperx-asr-service:0.4.3`. Registry metadata checked on 2026-10-12 reports:

- OCI index digest: `sha256:d46cbc82ae9b2421c4412db6b18910e30ae21d4f823f61eb68c93f0a4e3768b1`.
- AMD64 manifest digest: `sha256:210adbcb36749b4c918e845b531abcf3ab0e9efd2404c052db349ce275ce09d4`.

These are registry metadata checks, not a successful pull or runtime test.

## Migration checks

Upstream would ignore `WHISPER_DECODE_MODE`, `VAD_CHUNK_SIZE`, and the three decoder-threshold environment variables. VAD onset and offset carry over, but transcription would use batched decoding and 30-second chunks. `BATCH_SIZE=2` would become active in batched decoding, unlike the current serial native path. Keeping the same environment would therefore not reproduce the current behavior.

1. Build a representative audio set containing silence, room noise, recorder automatic gain control, and quiet or distant real speech. Include long meetings and multilingual recordings if those are part of normal use.
2. Compare the current fork's native 20-second configuration with upstream Pyannote at 30 seconds and upstream Silero at 30 seconds. Test a higher Pyannote onset separately if needed, checking quiet-speech loss as well as hallucinations.
3. Check dropped words, invented text, speaker assignments, word timestamps, embeddings, and the `/asr` and OpenAI API contracts required by consumers. Record latency and peak VRAM and host RAM.
4. Validate the selected image with existing cache mounts, read-only root filesystem, probes, and metrics. Account for the init-container image as well as the main service image when planning a migration.
5. Stage without disrupting other workloads on the shared GPU. Check capacity before running simultaneous replicas or model instances. Change production only after the results meet acceptance criteria, retaining the current image and configuration for rollback.

## Sources

- [Upstream v0.4.3 release](https://github.com/murtaza-nasir/whisperx-asr-service/releases/tag/v0.4.3) and [latest-release API](https://api.github.com/repos/murtaza-nasir/whisperx-asr-service/releases/latest): publication date, VAD settings, and reported audio tests.
- [Official fork comparison](https://api.github.com/repos/murtaza-nasir/whisperx-asr-service/compare/v0.4.3...ShadyF:main): ancestry and changed-file statistics; the `main` target can move after this research date.
- [Upstream pipeline at v0.4.3](https://github.com/murtaza-nasir/whisperx-asr-service/blob/v0.4.3/app/pipeline.py#L124-L169): VAD options and warning/fallback behavior.
- [Upstream Dockerfile at v0.4.3](https://github.com/murtaza-nasir/whisperx-asr-service/blob/v0.4.3/Dockerfile): default Torch version and floating WhisperX dependency.
- [Fork pipeline at 1537c02](https://github.com/ShadyF/whisperx-asr-service/blob/1537c02/app/pipeline.py), [Dockerfile](https://github.com/ShadyF/whisperx-asr-service/blob/1537c02/Dockerfile), and [native tests](https://github.com/ShadyF/whisperx-asr-service/blob/1537c02/tests/test_pipeline_native.py): decoder implementation, dependency pins, and mocked test coverage.
- [Docker Hub tag API](https://hub.docker.com/v2/repositories/learnedmachine/whisperx-asr-service/tags/0.4.3): architecture, tag status, and image digests.
- [Pinned WhisperX Silero VAD](https://github.com/sealambda/whisperX/blob/741f8defa185c41a0075876d6e3a6dbeaf342acf/whisperx/vads/silero.py) and [base VAD](https://github.com/sealambda/whisperX/blob/741f8defa185c41a0075876d6e3a6dbeaf342acf/whisperx/vads/vad.py): Silero threshold/chunk behavior, Torch Hub loading, and strict onset bounds.
- Local homelab paths cited above: service configuration, init-container image, and Speakr consumer settings. Live image/environment checks confirmed the stated runtime configuration without an audio test.
