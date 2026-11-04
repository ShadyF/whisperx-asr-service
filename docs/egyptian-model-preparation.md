# Egyptian Turbo model preparation

Select `mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching`
through the existing model request field or `DEFAULT_MODEL`. To load it during
startup, set `PRELOAD_MODEL` to that identifier too. An existing `ALLOWED_MODELS`
restriction still applies. No application, Docker or Compose defaults change.

The existing Whisper backend receives a local CTranslate2 artifact. Preparation
runs only when this model is loaded: during configured startup preload or on
first use. Other Whisper models retain their existing download path. Requests
without a language still use automatic detection; explicit language, context,
alignment and diarization options retain their existing behavior.

## Source and conversion

The source is pinned to revision
`2d9f6140a9a4e340541a3183e1700bafb336fd75`. Preparation verifies the
`model.safetensors` checkpoint SHA-256 against
`50a9a1cba2ca8c4889b2cc4fcfb7b6b294aecc4d900868da5a462923ff420364`
before conversion. Downloads occur at runtime, not during image builds.

The native CTranslate2 Transformers converter loads on CPU with float16 source
loading, low-memory loading and remote code disabled, then exports
`int8_float16` output. Processor and tokenizer assets use native Transformers
export APIs. The historical reference used Transformers 5.20.1 and CTranslate2
4.8.3; each cache entry records the actual installed converter and related
library versions. Different versions receive separate cache identities rather
than assuming equivalent output.

Each artifact includes `provenance.json` with source identity, conversion
settings, dependency versions, source/output hashes and the publisher's MIT
declaration and attribution. `SOURCE_README.md` retains the pinned model card.
This records the publisher's declaration, not an independent audit of dataset
rights. Regenerated output is not promised to be byte-identical to the reference.

## Persistent storage and operation

Persist the existing `CACHE_DIR` (default `/.cache`). Completed artifacts live
under its `egyptian-turbo/` directory. The service user needs read/write and
directory traversal permissions; workers sharing this cache should use the same
user because preparation locks are private to their owner. The filesystem must
support POSIX advisory locks and atomic directory renames.

First preparation needs network access, CPU and RAM for the original checkpoint
and conversion, and several GB of free storage for source and staged output.
The reference converted artifact alone was approximately 819 MB. First-load
duration and peak RAM have not been measured for this service integration;
allow for download and CPU conversion rather than normal cached startup time.
Source staging is removed after preparation. Changing conversion identity
creates a new artifact while retaining existing completed entries.

Concurrent processes coordinate through a cache lock. Only a complete artifact
with verified output hashes is atomically published. Subsequent loads verify
its integrity and reuse it without downloading or converting, including offline
when the same dependency versions are installed. An uncached offline load fails.

Download, permissions, conversion and integrity failures report an Egyptian
preparation error; there is no automatic model substitution. Existing startup
preload behavior logs failures and leaves loading to a later request. A corrupt
completed entry is rejected and retained for operator inspection, not silently
deleted or repaired. Abrupt termination can leave an ignored `.prepare-*`
directory; inspect and remove only abandoned staging directories when no
preparation is active. Do not remove a live lock file or unrelated cache data.

## Verification boundary

Software checks use tiny fake downloads and converters with real filesystem
locking and publication. No model weights, real conversion, inference, GPU,
quality evaluation or benchmark is part of this verification. Runtime
conversion compatibility, first-load resource use and hardware fit remain
unverified. The full opt-in decoding/VAD environment recipe is maintained with
the deployment configuration documentation.
