# Egyptian Turbo integration and isolated candidate image

## Problem Statement

The user prefers the Egyptian Whisper Turbo transcription samples for Egyptian Arabic mixed with English technical terms and wants this ASR service to use that model. The current service defaults to Whisper large-v3.

The integration must preserve the service's existing API and capabilities, dynamically obtain and cache its model rather than bundle weights into Docker images, and retain automatic language detection. The user wants to try a separately tagged candidate image from a separate branch before approving a merge into main.

The user explicitly does not want further model benchmarks, transcription-quality evaluations, or inference tests. The requested work is service integration and candidate-image delivery, not another model-selection experiment.

## Solution

Support opt-in Egyptian Turbo selection through the existing Whisper backend. The deployer selects the model using environment variables. Obtain its pinned original weights dynamically, convert them once on CPU into the CTranslate2 format required by that backend, and reuse the completed conversion from persistent model-cache storage.

Expose any necessary missing environment controls and document the complete selected Egyptian decoding and voice-activity-detection configuration in the README. Keep existing application, image and Compose defaults unchanged, along with automatic language detection, explicit request overrides, alignment, diarization and response contracts. This is the same application with an opt-in model, not a permanently separate product or a new ASR backend.

Implement in an isolated worktree on `feature/egyptian-turbo`. Publish only candidate image tags: `ghcr.io/shadyf/whisperx-asr-service:egyptian-turbo-test` and a commit-specific `egyptian-turbo-<commit>` tag. Do not publish stable aliases, merge into main, issue a normal release, or change the live deployment. The user will test the candidate before deciding whether to merge and subsequently release it normally.

## User Stories

### Model selection and service compatibility

1. As a service operator, I want to select Egyptian Turbo through environment configuration, so that I can opt in without changing existing defaults for other deployments.
2. As an existing API client, I want the request and response contracts to remain compatible, so that adopting the candidate does not require rewriting my integration.
3. As a service user, I want automatic language detection to remain the default, so that I do not have to supply a language for every request.
4. As a service user, I want an explicitly supplied language to remain effective, so that I can request Arabic or another supported language deliberately.
5. As a service user, I want large-v3 to remain selectable through the existing model configuration and request mechanisms, so that I retain control over which model is used.
6. As a service user, I want existing alignment and diarization capabilities to remain available, so that choosing Egyptian Turbo does not remove service features.
7. As a service user, I want existing explicit context and processing overrides to retain their documented behavior, so that opting into Egyptian does not silently discard my request settings.

### Deployment configuration

8. As a user of Egyptian Arabic–English technical conversations, I want a complete recommended decoding and VAD environment configuration, so that I can apply the settings behind the samples I preferred.
9. As a service operator, I want existing application, container and Compose defaults preserved with explicit override precedence, so that opting in remains my deployment decision.
10. As a service operator, I want configuration overrides to remain supported, so that I can intentionally change deployment behavior without building another application.
11. As a service user, I want the language-detection choice documented separately from the benchmark preset, so that I understand why automatic detection is not an exact reproduction of a run that explicitly requested Arabic.

### Dynamic model preparation and cache reuse

12. As a service operator, I want model weights downloaded dynamically rather than included in the image, so that model delivery retains the existing operational pattern.
13. As a service operator, I want the Egyptian source revision pinned, so that an uncached start does not silently obtain a different model revision.
14. As a service operator, I want the original weights converted on CPU once, so that no separately hosted converted-model repository or GPU conversion is required.
15. As a service operator, I want completed conversions reused from persistent cache storage, so that restarts do not repeatedly download and convert the model.
16. As a service operator, I want preparation to follow the existing preload and lazy-load behavior, so that enabling startup preload prepares the model at startup while lazy loading prepares it on first use.
17. As a service operator, I want concurrent model loads to coordinate preparation, so that multiple workers do not publish conflicting or incomplete cache entries.
18. As a service operator, I want interrupted or invalid conversions rejected, so that incomplete files are never mistaken for a usable model.
19. As a service operator, I want download and conversion failures reported clearly without silently substituting large-v3, so that I know which model actually serves my requests.
20. As a service operator, I want first-load time, CPU/RAM needs, writable-cache requirements, and cache permissions documented, so that I can provision the candidate correctly.
21. As a maintainer, I want model provenance and the publisher's license declaration retained, so that the conversion's source and conditions are traceable.

### Candidate delivery and review

22. As a repository owner, I want implementation isolated on a separate branch and worktree, so that unrelated changes in the existing main checkout remain untouched.
23. As a repository owner, I want a dedicated candidate Docker tag, so that I can pull the experimental application independently of the stable image.
24. As a repository owner, I want a commit-specific candidate tag and recorded image identity, so that my feedback refers to an identifiable build.
25. As a production operator, I want candidate publishing unable to overwrite stable tags or trigger a normal release, so that building the branch does not change production artifacts.
26. As a repository owner, I want software-only verification using controlled substitutes, so that integration errors can be caught without more model downloads or inference as tests.
27. As a repository owner, I want instructions for selecting the candidate image and returning to the existing image, so that I can conduct my own trial deliberately.
28. As a repository owner, I want merging, stable release, and live deployment to require later explicit approval, so that publishing a candidate is not treated as approval to promote it.

## Implementation Decisions

### Existing backend and API

- Reuse the current Whisper/WhisperX/Faster-Whisper backend and model-selection interfaces. Do not create an Egyptian backend, parallel API, or permanently separate image product.
- Keep existing model and tuning defaults unchanged. Support explicit Egyptian selection and existing large-v3 selection through the current configuration, request interfaces and allowlist behavior.
- Retain existing request shapes, response shapes, model-listing semantics, alignment and diarization capabilities, and documented explicit overrides. New controls are limited to necessary gaps in environment configuration; do not hardcode an Egyptian preset.
- Preserve automatic language detection for requests that omit language. Continue honoring an explicit `language=ar` or another supported language. Do not introduce Arabic as the implicit request language.
- The benchmark's disabled alignment and diarization flags describe its ASR-only evaluation scope; they are not new service defaults that remove or bypass those capabilities.

### Source model and preparation

- Use `mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching`, pinned to revision `2d9f6140a9a4e340541a3183e1700bafb336fd75`.
- The source repository provides Transformers weights, not the CTranslate2 artifact expected by the existing backend. Passing that source identifier unchanged to the native CT2 downloader is insufficient.
- Add only the preparation integration needed for this Egyptian model. Keep ordinary existing Whisper model downloads on their current path; do not build a general conversion framework for arbitrary models.
- Dynamically download the pinned source with the existing model-cache conventions, convert it on CPU using the existing Transformers/CTranslate2 toolchain, and load the completed local artifact through the existing Whisper model loader.
- Match the recorded conversion choices: float16 source loading, low-CPU-memory loading, `trust_remote_code=False`, CTranslate2 `int8_float16` quantization, and native processor/tokenizer export. The recorded conversion environment used Transformers 5.20.1 and CTranslate2 4.8.3; maintain a compatible, explicitly recorded conversion identity rather than silently reusing incompatible cached output.
- The recorded source checkpoint SHA-256 is `50a9a1cba2ca8c4889b2cc4fcfb7b6b294aecc4d900868da5a462923ff420364`. Validate pinned source identity and record conversion provenance and output integrity.
- The existing tested conversion is a provenance/reference artifact, not an image layer, a new public download dependency, or an item to check into Git. It contains seven files totaling 819,062,453 bytes. Do not promise byte-identical regenerated output across environments without establishing deterministic conversion.
- Do not publish a new Hugging Face repository or adopt an unverified third-party CT2 conversion as a substitute.
- Preparation occurs when the Egyptian model is loaded: during existing startup preloading when enabled, otherwise on first use. Preserve the existing preload/lazy-load controls rather than forcing eager downloads for every deployment.
- Cache identity must distinguish the source revision, conversion settings, and conversion compatibility. Publish a completed artifact atomically only after required files and recorded integrity checks pass.
- Coordinate concurrent preparation across processes sharing the cache. Never expose a partial conversion as the completed model. A failed download or conversion must not cause an implicit large-v3 fallback or broad cache deletion.
- Reuse valid completed cache entries without conversion. Respect the existing writable cache location, user permissions, and offline expectations; if preparation is required but unavailable, report that failure clearly.
- Keep recordings, transcripts, prompts, and credentials out of the model-artifact cache. Preserve the publisher's MIT declaration, source revision, and relevant attribution/provenance without claiming an independent legal audit of dataset rights.

### Recommended opt-in ASR configuration

The following values belong in a complete copyable README environment recipe. They are not replacement application, Docker or Compose defaults. Reuse existing environment controls and expose new ones only where a necessary setting is missing.

- Native decoding through the existing backend.
- CUDA compute type `int8_float16`, retaining explicit configuration overrides and existing device controls.
- Silero VAD; 20-second chunks; onset 0.5. No separate Silero offset override was part of the selected case.
- Beam size 8; best-of 5.
- Temperature fallback sequence: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
- Previous-text conditioning disabled.
- No default initial prompt or context. Explicit client-supplied context retains its existing supported behavior.
- No-speech threshold 0.8; log-probability threshold -0.8; compression-ratio threshold 2.4.
- Preserve existing application, image and Compose defaults and make environment/request precedence explicit. The deployer applies the README recipe, including when deployment configuration lives outside this repository.
- These recommended values come from the selected Egyptian f01 ASR-only case, not the earlier Egyptian full-pipeline baseline. Automatic language detection intentionally differs from f01's explicit Arabic hint. Do not promise identical transcription samples from different language or feature choices.

### Branch and candidate publication

- Isolate implementation on `feature/egyptian-turbo` in a separate worktree. Do not move, include, overwrite, or commit unrelated changes from the existing dirty main checkout.
- Keep candidate builds on a separate branch-specific publishing path using the existing registry and standard Linux amd64 image platform/toolchain.
- Publish `egyptian-turbo-test` and a commit-specific `egyptian-turbo-<commit>` tag under `ghcr.io/shadyf/whisperx-asr-service`. Record the resulting image digest/build identity for the user's trial.
- Candidate builds must not publish `latest`, stable semantic-version tags, Blackwell aliases, or normal release artifacts. The existing version-tag workflow unconditionally publishes `latest`, so it must not be reused by creating a candidate `v*` tag.
- Do not bundle model weights in the image, add a separate model-hosting account, or introduce a permanent Egyptian image variant.
- Preserve existing stable publishing behavior outside the candidate path. A future merge, normal release, and live deployment are separate decisions; none is authorized by candidate publication.
- Document candidate image selection and the previous-image rollback choice without editing or deploying live homelab configuration.

## Testing Decisions

- Test observable service behavior and configuration contracts rather than private helper structure. Favor existing test seams and existing substitutes for WhisperX, model loading, and external dependencies.
- The primary seam is the existing service/model-selection boundary: exercise request/default selection through the established stubbed service and pipeline tests. Verify that the Egyptian selection reaches a prepared local artifact, explicit large-v3 selection still works, language detection remains automatic, and explicit language/context/feature requests keep their contracts.
- Use a narrow filesystem preparation seam only where the service seam cannot demonstrate cache lifecycle behavior. Substitute downloader and converter operations with tiny deterministic fixtures; exercise real cache locking and atomic-publication behavior where practical.
- Cover first preparation, valid cache reuse, simultaneous loads, interruption, incomplete/corrupt cache rejection, pinned-source/provenance changes, and clear failure without silent model substitution. Do not execute the real converter or download weights to perform these checks.
- Reuse prior art from existing local-directory model-selection tests, decoder/VAD tests, documented-default tests, and stub-server end-to-end tests.
- Verify unchanged baseline defaults, explicitly configured Egyptian values and override precedence across application/container/Compose/documentation. Do not test that a particular transcription is accurate or that output exactly matches a saved sample.
- Check candidate workflow behavior for the intended branch/tags and prove stable aliases and release-trigger paths are excluded. Build/publish the candidate only through the approved candidate path.
- Use the repository's established Docker/Compose workflow for development verification. Do not install project dependencies into the base OpenCode environment or the user's workstation.
- No real model initialization/inference, ASR benchmark, quality evaluation, GPU smoke, or conversion-as-test is included. Software tests must use controlled substitutes and require no GPU or model-weight download. The user's later manual candidate trial is distinct from this automated verification scope.
- No claim of new accuracy, performance, full-pipeline quality, or GPU-fit validation may be inferred from software tests or a successful image build.

## Out of Scope

- Further model comparisons, benchmark runs, inference tests, quality scoring, or tuning sweeps.
- Integrating Audar, Meta Omnilingual, or another ASR model/backend.
- New API routes, response formats, changed model/tuning/language defaults, or removal of existing alignment/diarization features.
- Bundled model weights, a separately hosted converted-model repository, or generic conversion support for arbitrary model families.
- Automatic fallback to large-v3 after an Egyptian preparation failure.
- A permanently separate Egyptian product, additional candidate hardware/platform variants, or unrelated dependency/backend refactors.
- Moving existing unrelated work into the candidate branch.
- Merging into main, creating a stable release, overwriting stable image tags, editing live Kubernetes configuration, or deploying/replacing the current production service.
- Cleanup or deletion of earlier benchmark evidence or retained model-cache volumes.

## Further Notes

- The user explicitly confirmed environment-driven opt-in integration: preserve capabilities and existing defaults, put the complete recommended Egyptian configuration in the README, retain automatic language detection and dynamic downloads/cache, convert the original model once rather than host/bundle the converted artifact, isolate the branch/image, and perform no further model tests.
- Implementation follows child issues #2, #3 and #4 in dependency order. Their latest-scope clarification supersedes default-changing statements in the original published parent issue; the parent issue's body remains unchanged.
- First preparation requires network access, CPU/RAM, and several GB of writable temporary/cache capacity for original weights, conversion output, and staging. The converted artifact alone is approximately 819 MB. Document measured/derived provisioning requirements without claiming a nearly full existing cache will suffice.
- The preferred text samples are previous tuned ASR-only benchmark outputs, not the current production configuration. They are motivation for the user's choice, not a new accuracy guarantee or a test fixture that must be reproduced by real inference.
- No project glossary or ADR was found during discovery. Terms here follow existing service usage: model selection, model preparation, model cache, transcription, alignment, diarization, and candidate release. This spec does not introduce new domain entities or require an ADR for a reversible preparation choice.
- Track this spec in GitHub Issues for `ShadyF/whisperx-asr-service`, using the configured `ready-for-agent` label. Repository agent instructions define the tracker, triage vocabulary and single-context domain-document conventions. Implementation, candidate publication, merging and live deployment remain subject to the scope boundaries above.
