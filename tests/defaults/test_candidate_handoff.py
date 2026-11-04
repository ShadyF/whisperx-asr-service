"""Check the copyable candidate selection without starting a service."""

import re
import textwrap
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_readme_candidate_override_selects_only_image_and_torch_cache():
    # Read the same override the operator copies from the handoff guide.
    readme = (ROOT / "README.md").read_text()
    match = re.search(
        r"cat > compose\.egyptian-trial\.yaml <<'YAML'\n(.*?)\n   YAML", readme, re.S
    )
    assert match, "README needs a copyable candidate Compose override"
    override = yaml.safe_load(textwrap.dedent(match.group(1)))

    # Keep model tuning in the existing environment recipe, not another preset.
    assert override == {
        "services": {
            "whisperx-asr": {
                "image": "${CANDIDATE_IMAGE:?Set CANDIDATE_IMAGE to the reviewed digest}",
                "environment": {"TORCH_HOME": "/.cache/torch"},
            }
        }
    }

    # The regular Compose file must continue selecting the stable image.
    baseline = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert baseline["services"]["whisperx-asr"]["image"] == (
        "ghcr.io/shadyf/whisperx-asr-service:latest"
    )
