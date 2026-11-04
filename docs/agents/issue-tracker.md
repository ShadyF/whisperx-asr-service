# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `ShadyF/whisperx-asr-service`. Use the `gh` CLI.

## Operations

- Verify the target repository before writes; use `--repo ShadyF/whisperx-asr-service` when needed to avoid upstream/fork ambiguity.
- Create issues with `gh issue create`; use `--body-file` for a saved Markdown spec.
- Read with `gh issue view --json number,title,body,labels,comments`; list with `gh issue list` and appropriate state/label filters.
- Comment with `gh issue comment`; change labels with `gh issue edit --add-label` or `--remove-label`.
- Before closing, post the resolution, then use `gh issue close`.
- Prefer native GitHub sub-issues and issue dependencies when available; otherwise record parent/blocker links in the issue body.

## Publishing specs

When a skill says to publish to the issue tracker, create a GitHub issue. Use the configured triage vocabulary rather than inventing status labels.

## Pull requests as a triage surface

**PRs as a request surface: no.**

Pull requests are not part of the external-request triage queue.
