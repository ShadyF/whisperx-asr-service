"""Software-only checks for candidate registry identity and workflow isolation."""

import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import yaml


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
TAG = "egyptian-turbo-" + SHA
BODY = b'{"schemaVersion":2,"manifests":[]}'
DIGEST = "sha256:" + hashlib.sha256(BODY).hexdigest()
MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"


class Registry:
    """Tiny transport substitute retaining exact manifest bytes and writes."""

    def __init__(self):
        # Keep each test's manifests, writes and failures independent.
        self.manifests = {DIGEST: BODY}
        self.writes = []
        self.failure = None
        self.failure_reference = None
        self.missing = {}
        self.after_write_missing = {}
        self.reads = []
        self.timeouts = []
        self.bad_header = set()
        self.clock = 0
        self.clock_step = 0
        self.sleeps = []

    def monotonic(self):
        # Allow deadline tests to advance time without waiting in real time.
        self.clock += self.clock_step
        return self.clock

    def sleep(self, seconds):
        # Retain requested waits and advance only the controlled test clock.
        self.sleeps.append(seconds)
        self.clock += seconds

    def open(self, request, timeout):
        # Serve scoped authentication separately from registry manifest requests.
        if request.full_url.startswith("https://ghcr.io/token?"):
            return io.BytesIO(b'{"token":"fake"}')

        # Identify the manifest requested by the publication guard.
        reference = request.full_url.rsplit("/", 1)[1]

        # Count reads and their transport budgets independently of registry results.
        if request.get_method() == "GET":
            self.reads.append(reference)
            self.timeouts.append(timeout)

        # Simulate transport failures before reading or changing registry state.
        if self.failure is not None and self.failure_reference in (None, reference):
            raise self.failure

        # Retain published bytes so later requests can verify the same content.
        if request.get_method() == "PUT":
            self.writes.append((reference, request.data))
            self.manifests[reference] = request.data
            self.missing[reference] = self.after_write_missing.get(reference, 0)

        # Report explicit manifest absence separately from transport failures.
        if request.get_method() == "GET" and self.missing.get(reference, 0):
            self.missing[reference] -= 1
            raise HTTPError(
                request.full_url, 404, "missing", {},
                io.BytesIO(b'{"errors":[{"code":"MANIFEST_UNKNOWN"}]}'),
            )
        if reference not in self.manifests:
            raise HTTPError(
                request.full_url, 404, "missing", {},
                io.BytesIO(b'{"errors":[{"code":"MANIFEST_UNKNOWN"}]}'),
            )

        # Return the content-derived digest required by the real integrity guard.
        body = self.manifests[reference]
        response = io.BytesIO(body)
        response.headers = {
            "Docker-Content-Digest": "sha256:" + hashlib.sha256(body).hexdigest(),
            "Content-Type": MEDIA_TYPE,
        }

        # Simulate a response whose integrity header does not match its bytes.
        if reference in self.bad_header:
            response.headers["Docker-Content-Digest"] = "sha256:" + "0" * 64
        return response


class CandidatePublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the publication guard without running its command-line entry point.
        spec = importlib.util.spec_from_file_location(
            "candidate_publish", ROOT / "scripts/candidate_publish.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_main(self, registry, directory, mode, head=SHA):
        # Substitute only network transport and the remote branch lookup.
        environment = {
            "GITHUB_REF": self.module.BRANCH,
            "GITHUB_SHA": SHA,
            "GITHUB_ACTOR": "actor",
            "GH_TOKEN": "fake",
            "CANDIDATE_DIGEST": DIGEST,
            "GITHUB_OUTPUT": str(Path(directory) / "output"),
            "GITHUB_STEP_SUMMARY": str(Path(directory) / "summary"),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment))
            stack.enter_context(patch.object(self.module.sys, "argv", ["guard", mode]))
            stack.enter_context(patch.object(self.module, "urlopen", registry.open))
            stack.enter_context(patch.object(self.module.time, "monotonic", registry.monotonic))
            stack.enter_context(patch.object(self.module.time, "sleep", registry.sleep))
            stack.enter_context(patch.object(
                self.module.subprocess, "check_output",
                return_value=head + "\t" + self.module.BRANCH,
            ))
            self.module.main()

    def test_first_publication_and_rerun(self):
        # First resolve authorizes a build; publication copies exact manifest bytes.
        registry = Registry()
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(registry, directory, "resolve")
            self.assertEqual((Path(directory) / "output").read_text(), "digest=\n")
            self.run_main(registry, directory, "publish")
            self.assertEqual(registry.writes, [(TAG, BODY), ("egyptian-turbo-test", BODY)])
            summary = (Path(directory) / "summary").read_text()
            self.assertIn(SHA, summary)
            self.assertIn(DIGEST, summary)
            self.assertIn("Trial alias advanced: True", summary)

            # Reruns resolve the original digest and never rewrite its SHA tag.
            (Path(directory) / "output").unlink()
            registry.writes.clear()
            self.run_main(registry, directory, "resolve")
            self.assertEqual(
                (Path(directory) / "output").read_text(), "digest=" + DIGEST + "\n"
            )
            self.run_main(registry, directory, "publish")
            self.assertEqual(registry.writes, [("egyptian-turbo-test", BODY)])

    def test_digest_temporarily_absent_then_visible_publishes_exact_bytes(self):
        # A temporarily invisible built digest must still publish both exact aliases.
        registry = Registry()
        registry.missing[DIGEST] = 1
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(registry, directory, "publish")
            self.assertEqual(registry.reads.count(DIGEST), 2)
            self.assertEqual(registry.sleeps, [1])
            self.assertEqual(registry.writes, [(TAG, BODY), ("egyptian-turbo-test", BODY)])
            self.assertEqual(registry.manifests[TAG], BODY)
            self.assertEqual(registry.manifests["egyptian-turbo-test"], BODY)

    def test_conflicting_identity_has_no_writes(self):
        # A different digest at the commit tag blocks every mutation.
        registry = Registry()
        registry.manifests[TAG] = b'{"schemaVersion":2,"different":true}'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.run_main(registry, directory, "publish")
        self.assertEqual(registry.writes, [])

    def test_digest_absence_is_bounded_by_attempts_and_deadline(self):
        # Both a stationary clock and a jumping clock must terminate without writes.
        for step, expected_reads, expected_sleeps in [
            (0, 5, 5), (2, 1, 1), (20, 0, 0), (-1, 6, 5),
        ]:
            with self.subTest(clock_step=step):
                registry = Registry()
                registry.missing[DIGEST] = 100
                registry.clock_step = step
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ValueError, "Built digest: manifest remains absent"):
                        self.run_main(registry, directory, "publish")
                self.assertEqual(registry.reads.count(DIGEST), expected_reads)
                self.assertEqual(len(registry.sleeps), expected_sleeps)
                self.assertEqual(registry.writes, [])
                self.assertTrue(all(
                    0 < timeout <= 5
                    for reference, timeout in zip(registry.reads, registry.timeouts)
                    if reference == DIGEST
                ))
                if step == 0:
                    self.assertLessEqual(registry.clock, 5)

    def test_digest_wrong_content_and_header_fail_without_retry(self):
        # A present wrong body or malformed integrity header is never readiness lag.
        for bad_header in [False, True]:
            with self.subTest(bad_header=bad_header):
                registry = Registry()
                if bad_header:
                    registry.bad_header.add(DIGEST)
                else:
                    registry.manifests[DIGEST] = b'{"different":true}'
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        self.run_main(registry, directory, "publish")
                self.assertEqual(registry.reads.count(DIGEST), 1)
                self.assertEqual(registry.sleeps, [])
                self.assertEqual(registry.writes, [])

    def test_just_written_tags_wait_for_exact_content(self):
        # Each successful PUT may take one additional read to become visible.
        registry = Registry()
        registry.after_write_missing = {TAG: 1, "egyptian-turbo-test": 1}
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(registry, directory, "publish")
        self.assertEqual(registry.writes, [(TAG, BODY), ("egyptian-turbo-test", BODY)])
        self.assertEqual(registry.sleeps, [1, 1])
        self.assertTrue(all(timeout <= 5 for timeout in registry.timeouts if timeout != 60))

    def test_just_written_tags_fail_closed_on_absence_or_wrong_content(self):
        # Validate both publication stages without replacing the real read helpers.
        for tag, stage in [(TAG, "immutable tag"), ("egyptian-turbo-test", "trial alias")]:
            for failure in ["absent", "wrong", "header"]:
                with self.subTest(tag=tag, failure=failure):
                    class ChangedRegistry(Registry):
                        def open(self, request, timeout):
                            # Change only verification GET responses after the PUT.
                            if (
                                request.get_method() == "GET"
                                and request.full_url.endswith("/manifests/" + tag)
                                and any(reference == tag for reference, _ in self.writes)
                            ):
                                if failure == "wrong":
                                    self.manifests[tag] = b'{"different":true}'
                                elif failure == "header":
                                    self.bad_header.add(tag)

                            # Keep normal HTTP response integrity for other requests.
                            return super().open(request, timeout)

                    # Missing writes may wait, but wrong content must stop immediately.
                    registry = ChangedRegistry()
                    if failure == "absent":
                        registry.after_write_missing[tag] = 100
                    message = "digest does not match" if failure == "header" else stage
                    with tempfile.TemporaryDirectory() as directory:
                        with self.assertRaisesRegex(ValueError, message):
                            self.run_main(registry, directory, "publish")
                    self.assertEqual(registry.writes.count((tag, BODY)), 1)
                    if tag == TAG:
                        self.assertNotIn(("egyptian-turbo-test", BODY), registry.writes)
                    if failure != "absent":
                        self.assertEqual(registry.sleeps, [])

    def test_registry_failures_have_no_writes(self):
        # Authentication, network and ambiguous absence failures cannot enable writes.
        failures = [URLError("network unavailable")]
        for status, code in [
            (401, "UNAUTHORIZED"), (403, "DENIED"),
            (500, "UNKNOWN"), (404, "NAME_UNKNOWN"),
        ]:
            failures.append(HTTPError(
                "https://ghcr.io", status, "failure", {},
                io.BytesIO(json.dumps({"errors": [{"code": code}]}).encode()),
            ))

        # Exercise both command paths with the real registry error handling.
        for failure in failures:
            for mode in ["resolve", "publish"]:
                with self.subTest(failure=str(failure), mode=mode):
                    registry = Registry()
                    registry.failure = failure
                    if isinstance(failure, HTTPError):
                        failure.fp.seek(0)
                    with tempfile.TemporaryDirectory() as directory:
                        with self.assertRaises(URLError):
                            self.run_main(registry, directory, mode)
                    self.assertEqual(registry.writes, [])

    def test_digest_transport_failures_are_not_retried(self):
        # Only missing manifests qualify for readiness; all transport errors stop.
        cases = [(401, "UNAUTHORIZED"), (403, "DENIED"),
                 (500, "UNKNOWN"), (404, "NAME_UNKNOWN"), (None, None)]
        for status, code in cases:
            with self.subTest(status=status):
                registry = Registry()
                registry.failure_reference = DIGEST
                registry.failure = URLError("network unavailable")
                if status is not None:
                    registry.failure = HTTPError(
                        "https://ghcr.io", status, "failure", {},
                        io.BytesIO(json.dumps({"errors": [{"code": code}]}).encode()),
                    )
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(URLError):
                        self.run_main(registry, directory, "publish")
                self.assertEqual(registry.reads.count(DIGEST), 1)
                self.assertEqual(registry.sleeps, [])
                self.assertEqual(registry.writes, [])

    def test_stale_run_preserves_identity_without_alias(self):
        # A stale run still records its immutable image but does not promote it.
        registry = Registry()
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(registry, directory, "publish", head="c" * 40)
            self.assertEqual(registry.writes, [(TAG, BODY)])
            self.assertEqual(registry.manifests[TAG], BODY)
            self.assertNotIn("egyptian-turbo-test", registry.manifests)
            summary = (Path(directory) / "summary").read_text()
            self.assertIn(SHA, summary)
            self.assertIn(DIGEST, summary)
            self.assertIn("Trial alias advanced: False", summary)

    def test_branch_and_sha_guard(self):
        # Tags, other branches and malformed identities cannot publish candidate tags.
        for ref, sha in [
            ("refs/tags/v1", SHA), ("refs/heads/main", SHA),
            (self.module.BRANCH, "abc"),
        ]:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.module.identity(ref, sha)
        self.assertEqual(self.module.identity(self.module.BRANCH, SHA), TAG)

    def test_manifest_content_must_match_digest(self):
        # A corrupt registry response cannot establish image identity.
        with patch.object(self.module, "urlopen") as request:
            response = request.return_value.__enter__.return_value
            response.read.return_value = BODY
            response.headers = {"Docker-Content-Digest": "sha256:" + "a" * 64}
            with self.assertRaises(ValueError):
                self.module.manifest("fake", "tag")

    def test_workflow_contract(self):
        # Parse strings literally so YAML's legacy boolean rules cannot rename on.
        workflow = yaml.load(
            (ROOT / ".github/workflows/egyptian-candidate.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(workflow["on"], {"push": {"branches": ["feature/egyptian-turbo"]}})
        self.assertEqual(workflow["concurrency"], {
            "group": "egyptian-candidate-publication", "cancel-in-progress": "false",
        })
        self.assertEqual(workflow["permissions"], {"contents": "read", "packages": "write"})
        self.assertEqual(set(workflow["jobs"]), {"candidate"})
        job = workflow["jobs"]["candidate"]
        self.assertEqual(job["if"],
                         "github.event_name == 'push' && github.ref == "
                         "'refs/heads/feature/egyptian-turbo'")

        # Pin the standard toolchain and prohibit tag-based build publication.
        builds = [step for step in job["steps"]
                  if step.get("uses", "").startswith("docker/build-push-action@")]
        self.assertEqual(len(builds), 1)
        build = builds[0]
        self.assertEqual(build["if"], "steps.existing.outputs.digest == ''")
        self.assertNotIn("tags", build["with"])
        self.assertEqual(build["with"]["context"], ".")
        self.assertEqual(build["with"]["file"], "./Dockerfile")
        self.assertEqual(build["with"]["platforms"], "linux/amd64")
        self.assertEqual(build["with"]["outputs"],
                         "type=image,name=ghcr.io/shadyf/whisperx-asr-service,"
                         "push-by-digest=true,name-canonical=true,push=true")
        self.assertEqual(build["with"]["build-args"].splitlines(), [
            "TORCH_VERSION=2.7.1",
            "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126",
        ])

        # Ensure both lifecycle commands remain wired to the publication guard.
        commands = [step.get("run", "") for step in job["steps"]]
        self.assertTrue(any("python3 scripts/candidate_publish.py resolve" in command
                            for command in commands))
        self.assertTrue(any("python3 scripts/candidate_publish.py publish" in command
                            for command in commands))
