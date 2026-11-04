"""Publish only the isolated candidate tags using the OCI distribution API."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BRANCH = "refs/heads/feature/egyptian-turbo"
IMAGE = "ghcr.io/shadyf/whisperx-asr-service"
REPOSITORY = "shadyf/whisperx-asr-service"
ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def identity(ref, sha):
    # Limit all writes to the intended branch and a complete Git commit identity.
    if ref != BRANCH or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Candidate publication requires the candidate branch and full commit SHA")
    return "egyptian-turbo-" + sha


def manifest(token, reference):
    # Treat only explicit manifest absence as permission to create a new identity.
    request = Request(
        f"https://ghcr.io/v2/{REPOSITORY}/manifests/{reference}",
        headers={"Authorization": "Bearer " + token, "Accept": ACCEPT},
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
            digest = "sha256:" + hashlib.sha256(body).hexdigest()
            if response.headers.get("Docker-Content-Digest") != digest:
                raise ValueError("Registry manifest digest does not match its content")
            return digest, body, response.headers["Content-Type"].split(";")[0]
    except HTTPError as error:
        if error.code == 404:
            payload = json.loads(error.read())
            if payload.get("errors") and all(
                item.get("code") == "MANIFEST_UNKNOWN" for item in payload["errors"]
            ):
                return None
        raise


def put_manifest(token, tag, content):
    # Copy the exact manifest bytes so aliases retain the verified image digest.
    request = Request(
        f"https://ghcr.io/v2/{REPOSITORY}/manifests/{tag}",
        data=content[1],
        method="PUT",
        headers={"Authorization": "Bearer " + token, "Content-Type": content[2]},
    )
    with urlopen(request, timeout=60) as response:
        if response.headers.get("Docker-Content-Digest") != content[0]:
            raise ValueError("Published manifest digest changed")


def ensure_identity(token, tag, content):
    # Recheck immediately before mutation and never replace a published identity.
    existing = manifest(token, tag)
    if existing is not None:
        if existing[0] != content[0]:
            raise ValueError("Commit-specific candidate tag already has a different digest")
        return existing[0]

    # Create only an absent identity and confirm the resulting registry content.
    put_manifest(token, tag, content)
    if manifest(token, tag) != content:
        raise ValueError("Commit-specific candidate verification failed")
    return content[0]


def main():
    # Validate provenance before requesting registry write credentials.
    sha = os.environ["GITHUB_SHA"]
    tag = identity(os.environ["GITHUB_REF"], sha)
    mode = sys.argv[1]
    if mode not in ("resolve", "publish"):
        raise ValueError("Unknown publication mode")

    # Obtain a scoped native GHCR bearer token without exposing credentials in logs.
    credentials = base64.b64encode(
        (os.environ["GITHUB_ACTOR"] + ":" + os.environ["GH_TOKEN"]).encode()
    ).decode()
    request = Request(
        f"https://ghcr.io/token?service=ghcr.io&scope=repository:{REPOSITORY}:pull,push",
        headers={"Authorization": "Basic " + credentials},
    )
    with urlopen(request, timeout=60) as response:
        token = json.load(response)["token"]

    # Resolve an existing commit tag before any build; reruns reuse this digest.
    content = manifest(token, tag)
    if mode == "resolve":
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("digest=" + (content[0] if content else "") + "\n")
        return

    # Validate the digest produced by the build or reused from the registry.
    digest = os.environ["CANDIDATE_DIGEST"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid candidate digest")
    if content is None:
        content = manifest(token, digest)
    if content is None or content[0] != digest:
        raise ValueError("Candidate digest is missing or conflicts with the commit tag")
    ensure_identity(token, tag, content)

    # Stale runs may preserve their SHA identity but cannot promote the trial alias.
    head = subprocess.check_output(
        ["git", "ls-remote", "--exit-code", "origin", BRANCH], text=True
    ).split()[0]
    promoted = head == sha
    if promoted:
        put_manifest(token, "egyptian-turbo-test", content)
        if manifest(token, "egyptian-turbo-test") != content:
            raise ValueError("Trial alias verification failed")

    # Record the immutable pull reference even when a stale run skips promotion.
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(
            f"Source: `{sha}`\n\nImage: `{IMAGE}:{tag}`\n\n"
            f"Digest: `{IMAGE}@{digest}`\n\nTrial alias advanced: {promoted}\n"
        )


if __name__ == "__main__":
    # Run only when called by the branch-specific workflow.
    main()
