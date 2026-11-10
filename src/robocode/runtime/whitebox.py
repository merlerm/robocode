"""Read-only source views derived from a verified environment and runtime snapshot."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import EnvironmentArtifact, load_environment_artifact
from robocode.runtime.snapshots import file_manifest
from robocode.utils.source_views import copy_environment_source, copy_runtime_source


@dataclass(frozen=True)
class WhiteboxSource:
    """A filtered source view, shared by synthesis and evaluation."""

    path: Path
    source_id: str
    manifest: dict[str, Any]

    @property
    def pythonpath(self) -> str:
        """Container import paths for the pinned wrapper and environment source."""
        return (
            "/whitebox/runtime:/whitebox/source/"
            + self.manifest["source"]["import_root"]
        )

    def verify_imports(self) -> dict[str, list[str]]:
        """Check the actual environment package origins against this view."""
        return EnvironmentArtifact(
            self.path, self.manifest["artifact_id"], self.manifest
        ).verify_imports()


def _identity(manifest: dict[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    )


def create_whitebox_source(
    artifact: EnvironmentArtifact,
    runtime_source: Path,
    destination: Path,
) -> WhiteboxSource:
    """Preserve source bytes while excluding existing solution/demo surfaces."""
    artifact = load_environment_artifact(
        artifact.path, expected_id=artifact.artifact_id
    )
    if artifact.manifest["source"]["package"] != "kinder":
        raise ValueError(
            "Whitebox artifact pilot currently requires the kinder package"
        )
    destination = destination.resolve()
    destination.mkdir(parents=True)
    copy_runtime_source(runtime_source, destination / "runtime")
    copy_environment_source(artifact.path / "source", destination / "source")
    shutil.copy2(
        Path(__file__).resolve().parents[3] / "docker/artifact-entrypoint.sh",
        destination / "entrypoint.sh",
    )
    manifest = {
        "entrypoint_sha256": hashlib.sha256(
            (destination / "entrypoint.sh").read_bytes()
        ).hexdigest(),
        "artifact_id": artifact.artifact_id,
        "runtime_image": artifact.manifest["runtime_image"],
        "runtime_source_id": file_manifest(runtime_source)["id"],
        "source": artifact.manifest["source"],
        "source_files": file_manifest(destination / "source"),
        "runtime_files": file_manifest(destination / "runtime"),
    }
    identity = _identity(manifest)
    (destination / "view.json").write_text(
        json.dumps({**manifest, "source_id": identity}, indent=2) + "\n",
        encoding="utf-8",
    )
    return WhiteboxSource(destination, identity, manifest)


def load_whitebox_source(path: Path, expected_id: str) -> WhiteboxSource:
    """Reject changed source or a view for a different submission."""
    manifest = json.loads((path / "view.json").read_text(encoding="utf-8"))
    identity = manifest.pop("source_id")
    if identity != expected_id or _identity(manifest) != identity:
        raise ValueError("Whitebox source identity mismatch")
    if (
        hashlib.sha256((path / "entrypoint.sh").read_bytes()).hexdigest()
        != manifest["entrypoint_sha256"]
    ):
        raise ValueError("Whitebox entrypoint changed")
    if (
        file_manifest(path / "source") != manifest["source_files"]
        or file_manifest(path / "runtime") != manifest["runtime_files"]
    ):
        raise ValueError("Whitebox source files changed")
    return WhiteboxSource(path, identity, manifest)


def main() -> None:
    """Verify source and import origins before launching the synthesis CLI."""
    view = load_whitebox_source(Path("/whitebox"), sys.argv[1])
    importlib.import_module(view.manifest["source"]["package"])
    provenance = {
        "artifact_id": view.manifest["artifact_id"],
        "whitebox_source_id": view.source_id,
        "imports": view.verify_imports(),
    }
    Path("/sandbox/source-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
