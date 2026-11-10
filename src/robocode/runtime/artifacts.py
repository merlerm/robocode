"""Committed environment exports and their content identities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a nonempty relative path: {value!r}")
    if ".git" in path.parts:
        raise ValueError("Git metadata does not belong in an environment export")
    return path


def _files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(
                f"Symlinks are unsupported in environment artifacts: {path}"
            )
        if relative == "manifest.json":
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Expected a regular file: {path}")
        files[relative] = {
            "sha256": _digest(path),
            "executable": bool(path.stat().st_mode & 0o111),
        }
    return files


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _export_commit(repo: Path, commit: str, destination: Path) -> None:
    entries = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-rz", commit]
    ).split(b"\0")
    if any(entry.startswith(b"160000 ") for entry in entries):
        raise ValueError(
            "Export the editable package repository, not a submodule parent"
        )
    # Keep potentially large archives on the destination filesystem, not /tmp.
    with tempfile.TemporaryFile(dir=destination.parent) as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", commit],
            stdout=stream,
            check=True,
        )
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            for member in archive:
                _relative_path(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported Git archive entry: {member.name}")
                archive.extract(member, destination, filter="data")


@dataclass(frozen=True)
class EnvironmentArtifact:
    """A verified export; keep it private or mount it read-only in workers."""

    path: Path
    artifact_id: str
    manifest: dict[str, Any]

    @property
    def import_root(self) -> Path:
        """Directory to prepend to a fresh worker's Python import path."""
        return self.path / "source" / self.manifest["source"]["import_root"]

    def verify_imports(self) -> dict[str, list[str]]:
        """Reject installed/cached copies of the package and its loaded submodules."""
        package = self.manifest["source"]["package"]
        if package not in sys.modules:
            raise ValueError(f"Environment package has not been imported: {package}")
        origins = {}
        root = self.import_root.resolve()
        for name, module in tuple(sys.modules.items()):
            if name != package and not name.startswith(package + "."):
                continue
            filename = getattr(module, "__file__", None)
            locations = ([filename] if filename else []) + list(
                getattr(module, "__path__", [])
            )
            paths = [Path(location).resolve() for location in locations]
            if not paths or any(not path.is_relative_to(root) for path in paths):
                raise ValueError(
                    f"Environment module {name} loaded from {locations}, expected {root}"
                )
            origins[name] = [str(path) for path in paths]
        return origins


def export_environment(
    repo: Path,
    revision: str,
    destination: Path,
    *,
    package: str,
    config: Mapping[str, Any],
    runtime_image: str,
    dependencies: Mapping[str, Path],
    assets: Mapping[str, Path] | None = None,
    import_root: str = "src",
) -> EnvironmentArtifact:
    """Export a commit, declared assets and lockfiles into a fresh directory.

    ``runtime_image`` is a resolved OCI image digest or SIF SHA256. The launcher
    must enforce that identity; this function does not run or inspect containers.
    Asset keys are destinations relative to the exported source tree.
    """
    repo = repo.resolve()
    destination = destination.resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("repo must be the root of the editable Git repository")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image):
        raise ValueError("runtime_image must be a resolved sha256 digest")
    if not package or not all(part.isidentifier() for part in package.split(".")):
        raise ValueError(f"Invalid environment package name: {package!r}")
    relative_import_root = _relative_path(import_root)
    commit = _git(
        repo, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"
    )
    tree = _git(repo, "rev-parse", commit + "^{tree}")
    # Validate/normalize the config before creating any files.
    resolved_config = json.loads(_json_bytes(dict(config)))
    destination.mkdir(parents=True)
    source = destination / "source"
    source.mkdir()
    _export_commit(repo, commit, source)
    if not (source / relative_import_root).is_dir():
        raise ValueError(
            f"Import root is absent from the committed tree: {import_root}"
        )

    for relative, origin in (assets or {}).items():
        target = source / _relative_path(relative)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Asset would replace committed source: {target}")
        if origin.is_symlink():
            raise ValueError(f"Asset must not be a symlink: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, symlinks=True)
        elif origin.is_file():
            shutil.copyfile(origin, target)
        else:
            raise ValueError(f"Expected an asset file or directory: {origin}")

    for name, origin in dependencies.items():
        target = destination / "dependencies" / _relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not origin.is_file():
            raise ValueError(f"Expected a dependency file: {origin}")
        with origin.open("rb") as source_stream, target.open("xb") as stream:
            shutil.copyfileobj(source_stream, stream)

    manifest = {
        "schema_version": 1,
        "source": {
            "commit": commit,
            "tree": tree,
            "package": package,
            "import_root": relative_import_root.as_posix(),
        },
        "config": resolved_config,
        "runtime_image": runtime_image,
        "assets": sorted((assets or {}).keys()),
        "files": _files(destination),
    }
    artifact_id = "sha256:" + hashlib.sha256(_json_bytes(manifest)).hexdigest()
    manifest["artifact_id"] = artifact_id
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return EnvironmentArtifact(destination, artifact_id, manifest)


def load_environment_artifact(
    path: Path, *, expected_id: str | None = None
) -> EnvironmentArtifact:
    """Verify identity and every exported file before handing an artifact to a
    worker."""
    path = path.resolve()
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Artifact manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_id = manifest.pop("artifact_id")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported environment artifact schema")
    actual_id = "sha256:" + hashlib.sha256(_json_bytes(manifest)).hexdigest()
    if artifact_id != actual_id or (
        expected_id is not None and artifact_id != expected_id
    ):
        raise ValueError("Environment artifact identity mismatch")
    _relative_path(manifest["source"]["import_root"])
    if _files(path) != manifest["files"]:
        raise ValueError("Environment artifact files differ from the manifest")
    manifest["artifact_id"] = artifact_id
    return EnvironmentArtifact(path, artifact_id, manifest)
