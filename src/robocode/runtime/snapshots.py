"""Content identities for frozen policy files and evaluator source."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def file_manifest(root: Path) -> dict[str, Any]:
    """Hash names and bytes; reject links that could escape the snapshot."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Snapshot cannot contain symlinks: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    identity = (
        "sha256:"
        + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    )
    return {"id": identity, "files": files}


def snapshot_policy(source: Path, destination: Path) -> dict[str, Any]:
    """Freeze the submitted workspace without its development transport or caches."""
    if source.name != "approach.py" or not source.is_file():
        raise ValueError("Policy submission must name an existing approach.py")
    shutil.copytree(
        source.parent,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".mcp",
            ".codex",
            ".claude",
            "__pycache__",
            "*.pyc",
            "env_spaces.json",
            "env_client.py",
            "AGENTS.md",
            "CLAUDE.md",
            "mcp_renders",
            ".venv",
        ),
    )
    return file_manifest(destination)


def verify_snapshot(root: Path, expected_id: str) -> None:
    """Fail before execution if a submitted snapshot changed."""
    if file_manifest(root)["id"] != expected_id:
        raise ValueError(f"Snapshot identity mismatch: {root}")
