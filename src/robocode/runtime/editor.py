"""Mutable simulator workspace and frozen-policy access for an editor turn."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from robocode.runtime.snapshots import verify_snapshot
from robocode.runtime.whitebox import WhiteboxSource, load_whitebox_source

if TYPE_CHECKING:
    from robocode.runtime.policy_service import PolicyBinding


def prepare_editor_workspace(source: WhiteboxSource, workspace: Path) -> None:
    """Seed a new workspace from filtered source, without upstream Git history."""
    source = load_whitebox_source(source.path, source.source_id)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.path / "source", workspace / "source")
    (workspace / "environment-origin.json").write_text(
        json.dumps(
            {
                "environment_id": source.manifest["artifact_id"],
                "source_view_id": source.source_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EditorBinding:
    """Read-only runtime and policy service paired with mutable workspace source."""

    source: WhiteboxSource
    policy: PolicyBinding
    policy_source: Path | None = None

    @property
    def pythonpath(self) -> str:
        """Load the edited simulator before any image-installed package."""
        return (
            "/sandbox/source/"
            + self.source.manifest["source"]["import_root"]
            + ":/robocode/src"
        )

    @property
    def mounts(self) -> list[str]:
        """Extra mounts; only whitebox editors receive the frozen policy files."""
        mounts = [f"{self.source.path / 'entrypoint.sh'}:/editor-entrypoint.sh:ro"]
        if self.policy_source is not None:
            mounts.append(f"{self.policy_source.resolve()}:/frozen-policy:ro")
        return mounts

    def prepare(
        self, workspace: Path, runtime_image: str, host: str, port: int
    ) -> None:
        """Check pinned inputs and refresh the untracked client address each turn."""
        load_whitebox_source(self.source.path, self.source.source_id)
        if runtime_image != self.source.manifest["runtime_image"]:
            raise ValueError("Editor image differs from the environment artifact")
        if not (workspace / "source").is_dir():
            raise ValueError("Prepare the editor workspace before launching a turn")
        if self.policy_source is not None:
            verify_snapshot(self.policy_source, self.policy.policy_artifact_id)
        metadata = {**self.policy.client_metadata(), "host": host, "port": port}
        (workspace / "policy_service.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with (workspace / ".git/info/exclude").open("a", encoding="utf-8") as handle:
            handle.write("\npolicy_service.json\n")
