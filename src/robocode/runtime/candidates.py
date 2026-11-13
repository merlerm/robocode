"""Export editor submissions and validate their interfaces in a container."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import (
    EnvironmentArtifact,
    export_environment,
    load_environment_artifact,
)
from robocode.runtime.containers import container_process, image_identity


class CandidateSourceError(ValueError):
    """A submitted editor tree is missing the required simulator source."""


def export_candidate(
    workspace: Path, commit: str, parent: EnvironmentArtifact, destination: Path
) -> EnvironmentArtifact:
    """Export only the submitted source subtree, retaining the task configuration."""
    import_root = "source/" + parent.manifest["source"]["import_root"]
    entry = subprocess.check_output(
        ["git", "-C", str(workspace), "ls-tree", commit, import_root], text=True
    ).split()
    if not entry or entry[1] != "tree":
        raise CandidateSourceError(
            f"Submitted commit is missing directory {import_root}"
        )
    dependencies = parent.path / "dependencies"
    return export_environment(
        workspace,
        commit,
        destination,
        package=parent.manifest["source"]["package"],
        config=parent.manifest["config"],
        runtime_image=parent.manifest["runtime_image"],
        dependencies={
            str(p.relative_to(dependencies)): p
            for p in dependencies.rglob("*")
            if p.is_file()
        },
        import_root=parent.manifest["source"]["import_root"],
        source_subdir="source",
    )


def validate_environment(
    artifact: EnvironmentArtifact,
    directory: Path,
    *,
    backend: str,
    image: str,
    runtime_source: Path,
    expected: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    """Return candidate failures; raise infrastructure failures without accepting."""
    artifact = load_environment_artifact(
        artifact.path, expected_id=artifact.artifact_id
    )
    if image_identity(backend, image) != artifact.manifest["runtime_image"]:
        raise ValueError("Validation image differs from the environment artifact")
    directory = directory.resolve()
    directory.mkdir(parents=True)
    output = directory / "output"
    output.mkdir()
    with container_process(
        backend,
        image,
        directory,
        role="validation",
        worker=[
            "/robocode/.venv/bin/python",
            "-B",
            "-m",
            "robocode.runtime.validation_worker",
        ],
        mounts=[
            f"{artifact.path}:/artifact:ro",
            f"{runtime_source.resolve()}:/robocode/src:ro",
            f"{output}:/output:rw",
        ],
        env={
            "PYTHONPATH": "/artifact/source/"
            + artifact.manifest["source"]["import_root"]
            + ":/robocode/src",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MUJOCO_GL": "osmesa",
        },
        log_name="validation.log",
    ) as (process, _, log_thread):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if not (output / "started").exists():
                raise RuntimeError("Validation worker did not start") from None
            result = {"valid": False, "error": "Candidate validation timed out"}
        else:
            log_thread.join()
            if not (output / "started").exists():
                raise RuntimeError(
                    "Validation worker failed before probing: "
                    + (directory / "validation.log").read_text()
                )
            result = (
                json.loads((output / "report.json").read_text())
                if (output / "report.json").exists()
                else {
                    "valid": False,
                    "error": f"Candidate exited without a report ({process.returncode})",
                }
            )
    if result["valid"] and expected is not None and result["contract"] != expected:
        result.update(
            valid=False, error="Action/observation interface or count schedule changed"
        )
    result["environment_artifact_id"] = artifact.artifact_id
    (directory / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
