"""Candidate export fidelity and validation failure classification."""

# pylint: disable=missing-function-docstring,unused-import,redefined-outer-name,protected-access
import json
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from robocode.runtime import candidates
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.editor import prepare_editor_workspace
from robocode.utils.sandbox import _initial_commit, _setup_sandbox_dir
from robocode.utils.sandbox_types import SandboxConfig
from tests.runtime.test_whitebox import fixture_source


def test_export_committed_source_and_assets_only(source, tmp_path):
    _, parent, _, view = source
    workspace = tmp_path / "workspace"
    prepare_editor_workspace(view, workspace)
    (workspace / "source/src/kinder/texture.png").write_bytes(b"simulator asset")
    (workspace / "probe.png").write_bytes(b"temporary render")
    config = SandboxConfig(
        sandbox_dir=workspace, environment_editor=SimpleNamespace(), turn_id="e1"
    )
    _setup_sandbox_dir(config)
    _initial_commit(workspace)
    commit = subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
    ).strip()
    (workspace / "source/src/kinder/__init__.py").write_text("unsubmitted edit")
    artifact = candidates.export_candidate(
        workspace, commit, parent, tmp_path / "candidate"
    )
    assert artifact.manifest["source"]["commit"] == commit
    assert artifact.manifest["source"]["subdir"] == "source"
    assert (artifact.import_root / "kinder/__init__.py").read_text() == "VALUE = 1\n"
    assert (
        artifact.import_root / "kinder/texture.png"
    ).read_bytes() == b"simulator asset"
    assert not (artifact.path / "source/probe.png").exists()
    assert not (artifact.path / "source/environment-origin.json").exists()
    load_environment_artifact(parent.path)
    load_environment_artifact(artifact.path)


@pytest.mark.parametrize("failure", ["timeout", "before_start", "schema", "invalid"])
def test_validation_reports_candidate_vs_infrastructure(
    source, tmp_path, monkeypatch, failure
):
    _, artifact, runtime, _ = source
    directory = tmp_path / "validation"
    monkeypatch.setattr(
        candidates, "image_identity", lambda *_: artifact.manifest["runtime_image"]
    )

    @contextmanager
    def process(*_args, **_kwargs):
        if failure != "before_start":
            (directory / "output/started").touch()
        (directory / "validation.log").write_text("worker log")

        def wait(timeout):
            if failure == "timeout":
                raise subprocess.TimeoutExpired("worker", timeout)
            (directory / "output/report.json").write_text(
                json.dumps(
                    {
                        "valid": failure != "invalid",
                        "contract": {"different": True},
                        "error": "SyntaxError",
                    }
                )
            )

        yield SimpleNamespace(wait=wait, returncode=1), None, SimpleNamespace(
            join=lambda: None
        )

    monkeypatch.setattr(candidates, "container_process", process)
    kwargs = {
        "backend": "docker",
        "image": "image",
        "runtime_source": runtime,
        "expected": {"original": True},
    }
    if failure == "before_start":
        with pytest.raises(RuntimeError, match="before probing"):
            candidates.validate_environment(artifact, directory, **kwargs)
    else:
        result = candidates.validate_environment(artifact, directory, **kwargs)
        assert not result["valid"]
        assert result["environment_artifact_id"] == artifact.artifact_id
        assert json.loads((directory / "result.json").read_text()) == result


def test_missing_committed_source_is_a_candidate_error(source, tmp_path):
    repo, parent, _, _ = source
    with pytest.raises(candidates.CandidateSourceError, match="source/src"):
        candidates.export_candidate(repo, "HEAD", parent, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()
