"""Check editor revision mismatches and refresh without starting containers."""

# pylint: disable=missing-function-docstring,unused-import,redefined-outer-name
import asyncio
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from robocode.runtime.editor import EditorBinding, prepare_editor_workspace
from robocode.runtime.policy_service import PolicyBinding
from robocode.runtime.snapshots import file_manifest
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.sandbox import _setup_sandbox_common
from tests.runtime.test_whitebox import fixture_source


@pytest.fixture(name="editor")
def fixture_editor(source, tmp_path):
    view = source[3]
    workspace = tmp_path / "workspace"
    prepare_editor_workspace(view, workspace)
    _setup_sandbox_common(workspace, {})
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "approach.py").write_text("# frozen policy\n")
    binding = PolicyBinding(
        "127.0.0.1", 1234, "first-test-token", file_manifest(policy)["id"], {}, 30
    )
    return EditorBinding(view, binding, policy), workspace


def test_resume_refreshes_policy_connection_without_resetting_edits(editor):
    binding, workspace = editor
    source = workspace / "source/src/kinder/__init__.py"
    source.write_text("VALUE = 2\n")
    binding.prepare(
        workspace,
        binding.source.manifest["runtime_image"],
        "host.docker.internal",
        1234,
    )
    new = replace(
        binding, policy=replace(binding.policy, port=5678, token="second-test-token")
    )
    new.prepare(workspace, binding.source.manifest["runtime_image"], "127.0.0.1", 5678)
    meta = json.loads((workspace / "policy_service.json").read_text())
    assert (meta["host"], meta["port"], meta["token"]) == (
        "127.0.0.1",
        5678,
        "second-test-token",
    )
    assert source.read_text() == "VALUE = 2\n"
    assert not (workspace / "source/.git").exists()
    assert not (workspace / "source/demos").exists()
    assert (
        json.loads((workspace / "environment-origin.json").read_text())[
            "environment_id"
        ]
        == binding.source.manifest["artifact_id"]
    )
    subprocess.run(
        ["git", "-C", str(workspace), "check-ignore", "policy_service.json"],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("changed", ["image", "policy", "source", "runtime"])
def test_mismatched_frozen_input_fails(editor, changed):
    binding, workspace = editor
    image = binding.source.manifest["runtime_image"]
    if changed == "image":
        image = "different-image"
    else:
        target = {
            "policy": binding.policy_source / "approach.py",
            "source": binding.source.path / "source/src/kinder/__init__.py",
            "runtime": binding.source.path
            / "runtime/robocode/environments/kinder_geom2d_env.py",
        }[changed]
        target.write_text("changed\n")
    with pytest.raises(ValueError):
        binding.prepare(workspace, image, "127.0.0.1", 1234)
    assert not (workspace / "policy_service.json").exists()


@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("conflict", ["blackbox", "whitebox_source"])
def test_policy_source_modes_cannot_be_combined_with_editor(
    editor, tmp_path, runtime, conflict
):
    binding, workspace = editor
    options = {
        "sandbox_dir": workspace,
        "environment_editor": binding,
        conflict: True if conflict == "blackbox" else binding.source,
    }
    if runtime == "docker":
        config = docker_sandbox.DockerSandboxConfig(**options)
        run = docker_sandbox.run_agent_in_docker_sandbox
    else:
        image = tmp_path / "unused.sif"
        image.touch()
        config = apptainer_sandbox.ApptainerSandboxConfig(**options, sif_path=image)
        run = apptainer_sandbox.run_agent_in_apptainer_sandbox
    with pytest.raises(ValueError, match="cannot be combined"):
        asyncio.run(run(config, SimpleNamespace(name="claude")))
