"""Revision-bound source views must survive edits and reject mismatched inputs."""

import asyncio
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box

from robocode.approaches.agentic_approach import AgenticApproach
from robocode.runtime.artifacts import export_environment
from robocode.runtime.binding import EnvironmentBinding
from robocode.runtime.container_evaluation import evaluate_in_container
from robocode.runtime.snapshots import snapshot_runtime
from robocode.runtime.whitebox import create_whitebox_source, load_whitebox_source
from robocode.utils.apptainer_sandbox import (
    ApptainerSandboxConfig,
    run_agent_in_apptainer_sandbox,
)
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.docker_sandbox import (
    DockerSandboxConfig,
    run_agent_in_docker_sandbox,
)

# pylint: disable=missing-function-docstring


@pytest.fixture(name="source")
def fixture_source(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "src/kinder"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (repo / "demos").mkdir()
    (repo / "demos/solution.py").write_text("hidden = True\n")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "E0",
        ],
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    artifact = export_environment(
        repo,
        "HEAD",
        tmp_path / "artifact",
        package="kinder",
        config={},
        runtime_image="sha256:" + "1" * 64,
        dependencies={},
    )
    runtime = tmp_path / "runtime"
    snapshot_runtime(runtime)
    view = create_whitebox_source(artifact, runtime, tmp_path / "view")
    return repo, artifact, runtime, view


def test_view_freezes_source_and_filters_solutions(source):
    repo, artifact, runtime, view = source
    (repo / "src/kinder/__init__.py").write_text("VALUE = 999\n")
    assert (view.path / "source/src/kinder/__init__.py").read_text() == "VALUE = 1\n"
    assert not (view.path / "source/demos").exists()
    assert not (view.path / "runtime/robocode/oracles").exists()
    assert not (view.path / "runtime/robocode/approaches/agentic_base.py").exists()
    assert (view.path / "runtime/robocode/environments/kinder_geom2d_env.py").exists()
    assert (
        load_whitebox_source(view.path, view.source_id).manifest["artifact_id"]
        == artifact.artifact_id
    )
    assert runtime.exists()


@pytest.mark.parametrize(
    "target",
    [
        "source/src/kinder/__init__.py",
        "runtime/robocode/environments/kinder_geom2d_env.py",
        "entrypoint.sh",
    ],
)
def test_changed_view_fails(source, target):
    view = source[3]
    with (view.path / target).open("a") as stream:
        stream.write("\n# modified\n")
    with pytest.raises(ValueError, match="changed"):
        load_whitebox_source(view.path, view.source_id)


def test_wrong_revision_fails(source):
    view = source[3]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_whitebox_source(view.path, "sha256:" + "0" * 64)


def test_evaluation_rejects_different_wrapper_before_launch(source, tmp_path):
    _, artifact, runtime, view = source
    (runtime / "robocode/environments/kinder_geom2d_env.py").write_text("changed\n")
    with pytest.raises(ValueError, match="wrapper source"):
        evaluate_in_container(
            tmp_path / "approach.py",
            artifact,
            tmp_path / "evaluation",
            backend="docker",
            environment_image="unused",
            evaluation_image="unused",
            eval_seeds=[0],
            replicate_seed=0,
            max_steps=1,
            timeout=1,
            render=False,
            determinism_episodes=0,
            runtime_source=runtime,
            whitebox_source=view,
        )


def test_committed_edit_gets_a_new_source_view(source, tmp_path):
    repo, artifact, runtime, first = source
    (repo / "src/kinder/__init__.py").write_text("VALUE = 2\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qam",
            "E1",
        ],
        check=True,
    )
    edited = export_environment(
        repo,
        "HEAD",
        tmp_path / "edited",
        package="kinder",
        config={},
        runtime_image=artifact.manifest["runtime_image"],
        dependencies={},
    )
    second = create_whitebox_source(edited, runtime, tmp_path / "edited-view")
    assert first.source_id != second.source_id
    assert (first.path / "source/src/kinder/__init__.py").read_text() == "VALUE = 1\n"
    assert (second.path / "source/src/kinder/__init__.py").read_text() == "VALUE = 2\n"
    with pytest.raises(ValueError, match="does not match evaluation environment"):
        evaluate_in_container(
            tmp_path / "approach.py",
            edited,
            tmp_path / "evaluation",
            backend="docker",
            environment_image="unused",
            evaluation_image="unused",
            eval_seeds=[0],
            replicate_seed=0,
            max_steps=1,
            timeout=1,
            render=False,
            determinism_episodes=0,
            runtime_source=runtime,
            whitebox_source=first,
        )


@pytest.mark.parametrize("backend", ["docker", "apptainer"])
def test_blackbox_rejects_source_before_launch(source, tmp_path, backend):

    options = {
        "sandbox_dir": tmp_path / "sandbox",
        "blackbox": True,
        "whitebox_source": source[3],
    }
    if backend == "docker":
        config = DockerSandboxConfig(**options)
        launch = run_agent_in_docker_sandbox
    else:
        sif = tmp_path / "unused.sif"
        sif.touch()
        config = ApptainerSandboxConfig(**options, sif_path=sif)
        launch = run_agent_in_apptainer_sandbox
    with pytest.raises(ValueError, match="Blackbox sandboxes cannot receive"):
        asyncio.run(launch(config, SimpleNamespace(name="codex")))


def test_synthesis_rejects_different_binding(source):

    with pytest.raises(ValueError, match="does not match synthesis environment"):
        AgenticApproach(
            action_space=Box(-1, 1, (5,), dtype=np.float32),
            observation_space=Box(-1, 1, (5,), dtype=np.float32),
            seed=0,
            whitebox_source=source[3],
            primitives={},
            backend=DEFAULT_BACKEND_CFG,
            container_backend="docker",
            environment_binding=EnvironmentBinding(
                "test", "127.0.0.1", 1, "unused", artifact_id="another-revision"
            ),
        )
