"""Live checks for artifact-backed service boundaries and cleanup."""

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from robocode.runtime.artifacts import export_environment, load_environment_artifact
from robocode.runtime.container_service import image_identity
from robocode.runtime.environment import environment_service
from robocode.runtime.remote_environment import RemoteEnvironment


@pytest.fixture(params=["docker", "apptainer"])
def runtime(request):
    root = os.environ.get("ROBOCODE_SERVICE_TEST_ARTIFACTS")
    if root is None:
        pytest.skip("Set ROBOCODE_SERVICE_TEST_ARTIFACTS for live container checks")
    backend = request.param
    repo = Path(__file__).resolve().parents[2]
    image = (
        "robocode-sandbox:multi-agent-runtime"
        if backend == "docker"
        else str(repo / "robocode-sandbox.sif")
    )
    artifact = load_environment_artifact(Path(root) / (backend + "-artifact"))
    return backend, image, artifact


def test_metadata_rollout_and_exception_cleanup(runtime, tmp_path):
    """The worker supplies metadata and is removed even when its caller raises."""
    backend, image, artifact = runtime
    with pytest.raises(RuntimeError, match="caller stopped"):
        with environment_service(
            json.dumps(artifact.manifest["config"]),
            tmp_path / "service",
            artifact=artifact,
            container_backend=backend,
            image=image,
            strict=True,
        ) as binding:
            assert binding.artifact_id == artifact.artifact_id
            context = binding.metadata["log_context"]
            assert context["role"] == "environment"
            assert context["backend"] == backend
            if backend == "docker":
                assert len(binding.metadata["container_id"]) == 64
            assert binding.metadata["imports"]["kinder"][0].startswith(
                "/artifact/source/"
            )
            env = RemoteEnvironment(binding, 200)
            initial, _ = env.reset(seed=0)
            next_state, _, _, _, _ = env.step([0.05, 0, 0, 0, 0])
            assert next_state[0] - initial[0] == pytest.approx(0.05)
            assert env.render().shape == (750, 750, 3)
            env.close()
            raise RuntimeError("caller stopped")
    lines = (tmp_path / "service/env_server.log").read_text().splitlines()
    assert lines and all(line.startswith("[role=environment ") for line in lines)
    assert all(f"instance={context['instance']}]" in line for line in lines)
    assert not (tmp_path / "service" / "env_server_port").exists()
    command = json.loads((tmp_path / "service" / "command.json").read_text())
    assert binding.token not in command
    assert "--network" in command and "none" in command
    assert not Path(
        next(arg.split(":")[0] for arg in command if arg.endswith(":/bridge:rw"))
    ).exists()


def test_wrong_image_and_config_rejected(runtime, tmp_path, monkeypatch):
    backend, image, artifact = runtime
    config = json.dumps(artifact.manifest["config"])
    with pytest.raises(ValueError, match="configuration does not match"):
        with environment_service(
            "{}",
            tmp_path / "config",
            artifact=artifact,
            container_backend=backend,
            image=image,
        ):
            pytest.fail("Started with wrong config")
    monkeypatch.setattr(
        "robocode.runtime.container_service.image_identity",
        lambda *_: "sha256:" + "0" * 64,
    )
    with pytest.raises(ValueError, match="runtime image does not match"):
        with environment_service(
            config,
            tmp_path / "image",
            artifact=artifact,
            container_backend=backend,
            image=image,
        ):
            pytest.fail("Started with wrong image")
    assert not (tmp_path / "image" / "command.json").exists()


def test_failed_startup_keeps_error_and_cleans_container(runtime, tmp_path):
    """An unusable package fails startup; cleanup must not mask the worker error."""
    backend, image, _ = runtime
    repo = tmp_path / "broken-repo"
    (repo / "src/broken_env").mkdir(parents=True)
    (repo / "src/broken_env/__init__.py").write_text(
        'raise RuntimeError("broken environment")\n'
    )
    for args in [
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Runtime test",
            "-c",
            "user.email=runtime-test@example.invalid",
            "commit",
            "-qm",
            "broken fixture",
        ),
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    config = {"_target_": "broken_env.Environment"}
    artifact = export_environment(
        repo,
        "HEAD",
        tmp_path / "artifact",
        package="broken_env",
        config=config,
        runtime_image=image_identity(backend, image),
        dependencies={},
    )
    with pytest.raises(RuntimeError, match="Environment container failed"):
        with environment_service(
            json.dumps(config),
            tmp_path / "service",
            artifact=artifact,
            container_backend=backend,
            image=image,
        ):
            pytest.fail("Bad environment started")
    lines = (tmp_path / "service/env_server.log").read_text().splitlines()
    assert any("Traceback" in line for line in lines)
    assert all(
        line.startswith(f"[role=environment service=service backend={backend} ")
        for line in lines
    )
    command = json.loads((tmp_path / "service/command.json").read_text())
    assert not Path(
        next(arg.split(":")[0] for arg in command if arg.endswith(":/bridge:rw"))
    ).exists()


def test_worker_death_notifies_owner(runtime, tmp_path, monkeypatch):
    """A lost container process must fail the owning experiment and be cleaned up."""
    import signal
    import threading

    backend, image, artifact = runtime
    processes = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        if args[0][:2] in [["docker", "run"], ["apptainer", "exec"]]:
            processes.append(process)
        return process

    monkeypatch.setattr("robocode.runtime.container_service.subprocess.Popen", start)
    died = threading.Event()
    with pytest.raises(RuntimeError, match="env server"):
        with environment_service(
            json.dumps(artifact.manifest["config"]),
            tmp_path / "service",
            artifact=artifact,
            container_backend=backend,
            image=image,
            strict=True,
            on_failure=died.set,
        ):
            assert len(processes) == 1
            if backend == "docker":
                command = json.loads((tmp_path / "service/command.json").read_text())
                cid = Path(command[command.index("--cidfile") + 1]).read_text().strip()
                subprocess.run(["docker", "kill", cid], check=True, capture_output=True)
            else:
                os.killpg(processes[0].pid, signal.SIGKILL)
            assert died.wait(10)
    assert processes[0].poll() is not None


def test_service_telemetry_appends_to_shared_sink(tmp_path, monkeypatch):
    """Service events join agent-hook events already in the run's sink."""

    @contextmanager
    def fake_service(_artifact, _backend, _image, service_dir, *_args, **_kwargs):
        proc = subprocess.Popen(["sleep", "30"])
        try:
            yield proc, 1, {}
        finally:
            proc.kill()
            proc.wait()
            (service_dir / "telemetry.jsonl").write_text('{"source": "service"}\n')

    monkeypatch.setattr("robocode.runtime.environment.container_service", fake_service)
    sink = tmp_path / "telemetry" / "events.jsonl"
    sink.parent.mkdir()
    sink.write_text('{"source": "agent"}\n')
    artifact = SimpleNamespace(manifest={"config": {}}, artifact_id="sha256:test")
    with environment_service(
        "{}",
        tmp_path / "service",
        artifact=artifact,
        container_backend="docker",
        image="unused",
        telemetry_sink=sink,
    ):
        pass
    assert sink.read_text() == '{"source": "agent"}\n{"source": "service"}\n'
