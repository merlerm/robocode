"""Run held-out evaluation in a source-free container against a frozen service."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import EnvironmentArtifact
from robocode.runtime.containers import container_process, image_identity
from robocode.runtime.environment import environment_service
from robocode.runtime.snapshots import file_manifest, snapshot_policy, verify_snapshot
from robocode.utils.episode import _EPISODE_WALL_CAP_S
from robocode.utils.isolated_transport import UnixRelay

# Only generic harness code enters the blackbox evaluator.
_RUNTIME_FILES = (
    "approaches/base_approach.py",
    "environments/base_env.py",
    "environments/variable_count.py",
    "runtime/binding.py",
    "runtime/evaluation.py",
    "runtime/evaluation_worker.py",
    "runtime/remote_environment.py",
    "runtime/snapshots.py",
    "utils/episode.py",
    "utils/env_client.py",
    "utils/strict_blackbox.py",
    "utils/isolated_transport.py",
)


def evaluate_in_container(
    policy: Path,
    artifact: EnvironmentArtifact,
    output_dir: Path,
    *,
    backend: str,
    environment_image: str,
    evaluation_image: str,
    eval_seeds: list[int],
    replicate_seed: int,
    max_steps: int,
    timeout: float,
    render: bool,
    determinism_episodes: int,
) -> dict[str, Any]:
    """Freeze inputs, own the evaluator and its environment, and return scores."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    runtime = output_dir / "runtime"
    for name in _RUNTIME_FILES:
        target = runtime / "robocode" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    runtime_manifest = file_manifest(runtime)
    policy_manifest = snapshot_policy(policy, output_dir / "policy")
    for name, manifest in [
        ("policy", policy_manifest),
        ("runtime-source", runtime_manifest),
    ]:
        (output_dir / (name + ".json")).write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    evaluation_identity = image_identity(backend, evaluation_image)
    inputs, outputs = output_dir / "input", output_dir / "output"
    inputs.mkdir()
    outputs.mkdir()
    with (
        environment_service(
            json.dumps(artifact.manifest["config"]),
            output_dir / "environment",
            artifact=artifact,
            container_backend=backend,
            image=environment_image,
            strict=True,
        ) as binding,
        tempfile.TemporaryDirectory(prefix="rc-eval-", dir="/tmp") as temporary,
    ):
        bridge = Path(temporary)
        request = {
            "binding": {
                "service_id": "evaluation",
                "host": "127.0.0.1",
                "port": 8765,
                "token": binding.token,
                "strict": True,
                "artifact_id": binding.artifact_id,
                "metadata": {
                    "action_space": binding.metadata["action_space"],
                    "observation_space": binding.metadata["observation_space"],
                    "description": None,
                    "description_blackbox": None,
                },
            },
            "policy_id": policy_manifest["id"],
            "runtime_source_id": runtime_manifest["id"],
            "environment_artifact_id": artifact.artifact_id,
            "eval_seeds": eval_seeds,
            "replicate_seed": replicate_seed,
            "max_steps": max_steps,
            "timeout": timeout,
            "render": render,
            "determinism_episodes": determinism_episodes,
        }
        (inputs / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        (inputs / "transport.json").write_text(
            json.dumps(
                {
                    "listeners": [{"port": 8765, "socket": "/bridge/environment.sock"}],
                }
            )
        )
        with UnixRelay(
            str(bridge / "environment.sock"), (binding.host, binding.port)
        ) as relay:
            thread = threading.Thread(target=relay.serve_forever, daemon=True)
            thread.start()
            try:
                with container_process(
                    backend,
                    evaluation_identity if backend == "docker" else evaluation_image,
                    output_dir,
                    role="evaluation",
                    worker=[
                        "/usr/local/bin/python",
                        "-B",
                        "-m",
                        "robocode.utils.isolated_transport",
                        "/input/transport.json",
                        "/usr/local/bin/python",
                        "-B",
                        "-m",
                        "robocode.runtime.evaluation_worker",
                    ],
                    mounts=[
                        f"{runtime}:/runtime:ro",
                        f"{output_dir / 'policy'}:/policy:ro",
                        f"{inputs}:/input:ro",
                        f"{outputs}:/output:rw",
                        f"{bridge}:/bridge:ro",
                    ],
                    env={
                        "PYTHONPATH": "/runtime",
                        "HOME": "/tmp",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONUNBUFFERED": "1",
                    },
                    log_name="evaluation.log",
                ) as (proc, context, log_thread):
                    deadline = time.monotonic() + 60
                    while not (outputs / "ready").exists() and proc.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "Evaluation policy loading exceeded 60 seconds"
                            )
                        time.sleep(0.1)
                    episodes = len(eval_seeds) + 2 * max(
                        0, min(determinism_episodes, len(eval_seeds))
                    )
                    # Reset has its own 600-second RPC deadline before the rollout cap.
                    proc.wait(timeout=60 + episodes * (_EPISODE_WALL_CAP_S + 600 + 60))
                    log_thread.join()
                    if proc.returncode:
                        raise RuntimeError(
                            "Evaluation container failed:\n"
                            + (output_dir / "evaluation.log").read_text()
                        )
            finally:
                relay.shutdown()
                thread.join()
        results = json.loads((outputs / "results.json").read_text())
        for key, expected in [
            ("policy_artifact_id", policy_manifest["id"]),
            ("environment_artifact_id", artifact.artifact_id),
            ("evaluation_runtime_source_id", runtime_manifest["id"]),
        ]:
            if results[key] != expected:
                raise ValueError(f"Evaluation result identity mismatch: {key}")
        verify_snapshot(output_dir / "policy", policy_manifest["id"])
        verify_snapshot(runtime, runtime_manifest["id"])
        if (
            image_identity(
                backend,
                evaluation_identity if backend == "docker" else evaluation_image,
            )
            != evaluation_identity
        ):
            raise ValueError("Evaluation image changed during execution")
        if backend == "docker":
            results["evaluation_container_id"] = (
                (output_dir / "container.cid").read_text().strip()
            )
        results["evaluation_runtime_image"] = evaluation_identity
        results["evaluation_log_context"] = context
        results["evaluation_environment_runtime_source_id"] = binding.metadata[
            "runtime_source_id"
        ]
    return results
