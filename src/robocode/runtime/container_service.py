"""Launch a frozen environment behind a local relay, without container networking."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, TextIO

from robocode.runtime.artifacts import EnvironmentArtifact, load_environment_artifact
from robocode.utils.apptainer_environment import clean_apptainer_env
from robocode.utils.isolated_transport import TCPRelay

logger = logging.getLogger(__name__)


def _copy_output(stream: TextIO, path: Path, prefix: str) -> None:
    """Tag raw output too, including library prints and traceback lines."""
    with stream, path.open("w", encoding="utf-8", buffering=1) as log:
        for line in stream:
            log.write(f"{prefix} {line}")


def image_identity(backend: str, image: str) -> str:
    """Resolve the actual local image used to run the environment."""
    if backend == "docker":
        return subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
        ).strip()
    if backend == "apptainer":
        with Path(image).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    raise ValueError(f"Unsupported environment container backend: {backend}")


@contextmanager
def container_service(
    artifact: EnvironmentArtifact,
    backend: str,
    image: str,
    service_dir: Path,
    token: str,
    *,
    strict: bool,
    telemetry_run_id: str | None = None,
    telemetry: bool = False,
) -> Iterator[tuple[subprocess.Popen, int, dict[str, Any]]]:
    """Yield the worker, host relay port and metadata after verified startup."""
    artifact = load_environment_artifact(
        artifact.path, expected_id=artifact.artifact_id
    )
    identity = image_identity(backend, image)
    if identity != artifact.manifest["runtime_image"]:
        raise ValueError("Environment runtime image does not match the artifact")
    source = Path(__file__).resolve().parents[1].parent
    service_dir = service_dir.resolve()
    instance = "rc-env-" + uuid.uuid4().hex[:12]
    log_context = {
        "role": "environment",
        "service": service_dir.name,
        "backend": backend,
        "instance": instance,
    }
    prefix = (
        "[" + " ".join(f"{key}={value}" for key, value in log_context.items()) + "]"
    )
    (service_dir / "log-context.json").write_text(
        json.dumps(log_context, indent=2) + "\n"
    )
    # Unix socket names must fit the kernel's short pathname limit.
    with tempfile.TemporaryDirectory(prefix="rc-env-", dir="/tmp") as temporary:
        bridge = Path(temporary)
        sockets = bridge / "sockets"
        sockets.mkdir()
        runtime_source = bridge / "runtime"
        shutil.copytree(
            source,
            runtime_source,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        files = {
            str(path.relative_to(runtime_source)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(runtime_source.rglob("*"))
            if path.is_file()
        }
        runtime_digest = (
            "sha256:"
            + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        )
        (service_dir / "runtime-source.json").write_text(
            json.dumps({"id": runtime_digest, "files": files}, indent=2) + "\n"
        )
        mounts = [
            f"{artifact.path}:/artifact:ro",
            f"{runtime_source}:/robocode/src:ro",
            f"{service_dir}:/service:rw",
            f"{sockets}:/bridge:rw",
        ]
        env = {
            "PYTHONPATH": "/artifact/source/"
            + artifact.manifest["source"]["import_root"]
            + ":/robocode/src",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "MUJOCO_GL": "osmesa",
            "HOME": "/tmp",
        }
        if telemetry:
            env.update(
                ROBOCODE_TELEMETRY="/service/telemetry.jsonl",
                ROBOCODE_RUN_ID=telemetry_run_id or service_dir.name,
            )
        worker = [
            "/robocode/.venv/bin/python",
            "-B",
            "-m",
            "robocode.utils.env_server_runtime",
            "--artifact",
            "/artifact",
            "--expected-id",
            artifact.artifact_id,
            "--token",
            token,
            "--socket",
            "/bridge/environment.sock",
            "--metadata-file",
            "/service/metadata.json",
            *(["--strict"] if strict else []),
        ]
        if backend == "docker":
            cmd = [
                "docker",
                "run",
                "--init",
                "--name",
                instance,
                "--label",
                "robocode.role=environment",
                "--cidfile",
                str(bridge / "cid"),
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,exec",
                "--workdir",
                "/tmp",
                "--entrypoint",
                worker[0],
            ]
            for mount in mounts:
                cmd += ["--volume", mount]
            for key, value in env.items():
                cmd += ["--env", f"{key}={value}"]
            cmd += [identity, *worker[1:]]
        else:
            cmd = [
                "apptainer",
                "exec",
                "--userns",
                "--net",
                "--network",
                "none",
                "--containall",
                "--pid",
                "--writable-tmpfs",
                "--no-home",
                "--cleanenv",
                "--pwd",
                "/tmp",
            ]
            for mount in mounts:
                cmd += ["--bind", mount]
            for key, value in env.items():
                cmd += ["--env", f"{key}={value}"]
            cmd += [
                str(Path(image).resolve()),
                "setpriv",
                "--bounding-set=-all",
                "--inh-caps=-all",
                "--ambient-caps=-all",
                "--no-new-privs",
                "--",
                *worker,
            ]
        public_cmd = ["<token>" if arg == token else arg for arg in cmd]
        (service_dir / "command.json").write_text(
            json.dumps(public_cmd, indent=2) + "\n", encoding="utf-8"
        )
        log_path = service_dir / "env_server.log"
        log_path.touch()
        logger.info("[role=coordinator] Starting %s", prefix)
        proc = subprocess.Popen(
            cmd,
            env=clean_apptainer_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        assert proc.stdout is not None
        log_thread = threading.Thread(
            target=_copy_output, args=(proc.stdout, log_path, prefix), daemon=True
        )
        log_thread.start()
        try:
            metadata_file = service_dir / "metadata.json"
            deadline = time.monotonic() + 60
            while not metadata_file.exists():
                if proc.poll() is not None:
                    log_thread.join()
                    raise RuntimeError(
                        "Environment container failed:\n"
                        + (service_dir / "env_server.log").read_text()
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("Environment container startup timed out")
                time.sleep(0.1)
            metadata = json.loads(metadata_file.read_text())
            metadata["runtime_source_id"] = runtime_digest
            metadata["log_context"] = log_context
            if backend == "docker":
                metadata["container_id"] = (bridge / "cid").read_text().strip()
            (service_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            if metadata["artifact_id"] != artifact.artifact_id:
                raise ValueError("Environment worker reported the wrong artifact")
            with TCPRelay(
                0,
                str(sockets / "environment.sock"),
                host="0.0.0.0" if backend == "docker" else "127.0.0.1",
            ) as relay:
                thread = threading.Thread(target=relay.serve_forever, daemon=True)
                thread.start()
                try:
                    yield proc, relay.server_address[1], metadata
                finally:
                    relay.shutdown()
                    thread.join()
        finally:
            if backend == "docker" and (bridge / "cid").exists():
                subprocess.run(
                    ["docker", "rm", "--force", (bridge / "cid").read_text().strip()],
                    check=True,
                    capture_output=True,
                )
            if proc.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            proc.wait()
            log_thread.join()
            logger.info(
                "[role=coordinator] Stopped %s exit=%s", prefix, proc.returncode
            )
            shutil.make_archive(
                str(service_dir / "runtime-source"), "gztar", runtime_source
            )
