"""Launch a frozen environment behind a local relay, without container networking."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import EnvironmentArtifact, load_environment_artifact
from robocode.runtime.containers import container_process, image_identity
from robocode.utils.isolated_transport import TCPRelay


@contextmanager
def container_service(
    artifact: EnvironmentArtifact,
    backend: str,
    image: str,
    service_dir: Path,
    token: str,
    *,
    strict: bool,
    runtime_source: Path | None = None,
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
    source = runtime_source or Path(__file__).resolve().parents[1].parent
    service_dir = service_dir.resolve()
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
        try:
            with container_process(
                backend,
                identity if backend == "docker" else image,
                service_dir,
                role="environment",
                worker=worker,
                mounts=mounts,
                env=env,
                log_name="env_server.log",
                token=token,
            ) as (proc, log_context, log_thread):
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
                    metadata["container_id"] = (
                        (service_dir / "container.cid").read_text().strip()
                    )
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
            shutil.make_archive(
                str(service_dir / "runtime-source"), "gztar", runtime_source
            )
