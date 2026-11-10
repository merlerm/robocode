"""Isolated Docker/Apptainer process lifetime and tagged output."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, TextIO

from robocode.utils.apptainer_environment import clean_apptainer_env

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
def container_process(
    backend: str,
    image: str,
    service_dir: Path,
    *,
    role: str,
    worker: list[str],
    mounts: list[str],
    env: dict[str, str],
    log_name: str,
    token: str | None = None,
) -> Iterator[tuple[subprocess.Popen[str], dict[str, Any], threading.Thread]]:
    """Run a worker without host networking, credentials or writable source."""
    identity = image_identity(backend, image)
    service_dir = service_dir.resolve()
    instance = "rc-" + role + "-" + uuid.uuid4().hex[:12]
    log_context = {
        "role": role,
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
    if backend == "docker":
        cmd = [
            "docker",
            "run",
            "--init",
            "--name",
            instance,
            "--label",
            f"robocode.role={role}",
            "--cidfile",
            str(service_dir / "container.cid"),
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
    public_cmd = [
        "<token>" if token is not None and arg == token else arg for arg in cmd
    ]
    (service_dir / "command.json").write_text(
        json.dumps(public_cmd, indent=2) + "\n", encoding="utf-8"
    )
    log_path = service_dir / log_name
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
        yield proc, log_context, log_thread
    finally:
        if backend == "docker" and (service_dir / "container.cid").exists():
            subprocess.run(
                [
                    "docker",
                    "rm",
                    "--force",
                    (service_dir / "container.cid").read_text().strip(),
                ],
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
        logger.info("[role=coordinator] Stopped %s exit=%s", prefix, proc.returncode)
