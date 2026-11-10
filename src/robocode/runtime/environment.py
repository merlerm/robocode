"""Environment server lifetime, owned by the experiment rather than an agent."""

from __future__ import annotations

import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from robocode.utils.telemetry import host_launch_env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentBinding:
    """Connection details passed to an agent; the token is omitted from logs."""

    service_id: str
    host: str
    port: int
    token: str = field(repr=False)
    strict: bool = False

    def client_host(self, container_backend: str) -> str:
        """Docker uses the host gateway; Apptainer installs a loopback relay."""
        return {
            "docker": "host.docker.internal",
            "apptainer": self.host,
            "local": self.host,
        }[container_backend]


def _describe_exit(code: int) -> str:
    """Include the signal name when the server was killed."""
    if code < 0:
        return f"was killed by {signal.Signals(-code).name} ({code})"
    return f"exited with code {code}"


def _signal_group(pid: int, sig: signal.Signals) -> None:
    """Signal the service and its children; an already-exited group is fine."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


@contextmanager
def environment_service(
    env_cfg_json: str,
    service_dir: Path,
    *,
    telemetry_sink: Path | None = None,
    telemetry_run_id: str | None = None,
    strict: bool = False,
    on_failure: Callable[[], None] | None = None,
) -> Iterator[EnvironmentBinding]:
    """Start a server in a new directory, yield its binding, and stop it on exit."""
    service_dir = service_dir.resolve()
    service_dir.mkdir(parents=True)  # Refuse to reuse another run's directory.
    config_path = service_dir / "env_server_config.json"
    config_path.write_text(env_cfg_json, encoding="utf-8")
    port_file = service_dir / "env_server_port"
    log_path = service_dir / "env_server.log"
    token = secrets.token_hex(16)
    server_env = dict(os.environ)
    server_env.pop("ROBOCODE_TELEMETRY", None)
    server_env.pop("ROBOCODE_RUN_ID", None)
    if telemetry_sink is not None:
        telemetry_sink.parent.mkdir(parents=True, exist_ok=True)
        server_env.update(
            host_launch_env(
                telemetry_sink, telemetry_run_id or service_dir.name, with_hook=False
            )
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robocode.utils.env_server_runtime",
                "--env-config",
                str(config_path),
                "--token",
                token,
                "--port-file",
                str(port_file),
                *(["--strict"] if strict else []),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=server_env,
            start_new_session=True,
        )
    died = threading.Event()
    stopping = threading.Event()
    watcher: threading.Thread | None = None

    def failure() -> RuntimeError:
        assert proc.returncode is not None
        return RuntimeError(
            f"env server {_describe_exit(proc.returncode)}; see {log_path}"
        )

    def watch() -> None:
        proc.wait()
        if not stopping.is_set():
            died.set()
            logger.error("%s", failure())
            if on_failure is not None:
                on_failure()

    try:
        deadline = time.monotonic() + 60
        while not port_file.exists():
            if proc.poll() is not None:
                raise RuntimeError(
                    f"env server exited at startup; see {log_path}:\n"
                    f"{log_path.read_text(encoding='utf-8')}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"env server did not start; see {log_path}")
            time.sleep(0.1)
        port = int(port_file.read_text(encoding="utf-8"))
        watcher = threading.Thread(
            target=watch, name="env-server-watchdog", daemon=True
        )
        watcher.start()
        yield EnvironmentBinding(service_dir.name, "127.0.0.1", port, token, strict)
        if proc.poll() is not None:
            raise failure()
    except KeyboardInterrupt:
        # A crash callback may interrupt the owner; report the crash, not Ctrl-C.
        if died.is_set():
            raise failure() from None
        raise
    finally:
        stopping.set()
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
        _signal_group(proc.pid, signal.SIGKILL)  # Stop surviving children too.
        if watcher is not None:
            watcher.join()
