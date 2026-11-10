"""Environment service ownership, separate from any agent session."""

from __future__ import annotations

import fcntl
import json
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
    """An explicitly owned live service; credentials never appear in its repr."""

    service_id: str
    host: str
    port: int
    token: str = field(repr=False)
    strict: bool = False

    def client_host(self, container_backend: str) -> str:
        """Address this service through the sandbox transport's explicit relay."""
        if container_backend == "docker":
            return "host.docker.internal"
        return self.host


def _describe_exit(code: int | None) -> str:
    """Render a subprocess exit status, naming the signal when it was killed.

    A bare number is not enough to triage this: ``-11`` means the server crashed
    natively, while a plain non-zero code means it exited on its own.
    """
    if code is None:
        return "is still running"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "unknown signal"
        return f"was killed by {name} ({code})"
    return f"exited with code {code}"


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
    """Own a service in its private directory and stop it on every exit path.

    The caller chooses its lifetime and optional failure callback. No agent
    workspace is read or mounted here. Different services require distinct dirs.
    """
    parent = service_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".owner.lock").open("w", encoding="utf-8") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Environment service directory already in use: {parent}"
            ) from exc
        config_path = parent / "env_server_config.json"
        config_path.write_text(env_cfg_json, encoding="utf-8")
        port_file = parent / "env_server_port"
        port_file.unlink(missing_ok=True)
        log_path = parent / "env_server.log"
        token = secrets.token_hex(16)

        server_env = dict(os.environ)
        if telemetry_sink is not None:
            # The owner routes events to the experiment's existing telemetry sink.
            telemetry_sink.parent.mkdir(parents=True, exist_ok=True)
            server_env.update(
                host_launch_env(
                    telemetry_sink, telemetry_run_id or parent.name, with_hook=False
                )
            )
        else:
            # Make the flag authoritative: an inherited ROBOCODE_TELEMETRY must not
            # enable telemetry on the server when this run did not ask for it.
            server_env.pop("ROBOCODE_TELEMETRY", None)
            server_env.pop("ROBOCODE_RUN_ID", None)

        with open(log_path, "w", encoding="utf-8") as log_file:
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
        # Declared before the try so the finally can always signal shutdown, including
        # when the server never comes up at all.
        died = threading.Event()
        stopping = threading.Event()
        watcher: threading.Thread | None = None

        def _died_error() -> RuntimeError:
            return RuntimeError(
                f"env server {_describe_exit(proc.returncode)} during the run; "
                f"see {log_path}"
            )

        try:
            deadline = time.monotonic() + 60
            while not port_file.exists():
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"env server exited at startup; see {log_path}:\n"
                        f"{log_path.read_text(encoding='utf-8')}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError(f"env server did not start; see {log_path}")
                time.sleep(0.1)
            port = int(port_file.read_text(encoding="utf-8"))
            logger.info("env server listening on port %d", port)

            # The server can die mid-run; a native crash on one of its worker threads
            # will do it. Nothing downstream notices on its own, because the agent just
            # sees its env calls start failing and keeps working. Checking only after
            # the session returns means the run burns its entire remaining budget
            # talking to a dead server and is then discarded anyway, so watch the
            # process and notify its owner the moment it exits.
            def _watch_server() -> None:
                code = proc.wait()
                if stopping.is_set():
                    return  # our own shutdown below, not a failure
                died.set()
                logger.error(
                    "env server %s during the run; see %s",
                    _describe_exit(code),
                    log_path,
                )
                if on_failure is not None:
                    on_failure()

            watcher = threading.Thread(
                target=_watch_server, name="env-server-watchdog", daemon=True
            )
            watcher.start()

            (parent / "status.json").write_text(
                json.dumps(
                    {"service_id": parent.name, "status": "running", "pid": proc.pid}
                ),
                encoding="utf-8",
            )
            yield EnvironmentBinding(parent.name, "127.0.0.1", port, token, strict)

            if proc.poll() is not None:
                raise _died_error()
        except KeyboardInterrupt:
            # Tell our own interrupt apart from an operator's Ctrl-C, so a dead server
            # is not reported as somebody pressing a key. This handler sits on the same
            # try as the finally below rather than in a nested one, so that the single
            # try/finally wrapping the yield is the whole story: whatever is thrown in
            # here -- our interrupt, a real Ctrl-C, or the GeneratorExit a caller raises
            # by abandoning the block -- reaches the shutdown below by one path.
            if not died.is_set():
                raise
            raise _died_error() from None
        finally:
            stopping.set()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Also reap any child processes that outlived the service process.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            if watcher is not None:
                watcher.join(timeout=1)
            (parent / "status.json").write_text(
                json.dumps(
                    {
                        "service_id": parent.name,
                        "status": "stopped",
                        "exit_code": proc.returncode,
                        "pid": proc.pid,
                    }
                ),
                encoding="utf-8",
            )
