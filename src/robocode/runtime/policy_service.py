"""Run a frozen strict-blackbox policy as a service in an isolated container.

Callers reach the policy through :mod:`robocode.utils.policy_client` and never see
its files. See ``docs/runtime/policy-service.md`` for the protocol.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robocode.runtime.container_evaluation import _RUNTIME_FILES
from robocode.runtime.containers import container_process, image_identity
from robocode.runtime.snapshots import file_manifest, snapshot_policy
from robocode.utils.isolated_transport import TCPRelay
from robocode.utils.strict_blackbox import check_strict_imports

logger = logging.getLogger(__name__)

# The evaluator's generic subset plus the serving loop; nothing simulator-specific.
POLICY_RUNTIME_FILES = (*_RUNTIME_FILES, "runtime/policy_worker.py")
_LOAD_TIMEOUT_S = 60.0
_STARTUP_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class PolicyBinding:
    """Where a policy service listens; the token is omitted from logs."""

    host: str
    port: int
    token: str = field(repr=False)
    policy_artifact_id: str
    action_space: dict[str, Any] = field(repr=False)
    request_timeout_s: float

    def client_metadata(self) -> dict[str, Any]:
        """The ``meta`` argument of
        :class:`robocode.utils.policy_client.PolicyClient`."""
        return {
            "host": self.host,
            "port": self.port,
            "token": self.token,
            "policy_artifact_id": self.policy_artifact_id,
            "action_space": self.action_space,
            "request_timeout_s": self.request_timeout_s,
        }


def stage_policy_runtime(destination: Path) -> dict[str, Any]:
    """Copy the shipped source subset under ``destination/robocode``; return its
    hash."""
    source = Path(__file__).resolve().parents[1]
    for name in POLICY_RUNTIME_FILES:
        target = destination / "robocode" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return file_manifest(destination)


@contextmanager
def policy_service(
    policy: Path,
    spaces: dict[str, Any],
    service_dir: Path,
    *,
    backend: str,
    image: str,
    blackbox_strict: bool,
    seed: int,
    request_timeout: float = 30.0,
) -> Iterator[PolicyBinding]:
    """Freeze *policy*, serve it from *image* and always remove the container.

    *spaces* holds ``action_space`` and ``observation_space`` specs, as in an
    environment binding's metadata. *seed* is the evaluator's replicate seed.
    """
    if not blackbox_strict:
        raise ValueError(
            "The policy service runs only policies written in strict blackbox mode; "
            "source-authored (whitebox) policies are not supported"
        )
    action, observation = spaces["action_space"], spaces["observation_space"]
    if action["type"] != "Box" or observation["type"] not in ("Box", "ObjectCentric"):
        raise ValueError(
            "The policy service needs a Box action space and a Box or "
            "ObjectCentric observation space"
        )
    service_dir = service_dir.resolve()
    service_dir.mkdir(parents=True)  # Refuse to reuse another run's directory.
    policy_manifest = snapshot_policy(policy, service_dir / "policy")
    # Fails loudly here, before any container starts; workers check again on load.
    check_strict_imports(service_dir / "policy" / "approach.py")
    runtime_manifest = stage_policy_runtime(service_dir / "runtime")
    for name, manifest in [
        ("policy", policy_manifest),
        ("runtime-source", runtime_manifest),
    ]:
        (service_dir / (name + ".json")).write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    identity = image_identity(backend, image)
    token = secrets.token_hex(16)
    inputs = service_dir / "input"
    inputs.mkdir()
    (inputs / "request.json").write_text(
        json.dumps(
            {
                "policy": "/policy",
                "policy_id": policy_manifest["id"],
                "runtime": "/runtime",
                "runtime_source_id": runtime_manifest["id"],
                "action_space": action,
                "observation_space": observation,
                "seed": seed,
                "request_timeout": request_timeout,
                "load_timeout": _LOAD_TIMEOUT_S,
                "socket": "/bridge/policy.sock",
                "ready": "/bridge/ready",
            },
            indent=2,
        )
        + "\n"
    )
    reference = identity if backend == "docker" else image
    try:
        # Unix socket names must fit the kernel's short pathname limit.
        with tempfile.TemporaryDirectory(prefix="rc-policy-", dir="/tmp") as temporary:
            bridge = Path(temporary)
            with container_process(
                backend,
                reference,
                service_dir,
                role="policy",
                worker=[
                    "/usr/local/bin/python",
                    "-B",
                    "-m",
                    "robocode.runtime.policy_worker",
                    "/input/request.json",
                    token,
                ],
                mounts=[
                    f"{service_dir / 'runtime'}:/runtime:ro",
                    f"{service_dir / 'policy'}:/policy:ro",
                    f"{inputs}:/input:ro",
                    f"{bridge}:/bridge:rw",
                ],
                env={
                    "PYTHONPATH": "/runtime",
                    "HOME": "/tmp",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                log_name="policy_service.log",
                token=token,
            ) as (proc, context, log_thread):
                deadline = time.monotonic() + _STARTUP_TIMEOUT_S
                while not (bridge / "ready").exists():
                    if proc.poll() is not None:
                        log_thread.join()
                        raise RuntimeError(
                            "Policy container failed:\n"
                            + (service_dir / "policy_service.log").read_text()
                        )
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy container startup timed out")
                    time.sleep(0.1)
                metadata = {
                    "policy_artifact_id": policy_manifest["id"],
                    "runtime_source_id": runtime_manifest["id"],
                    "runtime_image": identity,
                    "log_context": context,
                }
                if backend == "docker":
                    metadata["container_id"] = (
                        (service_dir / "container.cid").read_text().strip()
                    )
                (service_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                with TCPRelay(
                    0,
                    str(bridge / "policy.sock"),
                    host="0.0.0.0" if backend == "docker" else "127.0.0.1",
                ) as relay:
                    thread = threading.Thread(target=relay.serve_forever, daemon=True)
                    thread.start()
                    try:
                        yield PolicyBinding(
                            "127.0.0.1",
                            relay.server_address[1],
                            token,
                            policy_manifest["id"],
                            action,
                            # Covers loading too, so the server always replies first.
                            _LOAD_TIMEOUT_S + request_timeout + 10,
                        )
                    finally:
                        relay.shutdown()
                        thread.join()
                if proc.poll() is not None:
                    raise RuntimeError(
                        "Policy container exited while in use; see its log"
                    )
    except BaseException as error:
        # Keep the original failure; record an image change on it.
        if not _same_image(backend, reference, identity):
            logger.error("Policy image changed during execution")
            error.add_note("Policy image changed during execution")
        raise
    if not _same_image(backend, reference, identity):
        raise ValueError("Policy image changed during execution")


def _same_image(backend: str, reference: str, identity: str) -> bool:
    """Whether *reference* still resolves to *identity*; an unreadable image does
    not."""
    try:
        return image_identity(backend, reference) == identity
    except (OSError, subprocess.CalledProcessError):
        return False
