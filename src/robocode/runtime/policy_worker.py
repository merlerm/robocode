"""Serve a frozen strict-blackbox policy on a Unix socket, one process per connection.

The serving process never runs policy code. It authenticates and validates each request,
forwards it to the connection's own forked worker, enforces deadlines and replies with
fixed public error codes only. Diagnostics go to this process's log.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import selectors
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robocode.runtime.binding import EnvironmentBinding
from robocode.runtime.evaluation_worker import FrozenPolicy
from robocode.runtime.remote_environment import RemoteEnvironment
from robocode.runtime.snapshots import verify_snapshot
from robocode.utils.env_client import _decode
from robocode.utils.isolated_transport import verify_namespace

logger = logging.getLogger(__name__)

_MAX_LINE = 16 * 2**20
_MAX_WORKERS = 8
_MAX_UNAUTHENTICATED = 8
_AUTH_TIMEOUT_S = 10.0
_SEND_TIMEOUT_S = 1.0
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FIELDS = {
    "reset": {"cmd", "state", "info"},
    "act": {"cmd", "state"},
    "update": {"cmd", "state", "reward", "done", "info"},
}


def strict_view(spaces: dict[str, Any]) -> RemoteEnvironment:
    """The evaluator's default strict spaces; this environment never connects."""
    metadata = {key: spaces[key] for key in ("action_space", "observation_space")}
    return RemoteEnvironment(
        EnvironmentBinding("policy", "", 0, "", True, None, metadata), 0
    )


def _check_features(values: Any, count: int) -> None:
    """Require *count* JSON numbers that stay finite as float32."""
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Feature vector has the wrong length")
    if not all(type(v) in (int, float) for v in values):
        raise ValueError("Features must be numbers")
    array = np.asarray(values, dtype=np.float64)
    if not (np.isfinite(array).all() and (np.abs(array) <= _FLOAT32_MAX).all()):
        raise ValueError("Features must be finite float32 values")


def _check_object_centric(payload: Any, declared: list[dict[str, Any]]) -> None:
    """Require declared types (name, parent, features in order) and matching objects."""
    if not isinstance(payload, dict) or not (
        {"types", "objects"} <= set(payload) <= {"types", "objects", "state_cls"}
    ):
        raise ValueError("Malformed object-centric payload")
    if not isinstance(payload["types"], list) or not isinstance(
        payload["objects"], list
    ):
        raise ValueError("Malformed object-centric payload")
    known = {entry["name"]: entry for entry in declared}
    widths: dict[str, int] = {}
    for entry in payload["types"]:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or entry != known.get(name):
            raise ValueError(f"Undeclared or altered type {name!r}")
        if entry["parent"] is not None and entry["parent"] not in widths:
            raise ValueError("Types must list parents first")
        widths[name] = len(entry["features"])
    names = set()
    for obj in payload["objects"]:
        if not isinstance(obj, dict) or set(obj) != {"name", "type", "features"}:
            raise ValueError("Malformed object")
        if not isinstance(obj["name"], str) or obj["name"] in names:
            raise ValueError("Object names must be unique strings")
        if obj["type"] not in widths:
            raise ValueError("Object has an unlisted type")
        _check_features(obj["features"], widths[obj["type"]])
        names.add(obj["name"])


def decode_state(raw: Any, spec: dict[str, Any]) -> Any:
    """Check a wire state against the declared space; decode it as at evaluation."""
    if spec["type"] == "ObjectCentric":
        if not isinstance(raw, dict) or set(raw) != {"__ocs__"}:
            raise ValueError("Object-centric states use the __ocs__ encoding")
        _check_object_centric(raw["__ocs__"], spec["types"])
        return _decode(raw)
    if not isinstance(raw, dict) or set(raw) != {"__ndarray__", "dtype"}:
        raise ValueError("Box states use the __ndarray__ encoding")
    state = _decode(raw)
    if state.dtype.kind not in "biuf" or list(state.shape) != spec["shape"]:
        raise ValueError("State does not match the Box observation space")
    with np.errstate(all="ignore"):
        converted = state.astype(spec["dtype"])
    if not (np.isfinite(state).all() and np.isfinite(converted).all()):
        raise ValueError("State is not finite in the observation dtype")
    return converted


def check_request(request: dict[str, Any], spaces: dict[str, Any], reset: bool) -> None:
    """Raise unless *request* is a well-formed call that is valid at this point."""
    cmd = request.get("cmd")
    if cmd not in _FIELDS or set(request) != _FIELDS[cmd]:
        raise ValueError(f"Malformed request fields: {sorted(request)}")
    if cmd != "reset" and not reset:
        raise ValueError(f"{cmd} before reset")
    decode_state(request["state"], spaces["observation_space"])
    if "info" in request:
        if not isinstance(request["info"], dict):
            raise ValueError("info must be an object")
        _decode(request["info"])
    if cmd == "update" and (
        not isinstance(request["done"], bool)
        or type(request["reward"]) not in (int, float)
        or not math.isfinite(float(request["reward"]))
    ):
        raise ValueError("update needs a finite reward and a boolean done")


def public_reply(cmd: str, reply: Any, action_shape: list[int]) -> dict[str, Any]:
    """Pass on only well-formed successful replies; anything else is a policy error."""
    if cmd != "act":
        return {"ok": True} if reply == {"ok": True} else {"error": "policy_error"}
    if not isinstance(reply, dict) or set(reply) != {"action"}:
        return {"error": "policy_error"}
    try:
        action = np.asarray(reply["action"], dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return {"error": "policy_error"}
    if list(action.shape) != action_shape:
        return {"error": "policy_error"}
    return {"action": action.tolist()}


def _send(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(json.dumps(message).encode("utf-8") + b"\n")


def _run_policy(channel: socket.socket, request: dict[str, Any], view: Any) -> int:
    """A connection's worker: load one policy, then serve its calls until EOF."""
    try:
        policy = FrozenPolicy(
            view,
            request["seed"],
            strict_imports=True,
            path=Path(request["policy"]) / "approach.py",
        )
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Policy failed to load")
        return 1
    _send(channel, {"ready": True})
    action_spec = request["action_space"]
    for line in channel.makefile("rb"):
        call = json.loads(line)
        try:
            state = decode_state(call["state"], request["observation_space"])
            if call["cmd"] == "reset":
                policy.reset(state, _decode(call["info"]))
                reply: dict[str, Any] = {"ok": True}
            elif call["cmd"] == "update":
                info = _decode(call["info"])
                policy.update(state, float(call["reward"]), call["done"], info)
                reply = {"ok": True}
            else:
                action = np.asarray(
                    policy.generated.get_action(state), dtype=action_spec["dtype"]
                )
                if list(action.shape) != action_spec["shape"]:
                    raise ValueError(f"Action has shape {action.shape}")
                reply = {"action": action.tolist()}
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Policy %s failed", call["cmd"])
            return 1
        _send(channel, reply)
    return 0


@dataclass(eq=False)
class _Connection:
    """One client connection and, once authenticated, the worker that owns its policy.

    ``deadline`` bounds authentication, then loading, then each pending call.
    """

    client: socket.socket
    deadline: float | None
    authenticated: bool = False
    channel: socket.socket | None = None
    pid: int | None = None
    buffer: bytes = b""
    replies: bytes = b""
    ready: bool = False
    reset: bool = False
    pending: str | None = None
    queued: bytes | None = None
    failure: str | None = None


class PolicyServer:
    """Accept connections and supervise one forked policy worker for each."""

    def __init__(self, request: dict[str, Any], token: str):
        self.request = request
        self.token = token.encode()
        self.view = strict_view(request)
        self.selector = selectors.DefaultSelector()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(request["socket"])
        self.listener.listen(16)
        self.selector.register(self.listener, selectors.EVENT_READ)
        self.connections: list[_Connection] = []

    def serve_forever(self) -> None:
        """Serve until terminated; on exit kill and reap every worker."""
        Path(self.request["ready"]).touch()
        try:
            while True:
                deadlines = [c.deadline for c in self.connections if c.deadline]
                timeout = (
                    max(0.0, min(deadlines) - time.monotonic()) if deadlines else None
                )
                events = self.selector.select(timeout)
                for connection in list(self.connections):
                    self._expired(connection)
                for key, _ in events:
                    if key.fileobj is self.listener:
                        self._accept()
                    else:
                        self._dispatch(key)
        finally:
            for connection in list(self.connections):
                self._close(connection)
            self.listener.close()

    def _dispatch(self, key: selectors.SelectorKey) -> None:
        """Run one connection's handler; any failure ends that connection only."""
        kind, connection = key.data
        if connection not in self.connections:
            return  # Closed earlier in this batch.
        if kind == "channel" and key.fileobj is not connection.channel:
            return
        try:
            if kind == "client":
                self._on_client(connection)
            else:
                self._on_channel(connection)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Connection handler failed")
            if connection in self.connections:
                code = "protocol_error" if kind == "client" else "policy_error"
                self._reject(connection, code, "Handler failed")

    def _accept(self) -> None:
        try:
            client, _ = self.listener.accept()
        except OSError:
            logger.exception("Accept failed")
            return
        client.settimeout(_SEND_TIMEOUT_S)
        connection = _Connection(client, time.monotonic() + _AUTH_TIMEOUT_S)
        self.connections.append(connection)
        self.selector.register(client, selectors.EVENT_READ, ("client", connection))
        waiting = sum(not c.authenticated for c in self.connections)
        if waiting > _MAX_UNAUTHENTICATED:
            self._reject(connection, "busy", "Too many unauthenticated connections")

    def _start_worker(self, connection: _Connection) -> None:
        """Fork this connection's worker in its own process group."""
        parent, child = socket.socketpair()
        inherited = [self.listener, parent]
        for other in self.connections:
            inherited += [other.client] + ([other.channel] if other.channel else [])
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                # Own group, so descendants die with it; keep only this channel.
                os.setpgid(0, 0)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self.selector.close()
                for sock in inherited:
                    sock.close()
                code = _run_policy(child, self.request, self.view)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)  # pylint: disable=protected-access
        try:
            os.setpgid(pid, pid)  # Also here, so the group exists before any kill.
        except OSError:
            pass  # The child already made its group.
        child.close()
        parent.settimeout(_SEND_TIMEOUT_S)
        connection.authenticated, connection.channel, connection.pid = True, parent, pid
        connection.deadline = time.monotonic() + self.request["load_timeout"]
        self.selector.register(parent, selectors.EVENT_READ, ("channel", connection))
        logger.info("Connection authenticated; policy worker pid %d", pid)

    def _on_client(self, connection: _Connection) -> None:
        try:
            data = connection.client.recv(65536)
        except OSError:
            data = b""
        if not data:
            logger.info("Client disconnected")
            self._close(connection)
            return
        connection.buffer += data
        while connection in self.connections and b"\n" in connection.buffer:
            line, connection.buffer = connection.buffer.split(b"\n", 1)
            self._on_request(connection, line)
        if connection in self.connections and len(connection.buffer) > _MAX_LINE:
            self._reject(connection, "protocol_error", "Request exceeds the size limit")

    def _on_request(self, connection: _Connection, line: bytes) -> None:
        try:
            request = json.loads(line)
        except (ValueError, RecursionError):
            request = None
        if not isinstance(request, dict):
            self._reject(connection, "protocol_error", "Request is not a JSON object")
            return
        token = request.pop("token", None)
        if not isinstance(token, str) or not secrets.compare_digest(
            token.encode("utf-8", "surrogatepass"), self.token
        ):
            self._reject(connection, "unauthorized", "Request has a bad token")
            return
        if connection.failure is not None:
            self._reject(connection, connection.failure, "Worker ended earlier")
            return
        if connection.pending is not None:
            self._reject(connection, "protocol_error", "Request while one is pending")
            return
        try:
            check_request(request, self.request, connection.reset)
            data = json.dumps(request).encode("utf-8") + b"\n"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._reject(connection, "protocol_error", f"{type(exc).__name__}: {exc}")
            return
        if not connection.authenticated:
            if sum(c.pid is not None for c in self.connections) >= _MAX_WORKERS:
                self._reject(connection, "busy", "Too many active policy workers")
                return
            self._start_worker(connection)
        connection.reset = connection.reset or request["cmd"] == "reset"
        connection.pending = request["cmd"]
        if connection.ready:
            self._forward(connection, data)
        else:
            connection.queued = data  # Sent once the policy has loaded.

    def _forward(self, connection: _Connection, data: bytes) -> None:
        assert connection.channel is not None
        connection.deadline = time.monotonic() + self.request["request_timeout"]
        try:
            connection.channel.sendall(data)
        except OSError:
            self._worker_failed(connection, "policy_error", "Worker stopped reading")

    def _on_channel(self, connection: _Connection) -> None:
        assert connection.channel is not None
        if self._expired(connection):
            return
        try:
            data = connection.channel.recv(65536)
        except OSError:
            data = b""
        if not data:
            self._worker_failed(connection, "policy_error", "Policy worker ended")
            return
        connection.replies += data
        while (
            connection in self.connections
            and connection.channel is not None
            and b"\n" in connection.replies
        ):
            line, connection.replies = connection.replies.split(b"\n", 1)
            self._on_reply(connection, line)
        if connection.channel is not None and len(connection.replies) > _MAX_LINE:
            self._worker_failed(connection, "policy_error", "Worker reply too large")

    def _on_reply(self, connection: _Connection, line: bytes) -> None:
        if self._expired(connection):
            return
        try:
            reply = json.loads(line)
        except (ValueError, RecursionError):
            reply = None
        if not connection.ready:
            if reply != {"ready": True}:
                self._worker_failed(connection, "policy_error", "Bad worker handshake")
                return
            connection.ready, connection.deadline = True, None
            if connection.queued is not None:
                data, connection.queued = connection.queued, None
                self._forward(connection, data)
            return
        if connection.pending is None:
            self._worker_failed(connection, "policy_error", "Unrequested worker reply")
            return
        message = public_reply(
            connection.pending, reply, self.request["action_space"]["shape"]
        )
        if "error" in message:
            self._worker_failed(connection, message["error"], "Malformed worker reply")
            return
        connection.pending, connection.deadline = None, None
        try:
            _send(connection.client, message)
        except OSError:
            logger.warning("Client send timed out or failed; closing its connection")
            self._close(connection)

    def _expired(self, connection: _Connection) -> bool:
        """Fail *connection* if its deadline has passed; return whether it did."""
        if connection not in self.connections or connection.deadline is None:
            return False
        if time.monotonic() < connection.deadline:
            return False
        if not connection.authenticated:
            self._reject(connection, "unauthorized", "No authenticated request in time")
        else:
            stage = connection.pending or "load"
            self._worker_failed(
                connection, "policy_timeout", f"{stage} deadline exceeded"
            )
        return True

    def _worker_failed(self, connection: _Connection, code: str, why: str) -> None:
        """End the worker; answer a pending call now, any later one with *code*."""
        logger.warning("Policy worker failed (%s): %s", code, why)
        self._kill(connection)
        connection.deadline, connection.queued = None, None
        if connection.pending is None:
            connection.failure = code
        else:
            self._reject(connection, code, why)

    def _reject(self, connection: _Connection, code: str, why: str) -> None:
        """Send one public error code and close the connection."""
        logger.warning("Closing connection with %s: %s", code, why)
        try:
            _send(connection.client, {"error": code})
        except OSError:
            logger.warning("Client send timed out or failed")
        self._close(connection)

    def _kill(self, connection: _Connection) -> None:
        """Kill and reap the worker's process group; the client may stay open."""
        if connection.channel is not None:
            self.selector.unregister(connection.channel)
            connection.channel.close()
            connection.channel = None
        if connection.pid is not None:
            # Signal the group before reaping, so its id cannot be reused meanwhile.
            try:
                os.killpg(connection.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(connection.pid, 0)
            logger.info("Policy worker group %d stopped", connection.pid)
            connection.pid = None

    def _close(self, connection: _Connection) -> None:
        if connection not in self.connections:
            return
        self._kill(connection)
        self.selector.unregister(connection.client)
        connection.client.close()
        self.connections.remove(connection)


def serve(request_path: Path, token: str) -> None:
    """Verify the frozen inputs, then serve until terminated."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    verify_snapshot(Path(request["policy"]), request["policy_id"])
    verify_snapshot(Path(request["runtime"]), request["runtime_source_id"])
    signal.signal(signal.SIGTERM, _stop)
    PolicyServer(request, token).serve_forever()


def _stop(*_: Any) -> None:
    """Exit through cleanup once; launchers may deliver SIGTERM more than once."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(0)


def main() -> None:
    """Container entry point: ``policy_worker <request.json> <token>``."""
    logging.basicConfig(level=logging.INFO)
    verify_namespace()
    serve(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
