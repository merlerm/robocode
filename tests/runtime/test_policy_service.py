"""Policy worker protocol, isolation and failure checks, served from a local process."""

from __future__ import annotations

import ast
import json
import os
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robocode.runtime.policy_service import stage_policy_runtime
from robocode.runtime.policy_worker import PolicyServer, _Connection, decode_state
from robocode.runtime.snapshots import snapshot_policy
from robocode.utils import policy_client
from robocode.utils.env_client import _decode, _ocs_from_payload
from robocode.utils.isolated_transport import TCPRelay
from robocode.utils.policy_client import PolicyClient, PolicyServiceError, encode_state

TOKEN = "t" * 32
SECRET = "SOURCE-SECRET-91f2"
BOX = {"type": "Box", "shape": [3], "low": [-9] * 3, "high": [9] * 3}
BOX_SPACES = {
    "action_space": {**BOX, "dtype": "float32"},
    "observation_space": {**BOX, "dtype": "float32"},
}
TYPES: list[dict[str, Any]] = [
    {"name": "thing", "parent": None, "features": ["x"]},
    {"name": "block", "parent": "thing", "features": ["x", "y"]},
]
OC_SPACES = {
    "action_space": BOX_SPACES["action_space"],
    "observation_space": {"type": "ObjectCentric", "types": TYPES},
}

COUNTER = f"""
import os
import numpy as np
SECRET = {SECRET!r}

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        assert type(observation_space).__name__ == "Box"
        self.calls = self.updates = 0

    def reset(self, state, info):
        assert info == {{"seed": 3}}

    def get_action(self, state):
        assert isinstance(state, np.ndarray) and state.dtype == np.float32
        self.calls += 1
        return [self.calls, self.updates, os.getpid()]

    def update(self, state, reward, done, info):
        assert reward == 1.5 and done is False and info == {{}}
        self.updates += 1
"""

OBJECT_CENTRIC = """
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.block = observation_space.get_type("block")

    def reset(self, state, info):
        assert type(state).__name__ == "_ObjectCentricState"

    def get_action(self, state):
        blocks = state.get_objects(self.block)
        assert [o.name for o in blocks] == ["b1", "b2"]
        x = state.get(state.get_object_from_name("t0"), "x")
        return [sum(state.get(o, "y") for o in blocks), len(list(state)), x]
"""


def _policy(fail: str = "", act: str = "return [0, 0, 0]") -> str:
    """A policy without update whose *fail* stage raises with a secret message."""
    body = {
        stage: f"raise RuntimeError({stage!r} + ' ' + SECRET)" if stage == fail else ok
        for stage, ok in [("load", "pass"), ("reset", "pass"), ("act", act)]
    }
    return f"""
SECRET = {SECRET!r}

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        {body["load"]}

    def reset(self, state, info):
        {body["reset"]}

    def get_action(self, state):
        {body["act"]}
"""


class Service:
    """A worker serving one scripted policy through the production TCP relay."""

    def __init__(
        self,
        root: Path,
        source: str,
        spaces: dict[str, Any],
        timeout: float,
        load_timeout: float,
    ):
        submission = root / "submission" / "approach.py"
        submission.parent.mkdir(parents=True)
        submission.write_text(source)
        self.policy_path = str(submission)
        policy = snapshot_policy(submission, root / "policy")
        runtime = stage_policy_runtime(root / "runtime")
        request = {
            "policy": str(root / "policy"),
            "policy_id": policy["id"],
            "runtime": str(root / "runtime"),
            "runtime_source_id": runtime["id"],
            **spaces,
            "seed": 0,
            "request_timeout": timeout,
            "load_timeout": load_timeout,
            "socket": str(root / "p.sock"),
            "ready": str(root / "ready"),
        }
        (root / "request.json").write_text(json.dumps(request))
        self.log = root / "service.log"
        # Hide any installed robocode package so only the shipped files import.
        code = (
            "import logging, os, sys; from pathlib import Path; "
            "sys.path[:] = [p for p in sys.path if not "
            "os.path.isfile(os.path.join(p, 'robocode', '__init__.py'))]; "
            "logging.basicConfig(level=logging.INFO); "
            "from robocode.runtime.policy_worker import serve; "
            "serve(Path(sys.argv[1]), sys.argv[2])"
        )
        env = {
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(root / "runtime"),
            "PYTHONUNBUFFERED": "1",
            "HOME": str(root),
        }
        with self.log.open("w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, "-B", "-c", code, str(root / "request.json"), TOKEN],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 30
        while not (root / "ready").exists():
            assert self.proc.poll() is None, self.log.read_text()
            assert time.monotonic() < deadline, "worker did not start"
            time.sleep(0.05)
        self.stopped = False
        self.relay = TCPRelay(0, str(root / "p.sock"))
        self.thread = threading.Thread(target=self.relay.serve_forever, daemon=True)
        self.thread.start()
        self.meta = {
            "host": "127.0.0.1",
            "port": self.relay.server_address[1],
            "token": TOKEN,
            "action_space": spaces["action_space"],
            "request_timeout_s": timeout + 20,
        }

    def workers(self) -> list[int]:
        """Live child processes of the serving process."""
        path = Path(f"/proc/{self.proc.pid}/task/{self.proc.pid}/children")
        return [int(pid) for pid in path.read_text(encoding="utf-8").split()]

    def wait_for_workers(self, count: int) -> None:
        """Wait until exactly *count* policy workers remain."""
        deadline = time.monotonic() + 10
        while len(self.workers()) != count:
            assert time.monotonic() < deadline, self.workers()
            time.sleep(0.05)

    def raw(self, *requests: dict[str, Any] | bytes) -> list[bytes]:
        """Send requests one at a time on a raw connection; return the reply lines.

        Dicts get the service token; bytes are sent as one line verbatim.
        """
        replies = []
        with socket.create_connection(("127.0.0.1", self.meta["port"]), 30) as sock:
            stream = sock.makefile("rb")
            for request in requests:
                if isinstance(request, dict):
                    request = json.dumps({"token": TOKEN, **request}).encode()
                sock.sendall(request + b"\n")
                reply = stream.readline()
                if not reply:
                    break
                replies.append(reply.strip())
                if b'"error"' in reply:
                    assert stream.read() == b""  # Every error closes the connection.
                    break
        return replies

    def stop(self) -> None:
        """Stop the relay and the server; no server or worker process may remain."""
        if self.stopped:
            return
        self.stopped = True
        self.relay.shutdown()
        self.thread.join()
        self.relay.server_close()
        workers = self.workers() if self.proc.poll() is None else []
        os.killpg(self.proc.pid, signal.SIGTERM)
        self.proc.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.killpg(self.proc.pid, 0)
        for pid in workers:
            _wait_gone(pid)


@pytest.fixture(name="serve")
def fixture_serve() -> Iterator[Any]:
    """Start services in a short directory; Unix socket paths are length-limited."""
    root = Path(tempfile.mkdtemp(prefix="rc-ps.", dir="/tmp"))
    services: list[Service] = []

    def start(
        source: str,
        spaces: dict[str, Any] | None = None,
        timeout: float = 5.0,
        load_timeout: float = 30.0,
    ) -> Service:
        directory = root / str(len(services))
        services.append(
            Service(directory, source, spaces or BOX_SPACES, timeout, load_timeout)
        )
        return services[-1]

    try:
        yield start
    finally:
        try:
            for service in services:
                service.stop()
        finally:
            shutil.rmtree(root)


def _state(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _box_state(values: list[Any], dtype: str = "float32") -> dict[str, Any]:
    return {"__ndarray__": values, "dtype": dtype}


def _gone(pid: int) -> bool:
    """True once *pid* has exited (a zombie awaiting its reaper counts as exited)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def _wait_gone(pid: int) -> None:
    deadline = time.monotonic() + 10
    while not _gone(pid):
        assert time.monotonic() < deadline, f"process {pid} survived"
        time.sleep(0.05)


def test_policy_state_persists_per_connection_and_restarts_fresh(serve) -> None:
    """Calls share one policy instance; a new connection loads a new one."""
    service = serve(COUNTER)
    with PolicyClient(service.meta) as first:
        first.reset(_state(0, 0, 0), {"seed": 3})
        assert first.act(_state(1, 2, 3))[:2].tolist() == [1, 0]
        first.update(_state(1, 2, 3), 1.5, False)
        action = first.act([1.0, 2.0, 3.0])
        assert action.dtype == np.float32 and action[:2].tolist() == [2, 1]
        pid = int(action[2])
    assert pid not in (service.proc.pid, os.getpid())
    with PolicyClient(service.meta) as second:
        second.reset(_state(0, 0, 0), {"seed": 3})
        assert second.act(_state(0, 0, 0))[:2].tolist() == [1, 0]
    service.wait_for_workers(0)


def test_simultaneous_connections_are_independent_processes(serve) -> None:
    """Interleaved calls on two connections never see each other's state."""
    service = serve(COUNTER)
    with PolicyClient(service.meta) as a, PolicyClient(service.meta) as b:
        a.reset(_state(0, 0, 0), {"seed": 3})
        b.reset(_state(0, 0, 0), {"seed": 3})
        for _ in range(3):
            first = a.act(_state(0, 0, 0))
        second = b.act(_state(0, 0, 0))
        assert first[0] == 3 and second[0] == 1
        assert first[2] != second[2]
        assert len(service.workers()) == 2


def test_update_is_a_no_op_without_a_policy_update(serve) -> None:
    """A policy without update still accepts the evaluator's update call."""
    service = serve(_policy())
    with PolicyClient(service.meta) as client:
        client.reset(_state(0, 0, 0))
        client.update(_state(0, 0, 0), 0.0, True, {"k": np.float32(1)})
        assert client.act(_state(0, 0, 0)).tolist() == [0, 0, 0]


def _objects() -> dict[str, Any]:
    return {
        "types": TYPES,
        "objects": [
            {"name": "b2", "type": "block", "features": [0.0, 2.5]},
            {"name": "t0", "type": "thing", "features": [-1.0]},
            {"name": "b1", "type": "block", "features": [0.0, 0.25]},
        ],
    }


def test_object_centric_states_reach_the_policy_as_client_mirrors(serve) -> None:
    """Mirror and real states arrive as the strict object-centric view."""
    service = serve(OBJECT_CENTRIC, OC_SPACES)
    states = [_ocs_from_payload(_objects())]
    relational_structs = pytest.importorskip("relational_structs")
    thing = relational_structs.Type("thing")
    block = relational_structs.Type("block", thing)
    states.append(
        relational_structs.ObjectCentricState(
            {
                relational_structs.Object("b2", block): np.array([0.0, 2.5]),
                relational_structs.Object("t0", thing): np.array([-1.0]),
                relational_structs.Object("b1", block): np.array([0.0, 0.25]),
            },
            {thing: ["x"], block: ["x", "y"]},
        )
    )
    for state in states:
        with PolicyClient(service.meta) as client:
            client.reset(state)
            assert client.act(state).tolist() == [2.75, 3, -1]


def test_real_state_encoding_matches_the_environment_codec() -> None:
    """Client bytes equal the codec's, and both paths decode to the same mirror."""
    pytest.importorskip("relational_structs")
    from relational_structs import (  # pylint: disable=import-outside-toplevel
        Object,
        ObjectCentricState,
        Type,
    )

    from robocode.utils.object_centric_codec import (  # pylint: disable=import-outside-toplevel
        OCS_TAG,
        encode_object_centric_state,
        serialize_object_centric_space,
    )

    base = Type("base")
    robot, block = Type("robot", base), Type("block", base)
    real = ObjectCentricState(
        {
            Object("z", block): np.array([0.1, 1 / 3], dtype=np.float32),
            Object("a", robot): np.array([np.pi], dtype=np.float32),
        },
        {base: [], robot: ["theta"], block: ["x", "y"]},
    )
    space = serialize_object_centric_space(real.type_features)
    service_bytes = json.dumps({OCS_TAG: encode_object_centric_state(real)})
    assert json.dumps(encode_state(real)) == service_bytes
    via_service = _decode(json.loads(service_bytes))
    via_client = decode_state(json.loads(json.dumps(encode_state(real))), space)
    for mirror in (
        via_client,
        decode_state(encode_state(via_service), space),
    ):
        assert mirror.type_features == via_service.type_features
        assert [(o.name, o.type.name) for o in mirror] == [
            (o.name, o.type.name) for o in via_service
        ]
        for obj in via_service:
            assert mirror.data[obj].dtype == np.float32
            assert np.array_equal(mirror.data[obj], via_service.data[obj])
    vector = np.array([0.1, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(
        decode_state(encode_state(vector), BOX_SPACES["observation_space"]), vector
    )


def _assert_private(service: Service, reply: bytes, code: str) -> None:
    assert json.loads(reply) == {"error": code}
    for leak in (SECRET, service.policy_path, "Traceback", "RuntimeError"):
        assert leak.encode() not in reply


@pytest.mark.parametrize("stage", ["load", "reset", "act"])
def test_policy_failures_return_only_a_public_code(serve, stage) -> None:
    """Raises become policy_error; the diagnostic stays in the private log."""
    service = serve(_policy(stage))
    replies = service.raw(
        {"cmd": "reset", "state": encode_state(_state(0, 0, 0)), "info": {}},
        {"cmd": "act", "state": encode_state(_state(0, 0, 0))},
    )
    if stage == "act":
        assert json.loads(replies[0]) == {"ok": True}
    assert len(replies) == 1 + (stage == "act")
    _assert_private(service, replies[-1], "policy_error")
    with pytest.raises(PolicyServiceError) as caught:
        with PolicyClient(service.meta) as client:
            client.reset(_state(0, 0, 0))
            client.act(_state(0, 0, 0))
    assert caught.value.code == str(caught.value) == "policy_error"
    log = service.log.read_text()
    assert f"RuntimeError: {stage} {SECRET}" in log and "Traceback" in log
    service.wait_for_workers(0)


def test_strict_import_violation_is_a_policy_error(serve) -> None:
    """The worker loads with the evaluator's strict import check."""
    service = serve("import gymnasium\n" + _policy("act"))
    replies = service.raw(
        {"cmd": "reset", "state": encode_state(_state(0, 0, 0)), "info": {}}
    )
    _assert_private(service, replies[0], "policy_error")
    assert "StrictImportError" in service.log.read_text()


@pytest.mark.parametrize(
    "request_, code",
    [
        ({"cmd": "act", "state": encode_state(_state(0, 0, 0))}, "protocol_error"),
        ({"cmd": "reset", "state": encode_state(_state(0, 0)), "info": {}}, None),
        ({"cmd": "reset", "state": [0, 0, 0], "info": {}}, None),
        ({"cmd": "step", "state": encode_state(_state(0, 0, 0))}, None),
        ({"cmd": "reset", "state": encode_state(_state(0, 0, 0))}, None),
        ({"cmd": "reset", "token": "wrong", "info": {}}, "unauthorized"),
        ({"cmd": "reset", "state": _box_state([float("nan"), 0, 0]), "info": {}}, None),
        ({"cmd": "reset", "state": _box_state([1e39, 0, 0]), "info": {}}, None),
        (
            {"cmd": "reset", "state": _box_state(["a", "b", "c"], "U1"), "info": {}},
            None,
        ),
        ({"cmd": "reset", "state": _box_state([10**400, 0, 0]), "info": {}}, None),
    ],
    ids=[
        "early_act",
        "shape",
        "untagged",
        "command",
        "fields",
        "token",
        "nan",
        "float32_overflow",
        "text_dtype",
        "huge_integer",
    ],
)
def test_bad_requests_return_only_a_public_code(serve, request_, code) -> None:
    """Act before reset, malformed states or fields, and bad tokens are refused."""
    service = serve(_policy("act"))
    replies = service.raw(request_)
    assert len(replies) == 1
    _assert_private(service, replies[0], code or "protocol_error")
    service.wait_for_workers(0)


def test_hung_action_times_out_and_kills_its_worker(serve) -> None:
    """A hung call returns policy_timeout at the deadline and leaves no process."""
    service = serve(_policy(act="while True: pass"), timeout=1.0)
    with PolicyClient(service.meta) as client:
        client.reset(_state(0, 0, 0))
        (pid,) = service.workers()
        start = time.monotonic()
        with pytest.raises(PolicyServiceError, match="^policy_timeout$"):
            client.act(_state(0, 0, 0))
        assert time.monotonic() - start < 3
    assert service.workers() == []
    assert not Path(f"/proc/{pid}").exists()
    with PolicyClient(service.meta) as client:
        client.reset(_state(0, 0, 0))  # The service still serves new connections.


def test_disconnect_ends_the_worker_even_mid_call(serve) -> None:
    """Closing the client stops its worker, including one stuck in a call."""
    service = serve(_policy(act="while True: pass"), timeout=60.0)
    with PolicyClient(service.meta) as idle, PolicyClient(service.meta) as stuck:
        idle.reset(_state(0, 0, 0))
        stuck.reset(_state(0, 0, 0))
        service.wait_for_workers(2)
        stuck._timeout = 0.5  # pylint: disable=protected-access
        with pytest.raises(PolicyServiceError, match="^policy_timeout$"):
            stuck.act(_state(0, 0, 0))
        service.wait_for_workers(1)
    service.wait_for_workers(0)


def test_client_is_standalone() -> None:
    """The copied client file imports only the standard library and numpy."""
    path = Path(policy_client.__file__)
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }
    assert modules - {"numpy"} <= set(sys.stdlib_module_names)


_RESET = {"cmd": "reset", "state": encode_state(_state(0, 0, 0)), "info": {}}
_ACT = {"cmd": "act", "state": encode_state(_state(0, 0, 0))}


def _connect(service: Service) -> socket.socket:
    return socket.create_connection(("127.0.0.1", service.meta["port"]), 30)


def _send_line(sock: socket.socket, request: dict[str, Any]) -> None:
    sock.sendall(json.dumps({"token": TOKEN, **request}).encode() + b"\n")


@pytest.mark.parametrize(
    "requests, code",
    [
        ([json.dumps({**_RESET, "token": "é" * 32}).encode()], "unauthorized"),
        ([b"[" * 100_000 + b"]" * 100_000], "protocol_error"),
        (
            [
                _RESET,
                {**_ACT, "cmd": "update", "reward": 10**400, "done": False, "info": {}},
            ],
            "protocol_error",
        ),
        (
            [{**_RESET, "info": {"x": json.loads("[" * 900 + "]" * 900)}}],
            "protocol_error",
        ),
    ],
    ids=["non_ascii_token", "deep_json", "huge_reward", "nested_info"],
)
def test_malformed_client_traffic_affects_only_its_connection(
    serve, requests, code
) -> None:
    """Undecodable requests close that connection alone; others keep working."""
    service = serve(_policy())
    with PolicyClient(service.meta) as other:
        other.reset(_state(0, 0, 0))
        replies = service.raw(*requests)
        if code is None:
            assert replies == [b'{"ok": true}']
        else:
            _assert_private(service, replies[-1], code)
        assert other.act(_state(0, 0, 0)).tolist() == [0, 0, 0]
    assert service.proc.poll() is None


_CHANNEL_WRITER = """
import os, socket, time

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass

    def reset(self, state, info):
        pass

    def get_action(self, state):
        for fd in os.listdir("/proc/self/fd"):
            try:
                sock = socket.socket(fileno=os.dup(int(fd)))
            except OSError:
                continue
            sock.sendall(PAYLOAD + b"\\n")
        time.sleep(30)
"""


@pytest.mark.parametrize(
    "payload",
    [b'{"action": [' + b"1" * 400 + b", 0, 0]}", b"[" * 100_000 + b"]" * 100_000],
    ids=["huge_number", "deep_json"],
)
def test_malformed_worker_output_affects_only_its_connection(serve, payload) -> None:
    """A worker writing garbage to its channel fails its own connection only."""
    service = serve(f"PAYLOAD = {payload!r}\n" + _CHANNEL_WRITER)
    with PolicyClient(service.meta) as other:
        other.reset(_state(0, 0, 0))
        replies = service.raw(_RESET, _ACT)
        assert replies[0] == b'{"ok": true}'
        _assert_private(service, replies[1], "policy_error")
        other.update(_state(0, 0, 0), 0.0, False)
    assert service.proc.poll() is None


@pytest.mark.parametrize("entry", ["_on_channel", "_on_reply"])
@pytest.mark.parametrize("stage", ["load", "act"])
def test_reply_after_the_deadline_is_not_forwarded(stage, entry) -> None:
    """A worker reply read after its deadline becomes policy_timeout, not a result."""
    server = PolicyServer.__new__(PolicyServer)
    server.request = {"action_space": {"shape": [3]}, "request_timeout": 5.0}
    server.selector = selectors.DefaultSelector()
    client, caller = socket.socketpair()
    channel, worker = socket.socketpair()
    caller.settimeout(2)
    connection = _Connection(
        client,
        time.monotonic() - 0.1,
        authenticated=True,
        channel=channel,
        ready=stage == "act",
        pending="act" if stage == "act" else "reset",
        queued=None if stage == "act" else b'{"cmd": "reset"}\n',
    )
    server.connections = [connection]
    server.selector.register(client, selectors.EVENT_READ, ("client", connection))
    server.selector.register(channel, selectors.EVENT_READ, ("channel", connection))
    line = b'{"ready": true}' if stage == "load" else b'{"action": [0, 0, 0]}'
    try:
        if entry == "_on_channel":
            worker.sendall(line + b"\n")
            server._on_channel(connection)  # pylint: disable=protected-access
        else:
            server._on_reply(connection, line)  # pylint: disable=protected-access
        assert caller.recv(1000) == b'{"error": "policy_timeout"}\n'
        assert not server.connections
    finally:
        for sock in (client, caller, channel, worker):
            sock.close()
        server.selector.close()


@pytest.mark.parametrize("how", ["disconnect", "timeout", "error", "teardown"])
def test_worker_descendants_are_killed_with_it(serve, tmp_path, how) -> None:
    """Processes a policy spawns die with its worker on every exit path."""
    pidfile = tmp_path / "child.pid"
    service = serve(
        f"""import subprocess, time
from pathlib import Path

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass

    def reset(self, state, info):
        pass

    def get_action(self, state):
        child = subprocess.Popen(["sleep", "300"])
        Path({str(pidfile)!r}).write_text(str(child.pid))
        if state[0] == 1:
            time.sleep(300)
        if state[0] == 2:
            raise RuntimeError("spawned")
        return [0, 0, 0]
""",
        timeout=1.0,
    )
    client = PolicyClient(service.meta)
    client.reset(_state(0, 0, 0))
    mode = {"disconnect": 0, "teardown": 0, "timeout": 1, "error": 2}[how]
    if mode:
        with pytest.raises(PolicyServiceError):
            client.act(_state(mode, 0, 0))
    else:
        client.act(_state(0, 0, 0))
    child = int(pidfile.read_text())
    if how == "disconnect":
        assert not _gone(child)
        client.close()
    elif how == "teardown":
        service.stop()
    _wait_gone(child)
    client.close()


def test_unauthenticated_connections_load_no_policy(serve, tmp_path) -> None:
    """Workers start only after a request with the right token."""
    marker = tmp_path / "loaded"
    service = serve(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + _policy()
    )
    with _connect(service):
        time.sleep(0.5)
        assert service.workers() == []
        _assert_private(
            service, service.raw({**_RESET, "token": "x"})[0], "unauthorized"
        )
        assert service.workers() == [] and not marker.exists()
    with PolicyClient(service.meta) as client:
        client.reset(_state(0, 0, 0))
    assert marker.exists()


def test_admission_limits_return_busy(serve) -> None:
    """Too many idle or active connections are refused with a public code."""
    service = serve(_policy())
    idle = [_connect(service) for _ in range(8)]
    try:
        time.sleep(0.5)
        with _connect(service) as extra:
            assert extra.makefile("rb").readline() == b'{"error": "busy"}\n'
    finally:
        for sock in idle:
            sock.close()
    clients = [PolicyClient(service.meta) for _ in range(8)]
    try:
        for client in clients:
            client.reset(_state(0, 0, 0))
        with pytest.raises(PolicyServiceError, match="^busy$"):
            with PolicyClient(service.meta) as extra_client:
                extra_client.reset(_state(0, 0, 0))
        assert len(service.workers()) == 8
    finally:
        for client in clients:
            client.close()


def _ocs(types: list[Any], objects: list[Any]) -> dict[str, Any]:
    return {"__ocs__": {"types": types, "objects": objects}}


_BLOCK = {"name": "b1", "type": "block", "features": [0.0, 0.25]}


@pytest.mark.parametrize(
    "state",
    [
        _ocs(TYPES + [{"name": "ghost", "parent": None, "features": []}], []),
        _ocs([TYPES[0], {**TYPES[1], "parent": None}], [_BLOCK]),
        _ocs([TYPES[0], {**TYPES[1], "features": ["y", "x"]}], [_BLOCK]),
        _ocs(TYPES, [{**_BLOCK, "features": [0.0]}]),
        _ocs(TYPES, [{**_BLOCK, "type": "ghost"}]),
        _ocs(TYPES, [{**_BLOCK, "features": [float("inf"), 0.0]}]),
        _ocs(TYPES, [{**_BLOCK, "features": [1e39, 0.0]}]),
        _ocs(TYPES, [_BLOCK, _BLOCK]),
        {"types": TYPES, "objects": [_BLOCK]},
        _box_state([0, 0, 0]),
    ],
    ids=[
        "unknown_type",
        "parent",
        "feature_order",
        "feature_count",
        "object_type",
        "non_finite",
        "float32_overflow",
        "duplicate_name",
        "untagged",
        "box_for_object_centric",
    ],
)
def test_object_centric_states_must_match_the_declared_space(serve, state) -> None:
    """States outside the declared schema are protocol errors."""
    service = serve(OBJECT_CENTRIC, OC_SPACES)
    replies = service.raw({"cmd": "reset", "state": state, "info": {}})
    _assert_private(service, replies[0], "protocol_error")


def test_send_timeout_closes_only_the_affected_connection() -> None:
    """A full client socket times out; the next normal-sized reply still works."""
    server = PolicyServer.__new__(PolicyServer)
    server.request = {"action_space": {"shape": [3]}}
    server.selector = selectors.DefaultSelector()
    server.connections = []
    peers = []
    try:
        for _ in range(2):
            client, peer = socket.socketpair()
            peers.append(peer)
            peer.settimeout(2)
            connection = _Connection(client, None, ready=True, pending="act")
            server.connections.append(connection)
            server.selector.register(
                client, selectors.EVENT_READ, ("client", connection)
            )
        stalled, healthy = server.connections
        stalled.client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        stalled.client.setblocking(False)
        while True:
            try:
                stalled.client.send(b"x" * 4096)
            except BlockingIOError:
                break
        stalled.client.settimeout(0.05)
        reply = b'{"action": [0, 0, 0]}'
        server._on_reply(stalled, reply)  # pylint: disable=protected-access
        assert server.connections == [healthy]
        assert stalled.client.fileno() == -1
        server._on_reply(healthy, reply)  # pylint: disable=protected-access
        assert json.loads(peers[1].recv(1000)) == {"action": [0, 0, 0]}
    finally:
        for connection in list(server.connections):
            server._close(connection)  # pylint: disable=protected-access
        for peer in peers:
            peer.close()
        server.selector.close()
