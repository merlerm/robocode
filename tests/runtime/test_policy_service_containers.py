"""Live policy-service checks on the generic evaluation image, for both backends."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robocode.runtime import policy_service as policy_service_module
from robocode.runtime.policy_service import policy_service
from robocode.utils.env_client import _ocs_from_payload
from robocode.utils.policy_client import PolicyClient, PolicyServiceError
from robocode.utils.strict_blackbox import StrictImportError

_REPO = Path(__file__).resolve().parents[2]
_DOCKER_IMAGE = "robocode-evaluation:multi-agent"
_SIF = _REPO / "robocode-evaluation.sif"


def _docker_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", _DOCKER_IMAGE],
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


_BACKENDS = [
    pytest.param(
        "docker",
        marks=pytest.mark.skipif(
            not _docker_image_available(), reason=f"{_DOCKER_IMAGE} is unavailable"
        ),
    ),
    pytest.param(
        "apptainer",
        marks=pytest.mark.skipif(
            shutil.which("apptainer") is None or not _SIF.exists(),
            reason=f"Apptainer or {_SIF} is unavailable",
        ),
    ),
]
_BOX = {"type": "Box", "shape": [2], "low": [-5, -5], "high": [5, 5]}
_ACTION = {**_BOX, "dtype": "float32"}
_TYPES = [
    {"name": "thing", "parent": None, "features": ["x"]},
    {"name": "block", "parent": "thing", "features": ["x", "y"]},
]
_SPACES = {
    "box": {"action_space": _ACTION, "observation_space": {**_BOX, "dtype": "float32"}},
    "object_centric": {
        "action_space": _ACTION,
        "observation_space": {"type": "ObjectCentric", "types": _TYPES},
    },
}


def _image(backend: str) -> str:
    return _DOCKER_IMAGE if backend == "docker" else str(_SIF)


def _assert_removed(service_dir: Path, backend: str) -> None:
    """The bridge directory and, for Docker, the container are gone."""
    command = json.loads((service_dir / "command.json").read_text())
    bridge = next(arg.split(":")[0] for arg in command if arg.endswith(":/bridge:rw"))
    assert not Path(bridge).exists()
    if backend == "docker":
        identity = (service_dir / "container.cid").read_text().strip()
        result = subprocess.run(
            ["docker", "inspect", identity], capture_output=True, check=False
        )
        assert result.returncode != 0


def _isolated_policy(host_secret: Path, observation: str) -> str:
    """Probe the evaluator's source isolation at load; act from the strict state."""
    action = (
        "return [float(state[0]) + self.calls, float(self.updates)]"
        if observation == "box"
        else "blocks = state.get_objects(self.space.get_type('block'))\n"
        "        return [sum(state.get(o, 'y') for o in blocks) + self.calls, "
        "float(len(list(state)))]"
    )
    return f"""import socket
from pathlib import Path
assert not Path({str(host_secret)!r}).exists()
assert not Path('/artifact').exists()
assert not Path('/runtime/robocode/environments/kinder_geom2d_env.py').exists()
assert {{name for _, name in socket.if_nameindex()}} == {{'lo'}}
try:
    Path('/policy/approach.py').write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('policy source is writable')
print('SOURCE_ISOLATION_AND_READONLY_POLICY_VERIFIED', flush=True)

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = observation_space
        self.calls = self.updates = 0

    def reset(self, state, info):
        pass

    def get_action(self, state):
        self.calls += 1
        {action}

    def update(self, state, reward, done, info):
        self.updates += 1
"""


@pytest.mark.parametrize("observation", ["box", "object_centric"])
@pytest.mark.parametrize("backend", _BACKENDS)
def test_policy_service_serves_an_isolated_strict_policy(
    backend: str, observation: str, tmp_path: Path
) -> None:
    """Host calls reach a policy that sees only its mounts and no network."""
    host_secret = tmp_path / "host-only.txt"
    host_secret.write_text("not mounted")
    policy = tmp_path / "submission" / "approach.py"
    policy.parent.mkdir()
    policy.write_text(_isolated_policy(host_secret, observation))
    state: Any
    if observation == "box":
        state = np.array([1.5, 0.0], dtype=np.float32)
    else:
        state = _ocs_from_payload(
            {
                "types": _TYPES,
                "objects": [
                    {"name": "b1", "type": "block", "features": [0.0, 0.25]},
                    {"name": "t0", "type": "thing", "features": [3.0]},
                ],
            }
        )
    service_dir = tmp_path / "policy-service"
    with policy_service(
        policy,
        _SPACES[observation],
        service_dir,
        backend=backend,
        image=_image(backend),
        blackbox_strict=True,
        seed=0,
    ) as binding:
        with PolicyClient(binding.client_metadata()) as client:
            client.reset(state, {"seed": 1})
            first = client.act(state)
            client.update(state, 0.0, False)
            second = client.act(state)
        assert first.dtype == np.float32
        expected = [1.5, 0.0] if observation == "box" else [0.25, 2.0]
        assert first.tolist() == [expected[0] + 1, expected[1]]
        assert second.tolist() == [
            expected[0] + 2,
            expected[1] + (observation == "box"),
        ]
    metadata = json.loads((service_dir / "metadata.json").read_text())
    assert metadata["policy_artifact_id"] == binding.policy_artifact_id
    assert metadata["runtime_image"].startswith("sha256:")
    lines = (service_dir / "policy_service.log").read_text().splitlines()
    assert any("SOURCE_ISOLATION_AND_READONLY_POLICY_VERIFIED" in x for x in lines)
    assert lines and all(line.startswith("[role=policy ") for line in lines)
    assert binding.token not in (service_dir / "command.json").read_text()
    _assert_removed(service_dir, backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_strict_import_violations_are_refused(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host refuses before launch; the container worker refuses on load too."""
    policy = tmp_path / "submission" / "approach.py"
    policy.parent.mkdir()
    policy.write_text(
        "import gymnasium\nSECRET = 'SOURCE-SECRET-7d1'\n"
        + _isolated_policy(tmp_path / "unused", "box")
    )

    def start(service_dir: Path, strict: bool = True) -> Any:
        return policy_service(
            policy,
            _SPACES["box"],
            service_dir,
            backend=backend,
            image=_image(backend),
            blackbox_strict=strict,
            seed=0,
        )

    with pytest.raises(StrictImportError, match="gymnasium"):
        with start(tmp_path / "host"):
            pass
    assert not (tmp_path / "host" / "command.json").exists()
    with pytest.raises(ValueError, match="strict blackbox"):
        with start(tmp_path / "whitebox", strict=False):
            pass
    assert not (tmp_path / "whitebox").exists()

    monkeypatch.setattr(policy_service_module, "check_strict_imports", lambda _: None)
    service_dir = tmp_path / "container"
    with pytest.raises(RuntimeError, match="caller failure"):
        with start(service_dir) as binding:
            with pytest.raises(PolicyServiceError) as caught:
                with PolicyClient(binding.client_metadata()) as client:
                    client.reset(np.zeros(2, dtype=np.float32))
            assert caught.value.code == "policy_error"
            assert "SOURCE-SECRET" not in str(caught.value)
            raise RuntimeError("caller failure")
    log = (service_dir / "policy_service.log").read_text()
    assert "StrictImportError" in log and "gymnasium" in log
    _assert_removed(service_dir, backend)


_LEAKY = """
SECRET = "SOURCE-SECRET-c4e8"

def leak(stage):
    raise RuntimeError(f"{stage} {SECRET} {__file__}")

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        if LOAD:
            leak("load")

    def reset(self, state, info):
        if info.get("fail"):
            leak("reset")

    def get_action(self, state):
        leak("act")
"""


@pytest.mark.parametrize("backend", _BACKENDS)
def test_caller_sees_no_policy_text_in_any_error(backend: str, tmp_path: Path) -> None:
    """Through the real container, error replies carry neither secret nor path."""
    leaks = [b"SOURCE-SECRET-c4e8", b"approach.py", b"/policy", b"Traceback"]
    replies = []
    for load in (True, False):
        policy = tmp_path / f"submission-{load}" / "approach.py"
        policy.parent.mkdir()
        policy.write_text(f"LOAD = {load}\n" + _LEAKY)
        leaks.append(str(policy.parent).encode())
        service_dir = tmp_path / f"service-{load}"
        with policy_service(
            policy,
            _SPACES["box"],
            service_dir,
            backend=backend,
            image=_image(backend),
            blackbox_strict=True,
            seed=0,
        ) as binding:
            meta = binding.client_metadata()
            state = {"__ndarray__": [0.0, 0.0], "dtype": "float32"}
            sequences: list[list[Any]] = (
                [[{"fail": True}], [{}, None]] if not load else [[{}]]
            )
            for infos in sequences:
                with socket.create_connection((meta["host"], meta["port"]), 30) as sock:
                    stream = sock.makefile("rb")
                    for info in infos:
                        request = (
                            {"cmd": "act", "state": state}
                            if info is None
                            else {"cmd": "reset", "state": state, "info": info}
                        )
                        request["token"] = meta["token"]
                        sock.sendall(json.dumps(request).encode() + b"\n")
                        replies.append(stream.readline())
                with pytest.raises(PolicyServiceError) as caught:
                    with PolicyClient(meta) as client:
                        client.reset(np.zeros(2, dtype=np.float32), infos[0])
                        client.act(np.zeros(2, dtype=np.float32))
                replies.append(str(caught.value).encode())
        log = (service_dir / "policy_service.log").read_text()
        assert "SOURCE-SECRET-c4e8" in log and "approach.py" in log
    # Load failure: raw and client. Reset failure: raw and client. Act: raw and client.
    assert len([reply for reply in replies if reply != b'{"ok": true}\n']) == 6
    for reply in replies:
        assert reply in (
            b'{"ok": true}\n',
            b'{"error": "policy_error"}\n',
            b"policy_error",
        ), reply
        for leak in leaks:
            assert leak not in reply


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("caller_fails", [False, True])
def test_image_change_is_detected_on_every_exit(
    backend: str, caller_fails: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed image is an error on normal exit and a note on a caller failure."""
    policy = tmp_path / "submission" / "approach.py"
    policy.parent.mkdir()
    policy.write_text(_isolated_policy(tmp_path / "unused", "box"))
    calls = []
    real = policy_service_module.image_identity

    def changing(backend_: str, image: str) -> str:
        calls.append(image)
        return real(backend_, image) if len(calls) == 1 else "sha256:changed"

    monkeypatch.setattr(policy_service_module, "image_identity", changing)
    expected = RuntimeError if caller_fails else ValueError
    with pytest.raises(expected) as caught:
        with policy_service(
            policy,
            _SPACES["box"],
            tmp_path / "service",
            backend=backend,
            image=_image(backend),
            blackbox_strict=True,
            seed=0,
        ):
            if caller_fails:
                raise RuntimeError("caller failure")
    if caller_fails:
        assert str(caught.value) == "caller failure"
        assert caught.value.__notes__ == ["Policy image changed during execution"]
    else:
        assert "image changed" in str(caught.value)
    assert len(calls) == 2
    _assert_removed(tmp_path / "service", backend)
