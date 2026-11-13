"""Client for a frozen-policy service: call a policy without seeing its code.

Copy this file next to the scripts that use it; it needs only the standard library
and numpy::

    from policy_client import PolicyClient

    with PolicyClient(meta) as policy:
        policy.reset(state, info)
        action = policy.act(state)
        policy.update(next_state, reward, done, info)

Each client is one connection to a freshly loaded policy, which keeps its internal
state across calls. States may be numpy vectors, ``relational_structs``
object-centric states, or the ``env_client`` mirror of those. Any failure raises
:class:`PolicyServiceError` with one public code and ends the connection.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

import numpy as np

PUBLIC_CODES = frozenset(
    {
        "policy_error",
        "policy_timeout",
        "protocol_error",
        "unauthorized",
        "connection_closed",
        "busy",
    }
)


class PolicyServiceError(RuntimeError):
    """A failed policy call; carries only a public code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _type_depth(typ: Any) -> int:
    depth, parent = 0, typ.parent
    while parent is not None:
        depth, parent = depth + 1, parent.parent
    return depth


def _object_centric_payload(state: Any) -> dict[str, Any]:
    """The ``object_centric_codec`` payload, built from the state's public API."""
    types = sorted(state.type_features, key=lambda t: (_type_depth(t), t.name))
    return {
        "types": [
            {
                "name": t.name,
                "parent": t.parent.name if t.parent is not None else None,
                "features": list(state.type_features[t]),
            }
            for t in types
        ],
        "objects": [
            {
                "name": obj.name,
                "type": obj.type.name,
                "features": [float(v) for v in state.vec([obj])],
            }
            for obj in sorted(state, key=lambda o: o.name)
        ],
        "state_cls": type(state).__name__,
    }


def encode_value(value: Any) -> Any:
    """Encode a state or info value in the environment service's tagged JSON."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "type_features") and hasattr(value, "vec"):
        return {"__ocs__": _object_centric_payload(value)}
    if isinstance(value, dict):
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    return value


def encode_state(state: Any) -> Any:
    """Encode an object-centric state, or anything else as a numpy vector."""
    if hasattr(state, "type_features") and hasattr(state, "vec"):
        return encode_value(state)
    return encode_value(np.asarray(state))


class PolicyClient:
    """One connection to a policy service, holding one policy instance."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self._token: str = meta["token"]
        self._action_dtype = np.dtype(meta["action_space"]["dtype"])
        self._timeout = float(meta.get("request_timeout_s", 120.0))
        self._sock: socket.socket | None = socket.create_connection(
            (meta["host"], meta["port"]), timeout=min(10.0, self._timeout)
        )
        self._buffer = b""

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise PolicyServiceError("connection_closed")
        deadline = time.monotonic() + self._timeout
        try:
            self._sock.settimeout(self._timeout)
            line = json.dumps({**payload, "token": self._token}).encode("utf-8")
            self._sock.sendall(line + b"\n")
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self._sock.settimeout(remaining)
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise ConnectionError
                self._buffer += chunk
            line, self._buffer = self._buffer.split(b"\n", 1)
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise ValueError
        except TimeoutError:
            self.close()
            raise PolicyServiceError("policy_timeout") from None
        except (OSError, ValueError):
            self.close()
            raise PolicyServiceError("connection_closed") from None
        if "error" in reply:
            self.close()
            code = reply["error"]
            raise PolicyServiceError(code if code in PUBLIC_CODES else "protocol_error")
        return reply

    def reset(self, state: Any, info: dict[str, Any] | None = None) -> None:
        """Start an episode from *state*."""
        self._request(
            {
                "cmd": "reset",
                "state": encode_state(state),
                "info": encode_value(info or {}),
            }
        )

    def act(self, state: Any) -> np.ndarray:
        """The policy's action for *state*."""
        reply = self._request({"cmd": "act", "state": encode_state(state)})
        return np.asarray(reply["action"], dtype=self._action_dtype)

    def update(
        self, state: Any, reward: float, done: bool, info: dict[str, Any] | None = None
    ) -> None:
        """Report the outcome of the last action, as the evaluator does after a step."""
        self._request(
            {
                "cmd": "update",
                "state": encode_state(state),
                "reward": float(reward),
                "done": bool(done),
                "info": encode_value(info or {}),
            }
        )

    def close(self) -> None:
        """Close the connection; the service then stops this policy's worker."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
