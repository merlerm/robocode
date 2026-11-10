"""Black-box environment server.

Runs on the HOST (which has the full environment source) while a sandboxed
agent runs in a container that has no environment code at all. The agent's
test scripts use the ``env_client`` module copied into its sandbox, which
connects to this server over TCP and speaks a JSON-lines protocol:

* ``{"cmd": "reset", "seed": ..., "options": ...}`` ->
  ``{"obs": ..., "info": ...}``
* ``{"cmd": "step", "action": ...}`` ->
  ``{"obs": ..., "reward": ..., "terminated": ..., "truncated": ..., "info": ...}``
* ``{"cmd": "get_state"}`` -> ``{"state": ...}``
* ``{"cmd": "set_state", "state": ...}`` -> ``{"ok": true}``
* ``{"cmd": "check_action_collision", "state": ..., "action": ...}`` ->
  ``{"collision": ...}`` (runs the env-dependent collision primitive on the
  host so the sandbox can use it without the env source)
* ``{"cmd": "close"}`` -> closes the connection

Every request must carry the per-run ``token`` so that other hosts on the
network cannot drive the environment. Each client connection gets its own
fresh environment instance, so the agent can run parallel test scripts.

With ``strict=True``, dispatch is reduced to reset, step, and render_state (plus
close at the connection layer); snapshot, conversion, collision, and primitive
proxy commands are rejected even if a modified client sends them directly.

Security: the protocol is JSON only, never pickle; unpickling
agent-controlled bytes on the host would be a sandbox escape. Error replies
contain only the exception type and message, not the traceback, because
traceback frames include environment source lines that the agent must not
see; full tracebacks go to the server log on the host.

This module holds the codecs, space serialization and client metadata writer.
Service ownership lives in :mod:`robocode.runtime.environment`; the serving loop
lives in :mod:`robocode.utils.env_server_runtime`. Keeping runtime render and
primitive imports separate avoids an import cycle through the approaches.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Space
from relational_structs.spaces import ObjectCentricStateSpace

from robocode.runtime.environment import EnvironmentBinding

# Source of the client module that approaches copy into blackbox sandboxes.
ENV_CLIENT_SRC: Path = Path(__file__).parent / "env_client.py"

_NDARRAY_TAG = "__ndarray__"

# Codec registry: the extension point for environments whose observations or
# states are not numpy arrays. Register an encoder/decoder pair here (and
# mirror the decoding in env_client.py if the agent should receive that
# type). Keys of _DECODERS are the JSON tag names used on the wire.
_ENCODERS: dict[type, tuple[str, Callable[[Any], Any]]] = {}
_DECODERS: dict[str, Callable[[Any], Any]] = {}


def register_codec(
    cls: type,
    tag: str,
    encode_fn: Callable[[Any], Any],
    decode_fn: Callable[[Any], Any],
) -> None:
    """Register a codec for a custom observation/state type.

    ``encode_fn`` must return a JSON-serializable value; on the wire the
    object becomes ``{tag: encode_fn(obj)}`` and ``decode_fn`` receives the
    tagged value back.
    """
    _ENCODERS[cls] = (tag, encode_fn)
    _DECODERS[tag] = decode_fn


def _register_builtin_codecs() -> None:
    """Register codecs for non-Box observation/state types shipped with robocode."""
    from relational_structs import (  # pylint: disable=import-outside-toplevel
        ObjectCentricState,
    )

    from robocode.utils.object_centric_codec import (  # pylint: disable=import-outside-toplevel
        OCS_TAG,
        decode_object_centric_state,
        encode_object_centric_state,
    )

    register_codec(
        ObjectCentricState,
        OCS_TAG,
        encode_object_centric_state,
        decode_object_centric_state,
    )


_register_builtin_codecs()


def encode(obj: Any) -> Any:
    """Encode a Python object as tagged JSON for the wire."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(
                    f"Cannot encode dict key {key!r}: JSON requires string keys"
                )
        return {key: encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(value) for value in obj]
    # Prefer an exact registration, then allow a codec registered for a base class.
    # Kinematic3D observations are family-specific ObjectCentricState subclasses and
    # intentionally share the ObjectCentricState wire format.
    codec = _ENCODERS.get(type(obj))
    if codec is None:
        codec = next(
            (
                registered_codec
                for cls, registered_codec in _ENCODERS.items()
                if isinstance(obj, cls)
            ),
            None,
        )
    if codec is not None:
        tag, encode_fn = codec
        return {tag: encode_fn(obj)}
    raise TypeError(
        f"No codec for type {type(obj).__name__}; register one with "
        "robocode.utils.env_server.register_codec"
    )


def decode(obj: Any) -> Any:
    """Decode tagged JSON from the wire back into Python objects."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            return np.array(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
        for tag, decode_fn in _DECODERS.items():
            if tag in obj:
                return decode_fn(obj[tag])
        return {key: decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode(value) for value in obj]
    return obj


def decode_registered_codec(obj: dict[str, Any]) -> tuple[bool, Any]:
    """Decode a registered-codec leaf (e.g. an ObjectCentricState payload).

    Returns ``(True, value)`` if *obj* carries a registered codec tag, else
    ``(False, None)``. Lets the handle-aware ``decode_ref`` reconstruct by-value
    custom types (their payloads carry no nested handles) the same way ``decode``
    does, so a local ObjectCentricState passed to a host remote-module primitive
    rebuilds as a real state instead of a raw dict.
    """
    for tag, decode_fn in _DECODERS.items():
        if tag in obj:
            return True, decode_fn(obj[tag])
    return False, None


def serialize_space(space: Space[Any]) -> dict[str, Any]:
    """Serialize a gym space's metadata for the agent's env_spaces.json.

    The extension point for environments with non-Box spaces: add a branch
    here and mirror it in env_client.SpaceInfo.
    """
    if isinstance(space, Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "dtype": str(space.dtype),
        }
    if isinstance(space, ObjectCentricStateSpace):
        # A variable-count env's observation space. It carries no feature names, so the
        # env attaches `type_features` (needed to reconstruct states on both sides).
        type_features = getattr(space, "type_features", None)
        if type_features is None:
            raise TypeError(
                "ObjectCentricStateSpace has no `type_features`; the env must attach it "
                "(VariableObjectCountEnv does) for blackbox serialization."
            )
        from robocode.utils.object_centric_codec import (  # pylint: disable=import-outside-toplevel
            serialize_object_centric_space,
        )

        return serialize_object_centric_space(type_features)
    raise TypeError(
        f"No serializer for space type {type(space).__name__}; blackbox mode "
        "supports Box and ObjectCentric spaces. Add a branch in "
        "robocode.utils.env_server.serialize_space and mirror it in "
        "env_client.SpaceInfo"
    )


def write_env_spaces(
    sandbox_dir: Path,
    *,
    container_backend: str,
    binding: EnvironmentBinding,
    observation_space: Space[Any],
    action_space: Space[Any],
    max_steps: int | None,
    primitives_manifest: list[dict[str, Any]] | None = None,
) -> None:
    """Write client metadata; Apptainer remaps the endpoint when it starts its relay."""
    meta = {
        "host": binding.client_host(container_backend),
        "port": binding.port,
        "token": binding.token,
        "observation_space": serialize_space(observation_space),
        "action_space": serialize_space(action_space),
        "max_steps": max_steps,
        "primitives": primitives_manifest or [],
        "strict": binding.strict,
        "service_id": binding.service_id,
        "request_timeout_s": 600.0,
    }
    (sandbox_dir / "env_spaces.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
