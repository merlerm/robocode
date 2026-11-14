"""Runner-side environment handle; simulator code stays in the service container."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import imageio.v3 as iio
import numpy as np
from gymnasium.spaces import Box

from robocode.environments.base_env import BaseEnv
from robocode.environments.variable_count import VariableCountEnv
from robocode.runtime.binding import EnvironmentBinding
from robocode.utils.env_client import BlackboxEnv, _ObjectCentricObservationSpace


def _box(spec: dict[str, Any]) -> Box:
    if spec["type"] != "Box":
        raise ValueError(f"Expected a Box space, got {spec['type']!r}")
    return Box(
        np.asarray(spec["low"], dtype=spec["dtype"]),
        np.asarray(spec["high"], dtype=spec["dtype"]),
        dtype=spec["dtype"],
    )


class RemoteEnvironment(BaseEnv):
    """Strict rollouts using metadata from a frozen environment worker.

    By default policies see what the strict ``env_client`` gives agent scripts: a
    Box, or the client's object-centric space and states. ``observation_space`` and
    ``convert_state`` substitute a source-access view. A service that reports
    ``counts`` yields a :class:`RemoteVariableCountEnvironment`.
    """

    def __new__(
        cls, binding: EnvironmentBinding, *_args: Any, **_kwargs: Any
    ) -> RemoteEnvironment:
        if cls is RemoteEnvironment and "counts" in binding.metadata:
            return super().__new__(RemoteVariableCountEnvironment)
        return super().__new__(cls)

    def __init__(
        self,
        binding: EnvironmentBinding,
        max_steps: int,
        *,
        observation_space: Any = None,
        convert_state: Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        self.binding = binding
        self._metadata = {
            **binding.metadata,
            "host": binding.host,
            "port": binding.port,
            "token": binding.token,
            "strict": binding.strict,
            "max_steps": max_steps,
        }
        self.action_space = _box(self._metadata["action_space"])
        spec = self._metadata["observation_space"]
        if observation_space is None:
            if spec["type"] == "Box":
                observation_space = _box(spec)
            elif spec["type"] == "ObjectCentric":
                observation_space = _ObjectCentricObservationSpace(spec)
            else:
                raise ValueError(f"Unsupported observation space {spec['type']!r}")
        self.observation_space = observation_space
        self._convert_state = convert_state or (lambda state: state)
        self._client: BlackboxEnv | None = None
        self._render_client: BlackboxEnv | None = None
        self._reset_seed: int | None = None
        self._reset_options: dict | None = None
        # The state as the client decoded it, before convert_state.
        self._state: Any = None

    @property
    def env_description(self) -> str | None:
        """Description supplied by the container worker."""
        return self._metadata["description"]

    @property
    def env_description_blackbox(self) -> str | None:
        """Source-free description supplied by the worker."""
        return self._metadata["description_blackbox"]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Use a fresh connection for each rollout, including forked evaluators."""
        self.close()
        self._client = BlackboxEnv(self._metadata)
        self._state, info = self._client.reset(seed=seed, options=options)
        self._reset_seed = seed
        self._reset_options = dict(options) if options is not None else None
        return self._convert_state(self._state), info

    def step(self, action):
        assert self._client is not None, "reset must precede step"
        self._state, reward, terminated, truncated, info = self._client.step(action)
        return self._convert_state(self._state), reward, terminated, truncated, info

    def get_state(self):
        return self._convert_state(self._state.copy())

    def set_state(self, state):
        raise NotImplementedError("Strict environment services do not expose set_state")

    def render(self):
        assert self._client is not None, "reset must precede render"
        # Restoring an observation does not restore all simulator internals.
        if self._render_client is None:
            self._render_client = BlackboxEnv(self._metadata)
            self._render_client.reset(
                seed=self._reset_seed, options=self._reset_options
            )
        state = (
            self._state.tolist()
            if isinstance(self._state, np.ndarray)
            else self._client._encode_state(  # pylint: disable=protected-access
                self._state
            )
        )
        reply = self._render_client._request(  # pylint: disable=protected-access
            {"cmd": "render_state", "state": state}
        )
        return iio.imread(
            base64.b64decode(reply["png_base64"], validate=True), extension=".png"
        )

    def close(self) -> None:
        if self._render_client is not None:
            self._render_client.close()
            self._render_client = None
        if self._client is not None:
            self._client.close()
            self._client = None


class RemoteVariableCountEnvironment(RemoteEnvironment, VariableCountEnv):
    """A remote environment whose count schedule comes from service metadata."""

    def __init__(self, binding: EnvironmentBinding, max_steps: int, **kwargs: Any):
        super().__init__(binding, max_steps, **kwargs)
        # Copied so the schedule stays readable after the service stops.
        counts = binding.metadata["counts"]
        self._design_counts = [int(c) for c in counts["design"]]
        self._eval_counts = [int(c) for c in counts["eval"]]
        self._count_steps = {int(c): int(n) for c, n in counts["max_steps"].items()}
        self._current_count: int | None = None

    @property
    def design_counts(self) -> list[int]:
        return list(self._design_counts)

    @property
    def eval_counts(self) -> list[int]:
        return list(self._eval_counts)

    @property
    def current_count(self) -> int:
        assert self._current_count is not None, "reset must precede current_count"
        return self._current_count

    def max_steps_for_count(self, count: int) -> int:
        if count not in self._count_steps:
            raise ValueError(
                f"Service reports no step limit for object count {count}; "
                f"known counts: {sorted(self._count_steps)}"
            )
        return self._count_steps[count]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        state, info = super().reset(seed=seed, options=options)
        self._current_count = int(info["object_count"])
        return state, info
