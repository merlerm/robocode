"""Runner-side environment handle; simulator code stays in the service container."""

from __future__ import annotations

import base64
from typing import Any

import imageio.v3 as iio
import numpy as np
from gymnasium.spaces import Box

from robocode.environments.base_env import BaseEnv
from robocode.runtime.binding import EnvironmentBinding
from robocode.utils.env_client import BlackboxEnv


class RemoteEnvironment(BaseEnv):
    """Strict Box-space rollouts using metadata from a frozen environment worker."""

    def __init__(self, binding: EnvironmentBinding, max_steps: int):
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
        for name in ("action_space", "observation_space"):
            spec = self._metadata[name]
            if spec["type"] != "Box":
                raise ValueError("Artifact runner currently requires Box spaces")
            setattr(
                self,
                name,
                Box(
                    np.asarray(spec["low"], dtype=spec["dtype"]),
                    np.asarray(spec["high"], dtype=spec["dtype"]),
                    dtype=spec["dtype"],
                ),
            )
        self._client: BlackboxEnv | None = None
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
        return self._state, info

    def step(self, action):
        assert self._client is not None, "reset must precede step"
        self._state, reward, terminated, truncated, info = self._client.step(action)
        return self._state, reward, terminated, truncated, info

    def get_state(self):
        return self._state.copy()

    def set_state(self, state):
        raise NotImplementedError("Strict environment services do not expose set_state")

    def render(self):
        assert self._client is not None, "reset must precede render"
        reply = self._client._request(  # pylint: disable=protected-access
            {"cmd": "render_state", "state": self._state.tolist()}
        )
        return iio.imread(
            base64.b64decode(reply["png_base64"], validate=True), extension=".png"
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
