"""Float32 angle boundaries must not break object-centric replay rendering."""

import base64
from copy import deepcopy

import numpy as np
import pytest

from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from robocode.utils.env_server import encode
from robocode.utils.env_server_runtime import _render_state


@pytest.mark.parametrize("angle", [np.pi, -np.pi])
def test_render_codec_boundary_restores_state(angle):
    """Render a pi rotation through the wire codec without altering the rollout."""
    env = VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
        count_kwarg="num_obstructions",
        count_object_prefix="obstruction",
        design_counts=[0],
        eval_counts=[0],
        include_static_objects=True,
    )
    try:
        env.reset(seed=0, options={"object_count": 0})
        saved = env.get_state().copy()
        state = saved.copy()
        block = state.get_object_from_name("target_block")
        state.data[block] = state.data[block].astype(np.float64)
        state.set(block, "theta", angle)
        payload = encode(state)
        before = deepcopy(payload)
        png = _render_state(env, 0, payload, object_count=0)
        assert base64.b64decode(png).startswith(b"\x89PNG\r\n\x1a\n")
        for obj in saved:
            np.testing.assert_array_equal(env.get_state()[obj], saved[obj])
        assert payload == before
    finally:
        env.close()
