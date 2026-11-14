"""Shared scene observations across layouts, codecs and restoration."""

# pylint: disable=protected-access,missing-function-docstring
from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from relational_structs import Object, ObjectCentricState

from robocode.runtime.validation_worker import state_schema
from robocode.utils.env_client import _decode, _encode
from robocode.utils.env_server import decode, encode, serialize_space

ROOT = Path(__file__).resolve().parents[2]
WALLS = {"left_wall", "right_wall", "bottom_wall", "top_wall", "table"}


def scene_config(domain):
    return OmegaConf.load(
        ROOT / f"experiments/conf/environment/{domain}obstruction2d_scene.yaml"
    )


@pytest.mark.parametrize("domain", ["", "dyn"])
def test_all_counts_expose_geometry_through_blackbox_codec(domain):
    env = instantiate(scene_config(domain))
    try:
        schema = None
        for count in env.eval_counts:
            state, info = env.reset(seed=4, options={"object_count": count})
            client_state = _decode(encode(state))
            assert WALLS <= set(client_state.get_object_names())
            assert info["object_count"] == count
            assert env.infer_count(state) == count
            assert env.observation_space.contains(state)
            for name in WALLS:
                obj = client_state.get_object_from_name(name)
                assert client_state.get(obj, "static") == 1
                assert client_state.get(obj, "width") > 0
            current_schema = state_schema(state)
            if schema is not None:
                assert current_schema == schema
            schema = current_schema
            following, *_ = env.step(np.zeros(5, dtype=np.float32))
            assert WALLS <= set(following.get_object_names())
        assert "Static scene objects" in env.env_description_blackbox
        assert "include_static_objects=True" in env.env_description
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["", "dyn"])
def test_restore_scene_keeps_constants_out_of_dynamic_state(domain):
    env = instantiate(scene_config(domain))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 2})
        restored = decode(_encode(_decode(encode(state))))
        env.set_state(restored)
        assert state.allclose(env.get_state())
        assert not WALLS.intersection(
            env._current_backend._object_centric_env.get_state().get_object_names()
        )
        reset_state, _ = env.reset(seed=4, options={"init_state": restored})
        assert state.allclose(reset_state)
        # The per-count planner still sees its original vector, not extra walls.
        assert env.to_box(state).shape == env._current_backend.observation_space.shape
    finally:
        env.close()


def test_added_static_object_changes_contents_not_contract():
    env = instantiate(scene_config(""))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 2})
        schema, space = state_schema(state), serialize_space(env.observation_space)
        inner = env._current_backend._object_centric_env
        source = inner.initial_constant_state.get_object_from_name("table")
        extra = Object("scene_partition", source.type)
        values = inner.initial_constant_state.data[source].copy()
        inner._initial_constant_state.data[extra] = values
        for field, value in {
            "x": 0.8,
            "y": 0.45,
            "width": 0.03,
            "height": 0.12,
        }.items():
            inner._initial_constant_state.set(extra, field, value)
        observed, *_ = env.step(np.zeros(5, dtype=np.float32))
        assert "scene_partition" in observed.get_object_names()
        assert state_schema(observed) == schema
        assert serialize_space(env.observation_space) == space
        assert env.infer_count(observed) == 2
        env.set_state(decode(_encode(_decode(encode(observed)))))
        assert observed.allclose(env.get_state())
        assert extra not in inner.get_state()
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["", "dyn"])
def test_scene_visibility_preserves_transition(domain):
    config = scene_config(domain)
    scene = instantiate(config)
    config.include_static_objects = False
    ordinary = instantiate(config)
    try:
        scene_state, _ = scene.reset(seed=4, options={"object_count": 2})
        plain_state, _ = ordinary.reset(seed=4, options={"object_count": 2})
        for _ in range(3):
            for obj in plain_state:
                np.testing.assert_allclose(scene_state.data[obj], plain_state.data[obj])
            action = np.zeros(5, dtype=np.float32)
            action[0] = 0.005
            scene_state, *scene_result = scene.step(action)
            plain_state, *plain_result = ordinary.step(action)
            assert scene_result == plain_result
    finally:
        scene.close()
        ordinary.close()


def test_contract_still_distinguishes_feature_changes_and_nonfinite_state():
    env = instantiate(scene_config(""))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 0})
        robot = state.get_object_from_name("robot")
        features = dict(state.type_features)
        features[robot.type] = [*features[robot.type], "new_feature"]
        data = {obj: values.copy() for obj, values in state.data.items()}
        data[robot] = np.append(data[robot], 0.0)
        changed = ObjectCentricState(data, features)
        assert state_schema(changed) != state_schema(state)
        state.set(robot, "x", np.nan)
        with pytest.raises(ValueError, match="Nonfinite"):
            state_schema(state)
    finally:
        env.close()
