"""Shared scene observations across layouts, codecs and restoration."""

# pylint: disable=protected-access,missing-function-docstring
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from relational_structs import Object, ObjectCentricState

from robocode.runtime import validation_worker
from robocode.runtime.validation_worker import state_schema
from robocode.utils.env_client import _decode, _encode
from robocode.utils.env_server import decode, encode, serialize_space

REQUIRED = {"robot", "target_block", "target_surface"}


def scene_schema(state):
    return state_schema(state, required_objects=REQUIRED)


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
            current_schema = scene_schema(state)
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
        assert not WALLS.intersection(
            env._current_backend._object_centric_env.get_state().get_object_names()
        )
        env.step(np.zeros(5, dtype=np.float32))
        # The per-count planner still sees its original vector, not extra walls.
        assert env.to_box(state).shape == env._current_backend.observation_space.shape
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["", "dyn"])
def test_added_static_object_changes_contents_not_contract(domain):
    env = instantiate(scene_config(domain))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 2})
        schema, space = scene_schema(state), serialize_space(env.observation_space)
        inner = env._current_backend._object_centric_env
        source = inner.initial_constant_state.get_object_from_name("table")
        extra = Object("scene_wall", source.type)
        values = inner.initial_constant_state.data[source].copy()
        inner._initial_constant_state.data[extra] = values
        for field, value in {
            "x": 1.7 if domain else 0.8,
            "y": 1.2 if domain else 0.45,
            "width": 0.03,
            "height": 0.12,
        }.items():
            inner._initial_constant_state.set(extra, field, value)
        observed, *_ = env.step(np.zeros(5, dtype=np.float32))
        assert "scene_wall" in observed.get_object_names()
        assert scene_schema(observed) == schema
        assert env.observation_space.contains(observed)
        assert (
            env.to_box(observed).shape == env._current_backend.observation_space.shape
        )
        assert serialize_space(env.observation_space) == space
        assert env.infer_count(observed) == 2
        env.set_state(decode(_encode(_decode(encode(observed)))))
        assert observed.allclose(env.get_state())
        assert extra not in inner.get_state()
        following, *_ = env.step(np.zeros(5, dtype=np.float32))
        assert "scene_wall" in following.get_object_names()
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
            for obj in plain_state:
                np.testing.assert_allclose(scene_state.data[obj], plain_state.data[obj])
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


@pytest.mark.parametrize("domain", ["", "dyn"])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_restore_rejects_different_fixed_geometry(domain, mutation):
    env = instantiate(scene_config(domain))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 0})
        wall = state.get_object_from_name("left_wall")
        if mutation == "missing":
            state.data.pop(wall)
        else:
            state.set(wall, "x", state.get(wall, "x") + 0.1)
        with pytest.raises(ValueError, match="different fixed geometry"):
            env.set_state(state)
    finally:
        env.close()


@pytest.mark.parametrize(
    "mutation", ["robot", "target_block", "target_surface", "count", "prefix"]
)
def test_scene_probe_rejects_broken_required_objects(monkeypatch, mutation):
    config = scene_config("")
    env = instantiate(config)
    reset = env.reset

    def altered_reset(**kwargs):
        state, info = reset(**kwargs)
        if mutation in REQUIRED:
            state.data.pop(state.get_object_from_name(mutation))
        else:
            obj = state.get_object_from_name("table")
            name = "obstruction999" if mutation == "count" else "obstruction_guard"
            state.data[Object(name, obj.type)] = state.data[obj].copy()
        return state, info

    monkeypatch.setattr(env, "reset", altered_reset)
    monkeypatch.setattr(validation_worker, "instantiate", lambda _: env)
    artifact = SimpleNamespace(
        manifest={"config": OmegaConf.to_container(config)}, verify_imports=lambda: None
    )
    with pytest.raises(ValueError, match="required scene objects|wrong object count"):
        validation_worker.probe(artifact)


def test_non_scene_schema_retains_object_identity():
    env = instantiate(scene_config(""))
    try:
        state, _ = env.reset(seed=4, options={"object_count": 0})
        original = state_schema(state)
        wall = state.get_object_from_name("table")
        state.data[Object("extra_wall", wall.type)] = state.data[wall].copy()
        assert state_schema(state) != original
    finally:
        env.close()


def test_observed_added_wall_blocks_robot_motion():
    blocked = instantiate(scene_config(""))
    control = instantiate(scene_config(""))
    try:
        state, _ = blocked.reset(seed=4, options={"object_count": 0})
        control.reset(seed=4, options={"object_count": 0})
        robot = state.get_object_from_name("robot")
        x = state.get(robot, "x")
        inner = blocked._current_backend._object_centric_env
        table = state.get_object_from_name("table")
        wall = Object("blocking_wall", table.type)
        inner._initial_constant_state.data[wall] = state.data[table].copy()
        for field, value in {
            "x": x + state.get(robot, "base_radius") + 0.005,
            "y": state.get(robot, "y"),
            "theta": 0,
            "width": 0.004,
            "height": 0.2,
        }.items():
            inner._initial_constant_state.set(wall, field, value)
        action = np.array([0.01, 0, 0, 0, 0], dtype=np.float32)
        stopped, *_ = blocked.step(action)
        moved, *_ = control.step(action)
        assert wall in stopped
        assert stopped.get(robot, "x") == pytest.approx(x)
        assert moved.get(robot, "x") == pytest.approx(x + action[0])
    finally:
        blocked.close()
        control.close()
