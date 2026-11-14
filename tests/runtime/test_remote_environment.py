"""Remote environments, count schedules and evaluator plumbing without containers."""

# pylint: disable=missing-function-docstring,protected-access,no-member
# pylint: disable=unidiomatic-typecheck
import base64
import json
from contextlib import contextmanager
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
from gymnasium.spaces import Box
from hydra.utils import instantiate
from omegaconf import OmegaConf
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from robocode.approaches.base_approach import BaseApproach
from robocode.environments.variable_count import VariableCountEnv
from robocode.runtime import container_evaluation, remote_environment
from robocode.runtime.binding import EnvironmentBinding
from robocode.runtime.environment import environment_service
from robocode.runtime.evaluation import evaluate_policy, run_determinism_check
from robocode.runtime.evaluation_worker import build_environment
from robocode.runtime.remote_environment import (
    RemoteEnvironment,
    RemoteVariableCountEnvironment,
)
from robocode.utils.env_client import _decode, _ObjectCentricObservationSpace
from robocode.utils.env_server import encode, serialize_space
from robocode.utils.object_centric_codec import object_centric_space_from_spec

_COUNT_CONFIG = {
    "_target_": "robocode.environments.variable_object_count_env."
    "VariableObjectCountEnv",
    "constant_object_env_path": "kinder.envs.kinematic2d.obstruction2d:"
    "Obstruction2DEnv",
    "count_kwarg": "num_obstructions",
    "count_object_prefix": "obstruction",
    "design_counts": [0, 1],
    "eval_counts": [0, 1, 2],
}
_BOX_CONFIG = {
    "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
    "env_id": "kinder/Motion2D-p0-v0",
}
# Small per-count limits keep rollouts short; they must come from this metadata.
_STEPS = {"0": 2, "1": 3, "2": 5}


@pytest.fixture(scope="module", name="metadata")
def fixture_metadata():
    """Service metadata as the artifact worker reports it, with short limits."""
    env = instantiate(OmegaConf.create(_COUNT_CONFIG))
    try:
        return {
            "action_space": serialize_space(env.action_space),
            "observation_space": serialize_space(env.observation_space),
            "description": env.env_description,
            "description_blackbox": env.env_description_blackbox,
            "counts": {
                "design": env.design_counts,
                "eval": env.eval_counts,
                "max_steps": _STEPS,
            },
        }
    finally:
        env.close()


def _binding(metadata, port=9, token="t"):
    return EnvironmentBinding("svc", "127.0.0.1", port, token, True, "a1", metadata)


def test_counts_make_a_variable_count_env_readable_after_close(metadata):
    env = RemoteEnvironment(_binding(metadata), 10)
    assert isinstance(env, RemoteVariableCountEnvironment)
    assert isinstance(env, VariableCountEnv)
    assert isinstance(env.observation_space, _ObjectCentricObservationSpace)
    assert env.observation_space.get_type("rectangle").name == "rectangle"
    env.close()
    assert env.design_counts == [0, 1]
    assert env.eval_counts == [0, 1, 2]
    assert [env.max_steps_for_count(c) for c in (0, 1, 2)] == [2, 3, 5]
    with pytest.raises(ValueError, match="no step limit for object count 7"):
        env.max_steps_for_count(7)


def test_fixed_and_countless_services_stay_plain(metadata):
    countless = {k: v for k, v in metadata.items() if k != "counts"}
    env = RemoteEnvironment(_binding(countless), 10)
    assert type(env) is RemoteEnvironment
    assert isinstance(env.observation_space, _ObjectCentricObservationSpace)
    box = {**countless, "observation_space": countless["action_space"]}
    env = RemoteEnvironment(_binding(box), 10)
    assert type(env) is RemoteEnvironment
    assert type(env.observation_space) is Box


def test_host_space_round_trips_through_serialization(metadata):
    spec = metadata["observation_space"]
    space = object_centric_space_from_spec(spec)
    assert isinstance(space, ObjectCentricStateSpace)
    assert serialize_space(space) == spec
    assert space.get_type("rectangle") in space.types
    env = RemoteEnvironment(_binding(metadata), 10, observation_space=space)
    assert env.observation_space is space


def test_render_cannot_change_hidden_rollout_state(monkeypatch):
    clients = []
    png = iio.imwrite("<bytes>", np.zeros((3, 4, 3), dtype=np.uint8), extension=".png")

    class Client:
        """Model a renderer that mutates hidden simulator state."""

        def __init__(self, _metadata):
            self.hidden = self.position = 0
            self.closed = False
            self.reset_args = None
            clients.append(self)

        def reset(self, **kwargs):
            self.reset_args = kwargs
            return np.array([0.0]), {}

        def step(self, _action):
            self.position += 1 + self.hidden
            return np.array([float(self.position)]), 0, False, False, {}

        def _request(self, request):
            assert request["cmd"] == "render_state"
            self.hidden = 100  # A partial state restore changes later dynamics.
            return {"png_base64": base64.b64encode(png).decode()}

        def close(self):
            self.closed = True

    monkeypatch.setattr(remote_environment, "BlackboxEnv", Client)
    space = {"type": "Box", "low": [-1000], "high": [1000], "dtype": "float32"}
    env = RemoteEnvironment(
        _binding({"action_space": space, "observation_space": space}), 10
    )
    env.reset(seed=17, options={"object_count": 2})
    assert env.step(np.zeros(1))[0].tolist() == [1.0]
    assert env.render().shape == (3, 4, 3)
    assert env.step(np.zeros(1))[0].tolist() == [2.0]
    env.render()
    assert len(clients) == 2
    assert clients[1].reset_args == {"seed": 17, "options": {"object_count": 2}}
    env.reset(seed=18, options={"object_count": 5})
    assert all(client.closed for client in clients[:2])
    env.render()
    assert clients[-1].reset_args == {"seed": 18, "options": {"object_count": 5}}
    env.close()
    assert all(client.closed for client in clients)


class _CountingPolicy(BaseApproach):
    """Check each state holds as many obstructions as the pinned count."""

    def reset(self, state, info):
        super().reset(state, info)
        rectangle = self._state_space.get_type("rectangle")
        obstructions = [
            o for o in state.get_objects(rectangle) if o.name.startswith("obstruction")
        ]
        assert len(obstructions) == info["object_count"]

    def _get_action(self):
        return np.zeros(self._action_space.shape, dtype=self._action_space.dtype)


def test_count_sweep_against_a_local_service(metadata, tmp_path):
    with environment_service(
        json.dumps(_COUNT_CONFIG), tmp_path / "service", strict=True
    ) as served:
        binding = _binding(metadata, served.port, served.token)
        env = RemoteEnvironment(binding, 1000)
        try:
            state, info = env.reset(seed=4, options={"object_count": 2})
            assert info["object_count"] == env.current_count == 2
            copy = env.get_state()
            assert copy is not state and copy.allclose(state)
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert env.render().ndim == 3
            policy = _CountingPolicy(env.action_space, env.observation_space, 0, {})
            counts = [0, 1, 2, 2]
            results = evaluate_policy(
                env,
                policy,
                [1, 2, 3, 4],
                max_steps=1000,
                timeout=60,
                render=False,
                output_dir=tmp_path,
                eval_counts=counts,
            )
            replay = run_determinism_check(
                env,
                policy,
                [1, 2, 3, 4],
                num_episodes=3,
                max_steps=1000,
                timeout=60,
                eval_counts=counts,
            )
        finally:
            env.close()
    episodes = results["per_episode"]
    assert results["num_crashed_episodes"] == 0, episodes
    assert [e["object_count"] for e in episodes] == counts
    assert [e["num_steps"] for e in episodes] == [2, 3, 5, 5]
    assert {c: b["n"] for c, b in results["by_count"].items()} == {0: 1, 1: 1, 2: 2}
    assert results["largest_count_any_solved"] is None
    assert replay["determinism_check_agreement_rate"] == 1.0
    assert [e["num_steps"] for e in replay["determinism_check"]["episodes"]] == [
        2,
        3,
        5,
    ]


def test_worker_requires_counts_exactly_for_variable_count_services(metadata):
    request = {"max_steps": 10, "eval_seeds": [1, 2], "eval_counts": [0, 2]}
    assert isinstance(
        build_environment(request, _binding(metadata), whitebox=False),
        VariableCountEnv,
    )
    with pytest.raises(ValueError, match="parallel"):
        build_environment(
            {**request, "eval_counts": [0]}, _binding(metadata), whitebox=False
        )
    with pytest.raises(ValueError, match="exactly"):
        build_environment(
            {**request, "eval_counts": None}, _binding(metadata), whitebox=False
        )
    fixed = {k: v for k, v in metadata.items() if k != "counts"}
    with pytest.raises(ValueError, match="exactly"):
        build_environment(request, _binding(fixed), whitebox=False)


def test_whitebox_worker_gives_the_local_simulation_view(metadata):
    box = instantiate(OmegaConf.create(_BOX_CONFIG))
    try:
        box_metadata = {
            "action_space": serialize_space(box.action_space),
            "observation_space": serialize_space(box.observation_space),
            "description": None,
            "description_blackbox": None,
        }
        obs, _ = box.reset(seed=0)
    finally:
        box.close()
    request = {
        "max_steps": 10,
        "eval_seeds": [1],
        "eval_counts": None,
        "environment_config": _BOX_CONFIG,
    }
    env = build_environment(request, _binding(box_metadata), whitebox=True)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    assert np.array_equal(
        env.observation_space.vectorize(env.observation_space.devectorize(obs)), obs
    )

    request = {**request, "eval_counts": [2], "environment_config": _COUNT_CONFIG}
    env = build_environment(request, _binding(metadata), whitebox=True)
    assert isinstance(env, VariableCountEnv)
    assert isinstance(env.observation_space, ObjectCentricStateSpace)
    local = instantiate(OmegaConf.create(_COUNT_CONFIG))
    try:
        real, _ = local.reset(seed=3, options={"object_count": 2})
    finally:
        local.close()
    converted = env._convert_state(_decode(json.loads(json.dumps(encode(real)))))
    assert type(converted) is type(real)
    assert converted.allclose(real)
    rectangle = env.observation_space.get_type("rectangle")
    assert converted.get_objects(rectangle) == real.get_objects(rectangle)


def test_container_request_carries_counts(metadata, tmp_path, monkeypatch):
    policy = tmp_path / "submission/approach.py"
    policy.parent.mkdir()
    policy.write_text("class GeneratedApproach: pass\n")
    binding = _binding({**metadata, "runtime_source_id": "r1"})
    seen = {}

    @contextmanager
    def service(*_args, **_kwargs):
        yield binding

    @contextmanager
    def container(_backend, _image, output_dir, **_kwargs):
        request = json.loads((output_dir / "input/request.json").read_text())
        seen["request"] = request
        (output_dir / "output/ready").touch()
        (output_dir / "output/results.json").write_text(
            json.dumps(
                {
                    "by_count": {"0": {"n": 1}, "2": {"n": 1}},
                    "policy_artifact_id": request["policy_id"],
                    "environment_artifact_id": request["environment_artifact_id"],
                    "evaluation_runtime_source_id": request["runtime_source_id"],
                }
            )
        )
        proc = SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0, returncode=0)
        yield proc, {}, SimpleNamespace(join=lambda: None)

    monkeypatch.setattr(container_evaluation, "environment_service", service)
    monkeypatch.setattr(container_evaluation, "container_process", container)
    monkeypatch.setattr(container_evaluation, "image_identity", lambda *_: "image")
    artifact = SimpleNamespace(artifact_id="a1", manifest={"config": _COUNT_CONFIG})
    options = {
        "backend": "apptainer",
        "environment_image": "env.sif",
        "evaluation_image": "eval.sif",
        "eval_seeds": [5, 6],
        "replicate_seed": 0,
        "max_steps": 10,
        "timeout": 1,
        "render": False,
        "determinism_episodes": 0,
    }
    results = container_evaluation.evaluate_in_container(
        policy, artifact, tmp_path / "evaluation", eval_counts=[0, 2], **options
    )
    assert list(results["by_count"]) == [0, 2]
    request = seen["request"]
    assert request["eval_counts"] == [0, 2]
    assert request["binding"]["metadata"]["counts"] == metadata["counts"]
    assert "environment_config" not in request
    with pytest.raises(ValueError, match="exactly"):
        container_evaluation.evaluate_in_container(
            policy, artifact, tmp_path / "missing", **options
        )
