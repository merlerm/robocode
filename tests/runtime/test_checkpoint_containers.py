"""Opt-in checkpoint acceptance across simulator families and revision handoffs."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from robocode.runtime.artifacts import export_environment, load_environment_artifact
from robocode.runtime.container_evaluation import evaluate_in_container
from robocode.runtime.environment import environment_service
from robocode.runtime.remote_environment import RemoteEnvironment
from robocode.runtime.whitebox import create_whitebox_source, load_whitebox_source

# pylint: disable=missing-function-docstring,redefined-outer-name

_CONFIGS = {
    "motion": {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    },
    "physics": {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/DynPushT2D-t1-v0",
    },
    "3d": {
        "_target_": "robocode.environments.kinder_geom3d_env.KinderGeom3DEnv",
        "env_id": "kinder/Packing3D-p1-v0",
    },
    # obstruction2d_generalized with short per-count horizons (5 + 2 * count).
    "generalized": {
        "_target_": "robocode.environments.variable_object_count_env."
        "VariableObjectCountEnv",
        "constant_object_env_path": "kinder.envs.kinematic2d.obstruction2d:"
        "Obstruction2DEnv",
        "count_kwarg": "num_obstructions",
        "count_object_prefix": "obstruction",
        "design_counts": [0, 1, 2],
        "eval_counts": [0, 1, 2, 3, 4],
        "bilevel_env_name": "obstruction2d",
        "base_steps": 5,
        "steps_per_object": 2,
    },
}
# The runner's schedule for six episodes: env.eval_counts cycled over the seeds.
_SCHEDULE = [0, 1, 2, 3, 4, 0]


@pytest.fixture(scope="module", params=["docker", "apptainer"])
def checkpoint(request):
    directory = os.environ.get("ROBOCODE_CHECKPOINT_TEST_ARTIFACTS")
    if directory is None:
        pytest.skip(
            "Set ROBOCODE_CHECKPOINT_TEST_ARTIFACTS to the prepared base artifacts"
        )
    root, backend = Path(directory), request.param
    repo = Path(__file__).resolve().parents[2]
    image = (
        "robocode-sandbox:multi-agent-runtime"
        if backend == "docker"
        else str(repo / "robocode-sandbox.sif")
    )
    artifact = load_environment_artifact(root / (backend + "-artifact"))
    view_path = root / (backend + "-whitebox")
    view = load_whitebox_source(
        view_path, json.loads((view_path / "view.json").read_text())["source_id"]
    )
    cache = {"motion": (artifact, view)}

    def get_source(family):
        if family not in cache:
            source = repo / "third-party/kindergarden"
            config = _CONFIGS.get(family, _CONFIGS["motion"])
            if family == "edited":
                source = root / (backend + "-editor")
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--shared",
                        "--quiet",
                        str(repo / "third-party/kindergarden"),
                        str(source),
                    ],
                    check=True,
                )
                module = source / "src/kinder/envs/kinematic2d/motion2d.py"
                code = module.read_text()
                anchor = (
                    "    def __init__(self, num_passages: int = 3, **kwargs) -> None:"
                )
                assert anchor in code
                code = code.replace(
                    anchor,
                    "    def step(self, action):\n"
                    "        return super().step(np.asarray(action) * 0.5)\n\n"
                    + anchor,
                )
                module.write_text(code)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(source),
                        "-c",
                        "user.name=Checkpoint test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-qam",
                        "Halve Motion2D actions for revision replay",
                    ],
                    check=True,
                )
            exported = export_environment(
                source,
                "HEAD",
                root / (backend + "-" + family + "-artifact"),
                package="kinder",
                config=config,
                runtime_image=artifact.manifest["runtime_image"],
                dependencies={
                    "uv.lock": repo / "uv.lock",
                    "pyproject.toml": repo / "pyproject.toml",
                },
            )
            filtered = create_whitebox_source(
                exported,
                root / "service-runtime",
                root / (backend + "-" + family + "-whitebox"),
            )
            cache[family] = exported, filtered
        return cache[family]

    return {"backend": backend, "image": image, "root": root, "source": get_source}


def _is_simulator(name):
    return name == "kinder" or name.startswith("kinder.")


@contextmanager
def _host_imports_no_simulator():
    """Check the block imports no simulator code, whatever earlier tests loaded."""
    loaded = {
        name: sys.modules.pop(name) for name in list(sys.modules) if _is_simulator(name)
    }
    try:
        yield
        assert not any(_is_simulator(name) for name in sys.modules)
    finally:
        sys.modules.update(loaded)


def _evaluate(
    checkpoint,
    family,
    policy,
    output,
    *,
    whitebox,
    render=False,
    eval_seeds=(42, 43),
    eval_counts=None,
):
    artifact, view = checkpoint["source"](family)
    backend = checkpoint["backend"]
    repo = Path(__file__).resolve().parents[2]
    generic_image = (
        "robocode-evaluation:multi-agent"
        if backend == "docker"
        else str(repo / "robocode-evaluation.sif")
    )
    with _host_imports_no_simulator():
        result = evaluate_in_container(
            policy,
            artifact,
            output,
            backend=backend,
            environment_image=checkpoint["image"],
            evaluation_image=checkpoint["image"] if whitebox else generic_image,
            eval_seeds=list(eval_seeds),
            replicate_seed=42,
            max_steps=3,
            timeout=10,
            render=render,
            determinism_episodes=1 if eval_counts is None else 2,
            runtime_source=checkpoint["root"] / "service-runtime",
            whitebox_source=view if whitebox else None,
            eval_counts=eval_counts,
        )
    assert result["environment_artifact_id"] == artifact.artifact_id
    assert result["num_crashed_episodes"] == 0, result["per_episode"]
    assert result["determinism_check_agreement_rate"] == 1.0
    for episode in result["per_episode"]:
        assert episode["environment_artifact_id"] == artifact.artifact_id
        assert episode["policy_artifact_id"] == result["policy_artifact_id"]
    if whitebox:
        assert result["whitebox_source_id"] == view.source_id
    else:
        command = json.loads((output / "command.json").read_text())
        assert not any("/whitebox" in arg or "/kindergarden" in arg for arg in command)
    return result


@pytest.mark.parametrize("family", ["motion", "physics", "3d"])
@pytest.mark.parametrize("whitebox", [False, True], ids=["blackbox", "whitebox"])
def test_family_rollouts_rendering_and_policy_helpers(
    checkpoint, tmp_path, family, whitebox
):
    """Exercise physics, 3D, rendering, helper files and image-native dependencies."""
    config = _CONFIGS[family]
    module, cls = config["_target_"].rsplit(".", 1)
    policy = tmp_path / "submission/approach.py"
    policy.parent.mkdir()
    (policy.parent / "helper.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def load_scale():\n"
        '    return json.loads(Path(__file__).with_name("scale.json")'
        '.read_text())["scale"]\n'
    )
    (policy.parent / "scale.json").write_text('{"scale": 0.0}\n')
    policy.write_text(f"""import numpy as np
from helper import load_scale
from pathlib import Path
{"from " + module + " import " + cls if whitebox else ""}
assert Path('/whitebox').exists() == {whitebox!r}
assert not Path('/artifact').exists()
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action = np.clip(np.zeros(action_space.shape), action_space.low, action_space.high)
        self.scale = load_scale()
        {"self.local = " + cls + "(" + repr(config['env_id']) + ")" if whitebox else "pass"}
    def reset(self, state, info):
        {"self.local.reset(seed=42)" if whitebox else "pass"}
    def get_action(self, state):
        {"self.local.step(self.action)" if whitebox else "pass"}
        return self.action * self.scale
""")
    result = _evaluate(
        checkpoint,
        family,
        policy,
        tmp_path / "evaluation",
        whitebox=whitebox,
        render=True,
    )
    assert result["num_evaluated_episodes"] == 2
    assert len(list((tmp_path / "evaluation/output/videos").glob("*.gif"))) == 2


@pytest.mark.parametrize("whitebox", [False, True], ids=["blackbox", "whitebox"])
def test_variable_count_sweep(checkpoint, tmp_path, whitebox):
    """Pinned counts, per-count limits and by-count results match the host path."""
    space_name, state_name = (
        ("ObjectCentricStateSpace", "ObjectCentricState")
        if whitebox
        else ("_ObjectCentricObservationSpace", "_ObjectCentricState")
    )
    policy = tmp_path / "submission/approach.py"
    policy.parent.mkdir()
    policy.write_text(f"""import numpy as np
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        assert type(observation_space).__name__ == {space_name!r}
        self.rectangle = observation_space.get_type("rectangle")
        self.shape = action_space.shape
    def reset(self, state, info):
        assert type(state).__name__ == {state_name!r}
        objects = state.get_objects(self.rectangle)
        count = sum(o.name.startswith("obstruction") for o in objects)
        assert count == info["object_count"], (count, info)
    def get_action(self, state):
        return np.zeros(self.shape, dtype=np.float32)
""")
    result = _evaluate(
        checkpoint,
        "generalized",
        policy,
        tmp_path / "evaluation",
        whitebox=whitebox,
        render=True,
        eval_seeds=range(100, 106),
        eval_counts=_SCHEDULE,
    )
    episodes = result["per_episode"]
    assert [e["object_count"] for e in episodes] == _SCHEDULE
    assert [e["num_steps"] for e in episodes] == [5 + 2 * c for c in _SCHEDULE]
    assert {int(c): b["n"] for c, b in result["by_count"].items()} == {
        0: 2,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }
    assert result["largest_count_all_solved"] is None
    assert result["largest_count_any_solved"] is None
    replays = result["determinism_check"]["episodes"]
    assert [e["num_steps"] for e in replays] == [5, 7]
    assert len(list((tmp_path / "evaluation/output/videos").glob("*.gif"))) == 6


@pytest.mark.parametrize("whitebox", [False, True], ids=["blackbox", "whitebox"])
def test_policy_observation_space_matches_synthesis(checkpoint, tmp_path, whitebox):
    """Whitebox policies devectorize as in local simulation; strict ones cannot."""
    policy = tmp_path / "submission/approach.py"
    policy.parent.mkdir()
    policy.write_text(f"""import numpy as np
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = observation_space
        assert hasattr(observation_space, "devectorize") == {whitebox!r}
    def reset(self, state, info):
        if {whitebox!r}:
            ocs = self.space.devectorize(state)
            assert "robot" in ocs.get_object_names()
            assert np.array_equal(self.space.vectorize(ocs), state)
    def get_action(self, state):
        return np.zeros(5)
""")
    result = _evaluate(
        checkpoint, "motion", policy, tmp_path / "evaluation", whitebox=whitebox
    )
    assert result["num_evaluated_episodes"] == 2
    assert "by_count" not in result


def test_committed_environment_round_trip(checkpoint, tmp_path):
    """E0 -> E1 -> E0 must change both local simulation and the scored server."""
    records = []
    for index, (family, delta) in enumerate(
        [("motion", 0.05), ("edited", 0.025), ("motion", 0.05)]
    ):
        policy = tmp_path / str(index) / "submission/approach.py"
        policy.parent.mkdir(parents=True)
        policy.write_text(f"""import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.local = KinderGeom2DEnv('kinder/Motion2D-p0-v0')
    def reset(self, state, info):
        self.previous = None
        a, _ = self.local.reset(seed=0)
        b, *_ = self.local.step(np.array([.05,0,0,0,0]))
        assert abs(float(b[0] - a[0]) - {delta}) < 1e-6
    def get_action(self, state):
        if self.previous is not None:
            assert abs(float(state[0] - self.previous[0]) - {delta}) < 1e-6
        self.previous = state.copy()
        return [.05,0,0,0,0]
""")
        result = _evaluate(
            checkpoint,
            family,
            policy,
            tmp_path / str(index) / "evaluation",
            whitebox=True,
        )
        records.append(
            {
                key: result[key]
                for key in [
                    "environment_artifact_id",
                    "whitebox_source_id",
                    "mean_eval_steps",
                ]
            }
        )
    assert records[0] == records[2]
    assert (
        records[0]["environment_artifact_id"] != records[1]["environment_artifact_id"]
    )
    (tmp_path / "round-trip.json").write_text(json.dumps(records, indent=2))


def test_concurrent_clients_and_services_are_independent(checkpoint, tmp_path):
    """Keep E0/E1 alive together; closing one must leave the other usable."""
    initial, _ = checkpoint["source"]("motion")
    edited, _ = checkpoint["source"]("edited")
    options = {
        "strict": True,
        "container_backend": checkpoint["backend"],
        "image": checkpoint["image"],
        "runtime_source": checkpoint["root"] / "service-runtime",
    }
    with environment_service(
        json.dumps(initial.manifest["config"]),
        tmp_path / "e0",
        artifact=initial,
        **options,
    ) as first:
        with environment_service(
            json.dumps(edited.manifest["config"]),
            tmp_path / "e1",
            artifact=edited,
            **options,
        ) as second:

            def trace(binding):
                env = RemoteEnvironment(binding, 10)
                try:
                    a, _ = env.reset(seed=42)
                    b, *_ = env.step([0.05, 0, 0, 0, 0])
                    return float(b[0] - a[0])
                finally:
                    env.close()

            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(trace, [first, second, first, second]))
            np.testing.assert_allclose(values, [0.05, 0.025, 0.05, 0.025], atol=1e-6)
        assert trace(first) == pytest.approx(0.05, abs=1e-6)


def test_normal_runner_synthesizes_then_evaluates(checkpoint, tmp_path):
    """Exercise prompt/config wiring, synthesis, snapshotting and eval in one run."""
    artifact, _ = checkpoint["source"]("motion")
    repo = Path(__file__).resolve().parents[2]
    driver = """import runpy, sys
from pathlib import Path
from contextlib import nullcontext
from tests.runtime.test_whitebox_containers import ProbeBackend, _PROBE
from robocode.utils import docker_sandbox, apptainer_sandbox
from robocode.utils.model_broker import BrokerUpstream
import robocode.approaches.agentic_base as approach
import robocode.utils.episode as episode
class RunnerProbe(ProbeBackend):
    def build_cli_cmd(self, config, **kwargs):
        assert 'read-only environment source is pinned' in config.prompt
        assert config.whitebox_source is not None
        (config.sandbox_dir / 'probe.py').write_text(_PROBE)
        return super().build_cli_cmd(config, **kwargs)
approach.create_backend = lambda cfg: RunnerProbe(cfg['backend'])
docker_sandbox._build_docker_auth_args = lambda _: nullcontext(([], {}))
apptainer_sandbox.load_broker_upstream = lambda _: BrokerUpstream('responses', 'api.openai.com', '/v1', {})
apptainer_sandbox._model_client = lambda _, command: (command, {})
def forbidden(*args, **kwargs):
    raise AssertionError('host loaded the submitted policy')
approach.load_generated_approach = episode.load_generated_approach = forbidden
runpy.run_path('experiments/run_experiment.py', run_name='__main__')
assert not any(n == 'kinder' or n.startswith('kinder.') for n in sys.modules)
print('NORMAL_RUNNER_HOST_LOADED_NEITHER_POLICY_NOR_ENVIRONMENT')
"""
    output = tmp_path / "runner"
    command = [
        sys.executable,
        "-B",
        "-c",
        driver,
        "approach=agentic",
        "approach/backend=codex_gpt6",
        "environment=motion2d_easy",
        "~environment.bilevel_env_name",
        "~environment.bilevel_env_model_kwargs",
        "primitive_level=none",
        "approach.blackbox=false",
        "approach.blackbox_strict=false",
        "mcp_tools=[render_state,render_policy]",
        "approach.telemetry=true",
        "approach.container_backend=" + checkpoint["backend"],
        "environment_runtime=artifact",
        "environment_runtime.artifact=" + str(artifact.path),
        "environment_runtime.expected_id=" + artifact.artifact_id,
        "environment_runtime.image=" + checkpoint["image"],
        "environment_runtime.evaluation_image=" + checkpoint["image"],
        "eval_seed=314159",
        "num_eval_tasks=2",
        "max_steps=2",
        "determinism_check_num_episodes=1",
        "render_videos=true",
        "hydra.run.dir=" + str(output),
    ]
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2))
    with (tmp_path / "runner.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            env=dict(os.environ, PYTHONPATH="src:."),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, (tmp_path / "runner.log").read_text()
    results = json.loads((output / "results.json").read_text())
    provenance = json.loads((output / "sandbox/source-provenance.json").read_text())
    assert results["whitebox_source_id"] == provenance["whitebox_source_id"]
    assert results["environment_artifact_id"] == artifact.artifact_id
    assert results["num_crashed_episodes"] == 0
    assert results["num_evaluated_episodes"] == 2
    assert results["determinism_check_agreement_rate"] == 1.0
    assert (
        "NORMAL_RUNNER_HOST_LOADED_NEITHER_POLICY_NOR_ENVIRONMENT"
        in (tmp_path / "runner.log").read_text()
    )


_STRICT_PROBE = """from pathlib import Path
from env_client import make_env
env = make_env()
rectangle = env.observation_space.get_type("rectangle")
for count in (0, 4):
    state, info = env.reset(seed=1, options={"object_count": count})
    assert info["object_count"] == count
    assert sum(o.name.startswith("obstruction") for o in state.get_objects(rectangle)) == count
env.close()
Path("/sandbox/approach.py").write_text('''import numpy as np
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.rectangle = observation_space.get_type("rectangle")
        self.shape = action_space.shape
    def reset(self, state, info):
        count = sum(o.name.startswith("obstruction") for o in state.get_objects(self.rectangle))
        assert count == info["object_count"], (count, info)
    def get_action(self, state):
        return np.zeros(self.shape, dtype=np.float32)
''')
print("STRICT_OBJECT_CENTRIC_PROBE_OK", flush=True)
"""


def test_strict_runner_sweeps_variable_counts(checkpoint, tmp_path):
    """The normal strict runner reports the host path's count fields."""
    artifact, _ = checkpoint["source"]("generalized")
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "probe.py").write_text(_STRICT_PROBE)
    driver = f"""import runpy, sys
from contextlib import nullcontext
from pathlib import Path
from tests.runtime.test_whitebox_containers import ProbeBackend
from robocode.utils import docker_sandbox, apptainer_sandbox
from robocode.utils.model_broker import BrokerUpstream
import robocode.approaches.agentic_base as approach
import robocode.utils.episode as episode
class StrictProbe(ProbeBackend):
    def build_cli_cmd(self, config, **kwargs):
        assert config.blackbox_strict and config.whitebox_source is None
        assert 'VARIABLE number of objects' in config.prompt
        (config.sandbox_dir / 'probe.py').write_text(Path({str(tmp_path / "probe.py")!r}).read_text())
        super().build_cli_cmd(config, **kwargs)
        return ['/opt/robocode-strict/bin/python', '-B', '/sandbox/probe.py']
approach.create_backend = lambda cfg: StrictProbe(cfg['backend'])
docker_sandbox._build_docker_auth_args = lambda _: nullcontext(([], {{}}))
apptainer_sandbox.load_broker_upstream = lambda _: BrokerUpstream('responses', 'api.openai.com', '/v1', {{}})
apptainer_sandbox._model_client = lambda _, command: (command, {{}})
def forbidden(*args, **kwargs):
    raise AssertionError('host loaded the submitted policy')
approach.load_generated_approach = episode.load_generated_approach = forbidden
runpy.run_path('experiments/run_experiment.py', run_name='__main__')
assert not any(n == 'kinder' or n.startswith('kinder.') for n in sys.modules)
print('STRICT_RUNNER_HOST_LOADED_NEITHER_POLICY_NOR_ENVIRONMENT')
"""
    output = tmp_path / "runner"
    command = [
        sys.executable,
        "-B",
        "-c",
        driver,
        "approach=agentic",
        "approach/backend=codex_gpt6",
        "environment=obstruction2d_generalized",
        "+environment.base_steps=5",
        "+environment.steps_per_object=2",
        "primitive_level=none",
        "approach.blackbox=true",
        "approach.blackbox_strict=true",
        "mcp_tools=[]",
        "approach.container_backend=" + checkpoint["backend"],
        "environment_runtime=artifact",
        "environment_runtime.artifact=" + str(artifact.path),
        "environment_runtime.expected_id=" + artifact.artifact_id,
        "environment_runtime.image=" + checkpoint["image"],
        "environment_runtime.evaluation_image="
        + (
            "robocode-evaluation:multi-agent"
            if checkpoint["backend"] == "docker"
            else str(repo / "robocode-evaluation.sif")
        ),
        "eval_seed=314159",
        "num_eval_tasks=6",
        "max_steps=3",
        "determinism_check_num_episodes=2",
        "hydra.run.dir=" + str(output),
    ]
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2))
    with (tmp_path / "runner.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            env=dict(os.environ, PYTHONPATH="src:."),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    log_text = (tmp_path / "runner.log").read_text()
    assert result.returncode == 0, log_text
    assert "STRICT_RUNNER_HOST_LOADED_NEITHER_POLICY_NOR_ENVIRONMENT" in log_text
    assert "STRICT_OBJECT_CENTRIC_PROBE_OK" in (output / "stream.jsonl").read_text()
    results = json.loads((output / "results.json").read_text())
    episodes = results["per_episode"]
    assert results["num_crashed_episodes"] == 0, episodes
    assert [e["object_count"] for e in episodes] == _SCHEDULE
    assert [e["num_steps"] for e in episodes] == [5 + 2 * c for c in _SCHEDULE]
    assert set(results["by_count"]) == {"0", "1", "2", "3", "4"}
    assert results["strict_blackbox_image_id"].startswith("sha256:")
    assert results["count_regimes"]["design"]["counts"] == [0, 1, 2]
    assert results["count_regimes"]["held_out"]["counts"] == [3, 4]
    assert results["design_count_solve_rate"] == 0.0
    assert results["held_out_count_solve_rate"] == 0.0
    assert results["determinism_check_agreement_rate"] == 1.0
    assert results["environment_artifact_id"] == artifact.artifact_id
