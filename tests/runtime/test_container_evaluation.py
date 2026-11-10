"""Live source-isolation and failure checks for the evaluation container."""

import json
import os
from pathlib import Path

import pytest

from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.container_evaluation import evaluate_in_container


@pytest.fixture(params=["docker", "apptainer"])
def evaluator(request):
    root = os.environ.get("ROBOCODE_SERVICE_TEST_ARTIFACTS")
    if root is None:
        pytest.skip("Set ROBOCODE_SERVICE_TEST_ARTIFACTS for live container checks")
    backend = request.param
    repo = Path(__file__).resolve().parents[2]
    return {
        "artifact": load_environment_artifact(Path(root) / (backend + "-artifact")),
        "backend": backend,
        "environment_image": (
            "robocode-sandbox:multi-agent-runtime"
            if backend == "docker"
            else str(repo / "robocode-sandbox.sif")
        ),
        "evaluation_image": (
            "robocode-evaluation:multi-agent"
            if backend == "docker"
            else str(repo / "robocode-evaluation.sif")
        ),
        "eval_seeds": [42],
        "replicate_seed": 42,
        "max_steps": 1,
        "timeout": 0.05,
        "render": False,
        "determinism_episodes": 0,
    }


@pytest.mark.parametrize("outcome", ["ok", "timeout", "crash", "load_error"])
def test_evaluation_isolation_and_outcomes(evaluator, tmp_path, outcome):
    policy = tmp_path / "submission" / "approach.py"
    policy.parent.mkdir()
    host_secret = tmp_path / "host-only.txt"
    host_secret.write_text("not mounted")
    action = {
        "ok": "return [0, 0, 0, 0, 0]",
        "timeout": "while True: pass",
        "crash": "os._exit(23)",
        "load_error": "return [0, 0, 0, 0, 0]",
    }[outcome]
    policy.write_text(f"""import os
from pathlib import Path
assert not Path({str(host_secret)!r}).exists()
assert not Path('/artifact').exists()
assert not Path('/runtime/robocode/environments/kinder_geom2d_env.py').exists()
assert not Path('/policy/env_spaces.json').exists()
try:
    Path('/policy/approach.py').write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('policy source is writable')
print('SOURCE_ISOLATION_AND_READONLY_POLICY_VERIFIED', flush=True)
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        {"raise RuntimeError('policy load failed')" if outcome == "load_error" else "pass"}
    def reset(self, state, info): pass
    def get_action(self, state):
        {action}
""")
    output = tmp_path / "evaluation"
    if outcome == "load_error":
        with pytest.raises(RuntimeError, match="policy load failed"):
            evaluate_in_container(policy, output_dir=output, **evaluator)
    else:
        results = evaluate_in_container(policy, output_dir=output, **evaluator)
        episode = results["per_episode"][0]
        assert episode["policy_artifact_id"] == results["policy_artifact_id"]
        assert episode["environment_artifact_id"] == results["environment_artifact_id"]
        assert bool(episode.get("timed_out")) == (outcome == "timeout")
        assert bool(episode.get("crashed")) == (outcome == "crash")
        if outcome == "ok":
            assert episode["num_steps"] == 1
        assert results["environment_artifact_id"] == evaluator["artifact"].artifact_id
    lines = (output / "evaluation.log").read_text().splitlines()
    assert any(
        "SOURCE_ISOLATION_AND_READONLY_POLICY_VERIFIED" in line for line in lines
    )
    assert lines and all(line.startswith("[role=evaluation ") for line in lines)
    command = json.loads((output / "command.json").read_text())
    bridge = next(arg.split(":")[0] for arg in command if arg.endswith(":/bridge:ro"))
    assert not Path(bridge).exists()
