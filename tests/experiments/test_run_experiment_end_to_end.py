"""End-to-end check of the default (host) runner path with a saved policy."""

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_POLICY = '''"""Zero-action policy that records whether the development service is up."""

import os
import socket
from pathlib import Path

import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self._action_space = action_space

    def reset(self, state, info):
        run = Path(os.environ["ROBOCODE_TEST_RUN_DIR"])
        port = int((run / "services/development/env_server_port").read_text())
        with socket.socket() as probe:
            reachable = probe.connect_ex(("127.0.0.1", port)) == 0
        (run / "service_reachable_during_evaluation").write_text(str(reachable))

    def get_action(self, state):
        return np.zeros(self._action_space.shape, dtype=self._action_space.dtype)
'''


def test_default_runner_scores_a_saved_policy_after_stopping_its_service(
    tmp_path: Path,
) -> None:
    """Evaluation must not depend on the synthesis-time environment service."""
    sandbox = tmp_path / "policy" / "sandbox"
    sandbox.mkdir(parents=True)
    (sandbox / "approach.py").write_text(_POLICY)
    run = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/run_experiment.py",
            "approach=agentic",
            "primitive_level=none",
            "environment=motion2d_easy",
            f"approach.load_dir={tmp_path / 'policy'}",
            "approach.blackbox=true",
            "approach.blackbox_strict=true",
            "approach.container_backend=docker",
            "eval_seed=123",
            "num_eval_tasks=2",
            "max_steps=3",
            "eval_timeout=60",
            "determinism_check_num_episodes=1",
            f"hydra.run.dir={run}",
        ],
        cwd=_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{_ROOT / 'src'}{os.pathsep}{_ROOT}",
            "ROBOCODE_TEST_RUN_DIR": str(run),
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = json.loads((run / "results.json").read_text())
    assert len(results["per_episode"]) == 2
    assert results["num_crashed_episodes"] == 0
    assert results["determinism_check_agreement_rate"] == 1.0
    assert (run / "services" / "development" / "env_server.log").exists()
    assert (run / "service_reachable_during_evaluation").read_text() == "False"
