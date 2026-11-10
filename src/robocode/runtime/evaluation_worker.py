"""Evaluate a frozen policy without environment source or host filesystem access."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from robocode.approaches.base_approach import BaseApproach
from robocode.runtime.binding import EnvironmentBinding
from robocode.runtime.evaluation import evaluate_policy, run_determinism_check
from robocode.runtime.remote_environment import RemoteEnvironment
from robocode.runtime.snapshots import verify_snapshot
from robocode.utils.episode import load_generated_approach
from robocode.utils.isolated_transport import verify_namespace


class FrozenPolicy(BaseApproach):
    """Apply the same generated-policy lifecycle as the synthesis approach."""

    def __init__(self, env: RemoteEnvironment, seed: int):
        super().__init__(env.action_space, env.observation_space, seed, {})
        self.generated = load_generated_approach(
            Path("/policy/approach.py"),
            self._action_space,
            self._state_space,
            {},
            strict_imports=True,
        )

    def reset(self, state: Any, info: dict[str, Any]) -> None:
        super().reset(state, info)
        self.generated.reset(state, info)

    def update(
        self, state: Any, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        super().update(state, reward, done, info)
        if hasattr(self.generated, "update"):
            self.generated.update(state, reward, done, info)

    def _get_action(self) -> Any:
        return self.generated.get_action(self._last_state)


def main() -> None:
    """Verify inputs, load the program here, and write ordinary evaluation results."""
    logging.basicConfig(level=logging.INFO)
    verify_namespace()
    request = json.loads(Path("/input/request.json").read_text(encoding="utf-8"))
    verify_snapshot(Path("/policy"), request["policy_id"])
    verify_snapshot(Path("/runtime"), request["runtime_source_id"])
    binding = EnvironmentBinding(**request["binding"])
    if binding.artifact_id != request["environment_artifact_id"]:
        raise ValueError("Evaluation environment revision mismatch")
    env = RemoteEnvironment(binding, request["max_steps"])
    try:
        policy = FrozenPolicy(env, request["replicate_seed"])
        Path("/output/ready").touch()
        options = {
            "max_steps": request["max_steps"],
            "timeout": request["timeout"],
        }
        results = evaluate_policy(
            env,
            policy,
            request["eval_seeds"],
            **options,
            render=request["render"],
            output_dir=Path("/output"),
        )
        if request["determinism_episodes"] > 0:
            results.update(
                run_determinism_check(
                    env,
                    policy,
                    request["eval_seeds"],
                    **options,
                    num_episodes=request["determinism_episodes"],
                )
            )
        identities = {
            "policy_artifact_id": request["policy_id"],
            "environment_artifact_id": binding.artifact_id,
        }
        results.update(identities)
        for episode in results["per_episode"]:
            episode.update(identities)
        if "determinism_check" in results:
            results["determinism_check"].update(identities)
            for replay in results["determinism_check"]["episodes"]:
                replay.update(identities)
        results["evaluation_runtime_source_id"] = request["runtime_source_id"]
        Path("/output/results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
