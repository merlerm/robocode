"""Probe an exported simulator's interface inside its pinned environment image."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from robocode.runtime.artifacts import EnvironmentArtifact, load_environment_artifact
from robocode.utils.env_server import serialize_space


def state_schema(
    state: Any, *, required_objects: set[str] | None = None
) -> dict[str, Any]:
    """Describe actual observations without retaining values or reward semantics."""
    if hasattr(state, "type_features"):
        if required_objects is not None:
            missing = required_objects - set(state.get_object_names())
            if missing:
                raise ValueError(f"Missing required scene objects: {sorted(missing)}")
        for obj in state:
            if not np.isfinite(state.vec([obj])).all():
                raise ValueError("Nonfinite object features")
        return {
            "types": sorted(
                (typ.name, list(features))
                for typ, features in state.type_features.items()
            ),
            "objects": sorted(
                (obj.name, obj.type.name, len(state.vec([obj])))
                for obj in state
                if required_objects is None or obj.name in required_objects
            ),
        }
    array = np.asarray(state)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite observation")
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def probe(artifact: EnvironmentArtifact) -> dict[str, Any]:
    """Check reset and step at every declared object count."""
    env = instantiate(OmegaConf.create(artifact.manifest["config"]))
    try:
        scene = artifact.manifest["config"].get("include_static_objects", False)

        def describe(state: Any, count: int | None) -> dict[str, Any]:
            if not scene:
                return state_schema(state)
            prefix = artifact.manifest["config"]["count_object_prefix"]
            if sum(obj.name.startswith(prefix) for obj in state) != count:
                raise ValueError("Scene observation has the wrong object count")
            return state_schema(
                state, required_objects={"robot", "target_block", "target_surface"}
            )

        contract = {
            "action_space": serialize_space(env.action_space),
            "observation_space": serialize_space(env.observation_space),
        }
        counts: list[Any] = [None]
        if hasattr(env, "eval_counts"):
            counts = sorted(set(env.design_counts + env.eval_counts))
            contract["counts"] = {
                "design": env.design_counts,
                "eval": env.eval_counts,
                "max_steps": {str(c): env.max_steps_for_count(c) for c in counts},
            }
        observations = {}
        for count in counts:
            state, info = env.reset(
                seed=0, options=None if count is None else {"object_count": count}
            )
            if count is not None and info["object_count"] != count:
                raise ValueError("Reset returned the wrong object count")
            reset = describe(state, count)
            action = np.clip(
                np.zeros(env.action_space.shape),
                env.action_space.low,
                env.action_space.high,
            ).astype(env.action_space.dtype)
            state, _, _, _, _ = env.step(action)
            observations[str(count)] = {"reset": reset, "step": describe(state, count)}
            artifact.verify_imports()
        contract["observations"] = observations
        return contract
    finally:
        env.close()


def main() -> None:
    """Separate candidate exceptions from failures to start the trusted worker."""
    artifact = load_environment_artifact(Path("/artifact"))
    Path("/output/started").touch()
    try:
        result = {"valid": True, "contract": probe(artifact)}
    except Exception:  # pylint: disable=broad-exception-caught
        result = {"valid": False, "error": traceback.format_exc()}
    result["environment_artifact_id"] = artifact.artifact_id
    Path("/output/report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
