"""Run edited dynamics against the frozen policy API inside the editor container."""

# pylint: disable=unspecified-encoding
import json
import sys
from pathlib import Path

import kinder
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from robocode.runtime.snapshots import verify_snapshot
from robocode.utils.policy_client import PolicyClient

turn_id, scale = sys.argv[1], float(sys.argv[2])
assert kinder.__file__.startswith("/sandbox/source/")
assert not Path("source/.git").exists()
assert not Path("source/demos").exists()
assert not Path("/robocode/src/robocode/oracles").exists()
assert not Path("/robocode/src/robocode/approaches/agentic_approach.py").exists()
mode = Path("access-mode.txt").read_text()
for name in ["/robocode/src/robocode/environments/kinder_geom2d_env.py"] + (
    ["/frozen-policy/approach.py"] if mode == "whitebox" else []
):
    try:
        with open(name, "a") as stream:
            stream.write("# forbidden")
    except OSError:
        pass
    else:
        raise AssertionError("Writable frozen input: " + name)
assert Path("/frozen-policy/approach.py").exists() == (mode == "whitebox")
assert not Path("/policy").exists()
assert not Path("/sandbox/../service").exists()
metadata = json.loads(Path("policy_service.json").read_text())
if mode == "whitebox":
    verify_snapshot(Path("/frozen-policy"), metadata["policy_artifact_id"])
env = instantiate(OmegaConf.create(json.loads(Path("probe-env.json").read_text())))
counts = env.eval_counts if hasattr(env, "eval_counts") else [None]
horizontal_action = 0.04 if turn_id == "e1" else 0.02
results = []
with PolicyClient(metadata) as policy:
    for count in counts:
        state, info = env.reset(
            seed=42, options={"object_count": count} if count is not None else None
        )
        policy.reset(state, info)
        action = policy.act(state)
        np.testing.assert_allclose(action, [horizontal_action, 0, 0, 0, 0])
        after, reward, terminated, truncated, info = env.step(action)
        policy.update(after, reward, terminated or truncated, info)
        np.testing.assert_allclose(policy.act(after), [0.03, 0, 0, 0, 0])
        delta = (
            float(
                max(
                    np.max(np.abs(after.vec([obj]) - state.vec([obj]))) for obj in state
                )
            )
            if count is not None
            else float(np.max(np.abs(after - state)))
        )
        np.testing.assert_allclose(delta, horizontal_action * scale, atol=1e-6)
        results.append({"object_count": count, "transition_delta": delta})
        Image.fromarray(env.render()).save(f"{turn_id}-count-{count}.png")
    Image.fromarray(env.render()).save(f"{turn_id}.png")
env.close()
Path(f"{turn_id}-probe.json").write_text(
    json.dumps(
        {
            "turn_id": turn_id,
            "mode": mode,
            "policy_artifact_id": metadata["policy_artifact_id"],
            "kinder_origin": kinder.__file__,
            "transition_delta": delta,
            "action": action.tolist(),
            "counts": results,
            "runtime_readonly": True,
        }
    )
)
