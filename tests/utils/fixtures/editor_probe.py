"""Run edited dynamics against the frozen policy API inside the editor container."""

# pylint: disable=unspecified-encoding
import json
import sys
from pathlib import Path

import kinder
import numpy as np
from PIL import Image

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
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
env = KinderGeom2DEnv("kinder/Motion2D-p0-v0")
state, info = env.reset(seed=42)
with PolicyClient(metadata) as policy:
    policy.reset(state, info)
    action = policy.act(state)
    np.testing.assert_allclose(action, [0.04, 0, 0, 0, 0])
    after, reward, terminated, truncated, info = env.step(action)
    policy.update(after, reward, terminated or truncated, info)
    np.testing.assert_allclose(policy.act(after), [0.03, 0, 0, 0, 0])
    delta = float(np.max(np.abs(after - state)))
    np.testing.assert_allclose(delta, 0.04 * scale, atol=1e-6)
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
            "runtime_readonly": True,
        }
    )
)
