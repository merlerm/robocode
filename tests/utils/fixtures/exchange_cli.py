"""Scripted exchange actors, retaining production launchers and stream parsers."""

import json
import re
import subprocess
import sys
from pathlib import Path

provider = sys.argv[1]
prompt = sys.stdin.read()
turn_id = re.search(r"Coordinator turn: ([\w-]+)", prompt).group(1)
resumed = "resume" in sys.argv or "--resume" in sys.argv
store = Path("/home/node") / (
    ".codex/sessions" if provider == "codex" else ".claude/projects"
)
context = store / "exchange-context.json"
if resumed:
    assert "fixture" in sys.argv
    previous = json.loads(context.read_text())["turn"]
    assert previous == ("p0" if turn_id == "p1" else "e1-a0")
else:
    assert not context.exists()
context.write_text(json.dumps({"turn": turn_id}))
if turn_id.startswith("p"):
    import numpy as np
    from env_client import make_env

    env = make_env()
    state, info = env.reset(seed=1)
    env.step(np.zeros(env.action_space.shape))
    assert not Path("/sandbox/source").exists()
    env.close()
    Path("approach.py").write_text(
        "class GeneratedApproach:\n"
        "    def __init__(self, action_space, observation_space, primitives): pass\n"
        "    def reset(self, state, info): pass\n"
        f"    def get_action(self, state): return [{0 if turn_id == 'p0' else .01}, 0, 0, 0, 0]\n"
    )
else:
    source = Path("source/src/kinder/envs/kinematic2d/base_env.py")
    if turn_id.endswith("a0") and "--valid-editor" not in sys.argv:
        source.write_text(source.read_text() + "\nINVALID PYTHON EDIT\n")
    else:
        if "--valid-editor" not in sys.argv:
            assert "previous commit was rejected" in prompt
        text = source.read_text().replace("\nINVALID PYTHON EDIT\n", "")
        source.write_text(
            text.replace(
                "dx, dy, dtheta, darm, vac = action",
                "dx, dy, dtheta, darm, vac = action\n        dx *= 0.5",
                1,
            )
        )
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from robocode.utils.policy_client import PolicyClient

        env = instantiate(
            OmegaConf.create(json.loads(Path("environment-config.json").read_text()))
        )
        state, info = env.reset(seed=1, options={"object_count": 0})
        with PolicyClient(
            json.loads(Path("policy_service.json").read_text())
        ) as policy:
            policy.reset(state, info)
            action = policy.act(state)
            assert all(x == 0 for x in action)
            state, reward, terminated, truncated, info = env.step(action)
            policy.update(state, reward, terminated or truncated, info)
        env.close()
        assert Path("/frozen-policy/approach.py").exists() == (
            "Read-only policy code" in prompt
        )
subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(["git", "commit", "-qm", turn_id], check=True)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
Path(".agent_submission.json").write_text(
    json.dumps({"turn_id": turn_id, "commit": commit, "retained": False})
)
if provider == "codex":
    with (store / "root.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "token_usage_record",
                    "payload": {
                        "response_id": turn_id,
                        "usage": {
                            "input_tokens": 0,
                            "cached_input_tokens": 0,
                            "output_tokens": 12500,
                        },
                    },
                }
            )
            + "\n"
        )
    print(json.dumps({"type": "thread.started", "thread_id": "fixture"}), flush=True)
    print(json.dumps({"type": "turn.completed"}), flush=True)
else:
    with (store / "session.jsonl").open("a") as stream:
        stream.write(json.dumps({"turn_id": turn_id}) + "\n")
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "total_cost_usd": 0.25
                * len((store / "session.jsonl").read_text().splitlines()),
                "session_id": "fixture",
            }
        ),
        flush=True,
    )
