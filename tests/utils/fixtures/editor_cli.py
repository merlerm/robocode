"""Scripted source editor; delegates provider stream fixtures after doing real work."""

# pylint: disable=unspecified-encoding
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
context_file = store / "editor-context.json"
if resumed:
    assert "fixture" in sys.argv
    assert json.loads(context_file.read_text())["previous_turn"] == "e1"
    assert not Path(".agent_submission.json").exists()
else:
    assert not context_file.exists()
    assert (
        subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"], text=True
        ).strip()
        == "1"
    )
context_file.write_text(json.dumps({"previous_turn": turn_id}))
source = Path("source/src/kinder/envs/kinematic2d/base_env.py")
old = (
    "dx, dy, dtheta, darm, vac = action"
    if not resumed
    else "dx, dy, dtheta, darm, vac = action\n        dx *= 0.5"
)
new = "dx, dy, dtheta, darm, vac = action\n        dx *= " + (
    "0.25" if resumed else "0.5"
)
assert old in source.read_text()
source.write_text(source.read_text().replace(old, new, 1))
# A fresh interpreter must pick up the new transition function.
subprocess.run(
    [sys.executable, "-B", "editor_probe.py", turn_id, "0.25" if resumed else "0.5"],
    check=True,
)
with Path("editor-trace.jsonl").open("a") as stream:
    stream.write(json.dumps({"turn_id": turn_id, "resumed": resumed}) + "\n")
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
                "total_cost_usd": 0.25,
                "session_id": "fixture",
            }
        ),
        flush=True,
    )
