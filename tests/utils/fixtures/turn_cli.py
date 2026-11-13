"""Deterministic provider protocol fixture, never contacts a model."""

import json
import re
import subprocess
import sys
from pathlib import Path

provider = sys.argv[1]
prompt = sys.stdin.read()
turn_id = re.search(r"Coordinator turn: ([\w-]+)", prompt).group(1)
resumed = "resume" in sys.argv or "--resume" in sys.argv
if resumed:
    assert "fixture" in sys.argv
    assert "--last" not in sys.argv and "--continue" not in sys.argv
store = Path("/home/node") / (
    ".codex/sessions" if provider == "codex" else ".claude/projects"
)
state_file = store / "fixture-context.json"
context = (
    json.loads(state_file.read_text(encoding="utf-8"))
    if state_file.exists()
    else {"turns": []}
)
assert resumed == bool(context["turns"])
if resumed:
    assert Path("approach.py").read_text(encoding="utf-8") == "remembered = 17\n"
    assert Path("helper.json").read_text(encoding="utf-8") == '{"nonce": 4319}'
    assert context["nonce"] == 4319
    assert not Path(".agent_submission.json").exists()
else:
    context["nonce"] = 4319
context["turns"].append(turn_id)
state_file.write_text(json.dumps(context), encoding="utf-8")
Path("approach.py").write_text(
    "remembered = 17\n" if not resumed else "remembered = 18\n", encoding="utf-8"
)
Path("helper.json").write_text('{"nonce": 4319}', encoding="utf-8")
with Path("trace.jsonl").open("a", encoding="utf-8") as trace:
    trace.write(
        json.dumps(
            {
                "turn_id": turn_id,
                "resumed": resumed,
                "context": context,
                "argv": sys.argv,
                "prompt": prompt,
            }
        )
        + "\n"
    )
subprocess.run(["git", "add", "-A"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(["git", "commit", "-qm", turn_id], check=True, stdout=subprocess.DEVNULL)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
Path(".agent_submission.json").write_text(
    json.dumps({"turn_id": turn_id, "commit": commit, "retained": False}),
    encoding="utf-8",
)
if provider == "codex":
    event = {
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
    with (store / "root.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps({"type": "thread.started", "thread_id": "fixture"}), flush=True)
    print(json.dumps({"type": "turn.completed"}), flush=True)
else:
    with (store / "session.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"turn_id": turn_id}) + "\n")
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
