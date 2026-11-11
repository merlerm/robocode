"""One paid Opus 5.5 turn; invoke twice to recreate the coordinator."""

import argparse
import json
import logging
import secrets
from pathlib import Path

from omegaconf import OmegaConf

from robocode.runtime.agent_session import AgentSession
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import DockerSandboxConfig


def main() -> None:
    """Check file and conversation continuity with a $1 turn allowance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("turn", type=int, choices=[0, 1])
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [live-turn={args.turn}] %(name)s %(message)s",
    )
    session = AgentSession(root / "agent")
    if args.turn == 0:
        with (root / "expected.json").open("x", encoding="utf-8") as handle:
            nonce = secrets.token_hex(12)
            json.dump({"nonce": nonce, "model": "claude-opus-5-5"}, handle)
        prompt = (
            "This is a short infrastructure test, not a robot task. "
            f"Remember the nonce {nonce} in conversation only. Do not write the "
            "nonce in files or commit messages. Create approach.py containing exactly "
            "value = 17 followed by a newline. Commit it and submit as instructed. "
            "Do not investigate the environment or install anything. Then stop."
        )
    else:
        prompt = (
            "Continue the short infrastructure test. Recall the nonce from our "
            "previous conversation and write it alone into remembered_nonce.txt. "
            "Do not read conversation/session files or logs to recover it. "
            "Read approach.py and verify it still contains value = 17, then change "
            "it to value = 18 followed by a newline. Commit both files, submit as "
            "instructed, and stop. "
            "Do not investigate the environment or install anything."
        )
    metadata = root / "env_spaces.json"
    metadata.write_text('{"port": 9}\n', encoding="utf-8")
    config = DockerSandboxConfig(
        sandbox_dir=session.workspace,
        init_files={"env_spaces.json": metadata},
        prompt=prompt,
        system_prompt="Complete this bounded infrastructure check and submit promptly.",
        output_filename="approach.py",
        model="claude-opus-5-5",
        effort="high",
        max_budget_usd=1.0,
        max_turns=8,
        blackbox=True,
        blackbox_strict=True,
    )
    result = session.run_turn(
        f"p{args.turn}",
        config,
        create_backend(OmegaConf.create({"backend": "claude"})),
        environment_id="continuation-probe-no-environment",
    )
    assert result["outcome"] == "submitted", result
    for attempt in result["attempts"]:
        assert set(attempt["metrics"]["model_usage"]) == {"claude-opus-5-5"}
    artifact = session.directory / "turns" / f"p{args.turn}" / "artifact"
    assert (artifact / "approach.py").read_text().strip() == f"value = {17 + args.turn}"
    nonce = json.loads((root / "expected.json").read_text())["nonce"]
    if args.turn == 0:
        assert all(
            nonce not in p.read_text() for p in artifact.rglob("*") if p.is_file()
        )
    else:
        assert (artifact / "remembered_nonce.txt").read_text().strip() == nonce
        first = json.loads((session.directory / "turns/p0/turn.json").read_text())
        assert (
            result["attempts"][-1]["session_id"] == first["attempts"][-1]["session_id"]
        )
        assert result["commit"] != first["commit"]
        assert (
            session.directory / "turns/p0/artifact/approach.py"
        ).read_text().strip() == "value = 17"
    (root / f"verified-turn-{args.turn}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "turn": args.turn,
                "outcome": result["outcome"],
                "cost_usd": result["cost_usd"],
                "cumulative_cost_usd": result["cumulative_cost_usd"],
            }
        )
    )


if __name__ == "__main__":
    main()
