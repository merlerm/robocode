"""Recreate coordinator, provider adapter and container between two turns."""

# pylint: disable=missing-function-docstring,protected-access
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robocode.runtime.agent_session import AgentSession
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.backends import create_backend
from robocode.utils.model_broker import BrokerUpstream


class FixtureBackend:
    """Use production provider setup and parsing, substituting only the CLI."""

    def __init__(self, name):
        self.name = name
        self.delegate = create_backend(
            OmegaConf.create(
                {
                    "backend": name,
                    "output_usd_per_mtok": 20,
                }
            )
        )

    def build_cli_cmd(self, config, **kwargs):
        command = self.delegate.build_cli_cmd(config, **kwargs)
        return [
            "/opt/robocode-strict/bin/python",
            "/sandbox/turn_cli.py",
            self.name,
            *command,
        ]

    def setup_sandbox_files(self, *args, **kwargs):
        return self.delegate.setup_sandbox_files(*args, **kwargs)

    def stdin_text(self, config):
        return self.delegate.stdin_text(config)

    def build_env(self, config, extra=None):
        return self.delegate.build_env(config, extra)

    def parse_stream(self, *args, **kwargs):
        return self.delegate.parse_stream(*args, **kwargs)


@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_two_turns_in_fresh_containers(tmp_path, monkeypatch, runtime, provider):
    if not os.environ.get("ROBOCODE_TEST_AGENT_TURNS"):
        pytest.skip("Set ROBOCODE_TEST_AGENT_TURNS=1 for real container turn checks")
    root = Path(__file__).resolve().parents[2]
    session = AgentSession(tmp_path / "developer")
    fixture = root / "tests/utils/fixtures/turn_cli.py"
    files = {"turn_cli.py": fixture}
    if runtime == "docker":
        monkeypatch.setattr(
            docker_sandbox, "_build_docker_auth_args", lambda _: nullcontext(([], {}))
        )
        spaces = tmp_path / "env_spaces.json"
        spaces.write_text('{"port": 9}')
        files["env_spaces.json"] = spaces
        config_type = docker_sandbox.DockerSandboxConfig
        options = {}
    else:
        monkeypatch.setattr(
            apptainer_sandbox,
            "load_broker_upstream",
            lambda _: BrokerUpstream("responses", "api.openai.com", "/v1", {}),
        )
        monkeypatch.setattr(
            apptainer_sandbox, "_model_client", lambda _, cmd: (cmd, {})
        )
        config_type = apptainer_sandbox.ApptainerSandboxConfig
        options = {"strict_sif_path": root / "robocode-strict-blackbox.sif"}
    config = config_type(
        sandbox_dir=session.workspace,
        init_files=files,
        output_filename="approach.py",
        model="fixture-model",
        max_budget_usd=2,
        blackbox=True,
        blackbox_strict=True,
        **options,
    )
    first = session.run_turn(
        "p0", config, FixtureBackend(provider), environment_id="e0"
    )
    second = AgentSession(session.directory).run_turn(
        "p1",
        replace(config, max_budget_usd=3),
        FixtureBackend(provider),
        environment_id="e1",
    )
    assert first["outcome"] == second["outcome"] == "submitted"
    assert first["cost_usd"] == second["cost_usd"] == pytest.approx(0.25)
    assert second["cumulative_cost_usd"] == pytest.approx(0.5)
    assert first["commit"] != second["commit"]
    assert second["stream_start_bytes"] == first["stream_end_bytes"]
    assert (session.directory / "turns/p1/stream.jsonl").is_file()
    assert (
        json.loads((session.directory / "session.json").read_text())["session_id"]
        == "fixture"
    )
    assert "AGENTS.md" not in second["artifact"]["files"]
    assert "env_spaces.json" not in second["artifact"]["files"]
    assert first["artifact"]["id"] != second["artifact"]["id"]
    assert (
        session.directory / "turns/p0/artifact/approach.py"
    ).read_text() == "remembered = 17\n"
    trace = [
        json.loads(line)
        for line in (session.workspace / "trace.jsonl").read_text().splitlines()
    ]
    assert [t["resumed"] for t in trace] == [False, True]
    if provider == "codex":
        status = json.loads(
            (session.workspace / ".agent_sessions/codex/budget_status.json").read_text()
        )
        assert status["attempt_budget_usd"] == 3
        assert status["turn_id"] == "p1"
        assert not (
            session.workspace / ".agent_sessions/codex/solution_confident"
        ).exists()
    else:
        assert trace[1]["argv"][trace[1]["argv"].index("--max-budget-usd") + 1] == "3"
