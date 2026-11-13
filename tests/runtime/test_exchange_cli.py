"""Exercise the actual CLI's live configuration with scripted agent inference."""

import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from experiments import run_exchange
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.model_broker import BrokerUpstream
from tests.runtime.test_exchange_containers import ExchangeBackend


class CLIBackend(ExchangeBackend):
    """Install the fixture without changing the CLI's sandbox configuration."""

    def setup_sandbox_files(self, *args, **kwargs):
        config = args[0]
        super().setup_sandbox_files(*args, **kwargs)
        shutil.copy2(
            Path(__file__).parents[1] / "utils/fixtures/exchange_cli.py",
            config.sandbox_dir / "exchange_cli.py",
        )

    def build_cli_cmd(self, config, **kwargs):
        return [*super().build_cli_cmd(config, **kwargs), "--valid-editor"]


def test_live_cli_configuration(tmp_path, monkeypatch):
    """Render all five counts with telemetry and $2-turn config; spend nothing."""
    assets = os.environ.get("ROBOCODE_EXCHANGE_TEST_ARTIFACTS")
    if not assets:
        pytest.skip("Set ROBOCODE_EXCHANGE_TEST_ARTIFACTS for the CLI container check")
    monkeypatch.setattr(
        docker_sandbox, "_build_docker_auth_args", lambda _: nullcontext(([], {}))
    )
    monkeypatch.setattr(run_exchange, "create_backend", lambda _: CLIBackend("claude"))
    output = tmp_path / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_exchange.py",
            str(Path(assets) / "docker-artifact"),
            str(output),
            "--counts",
            "0",
            "1",
            "2",
            "3",
            "4",
        ],
    )
    run_exchange.main()
    ledger = json.loads((output / "run/exchange.json").read_text())
    assert ledger["status"] == "completed"
    assert len(ledger["turns"]) == 3
    assert all(turn["allowance_usd"] == 2 for turn in ledger["turns"])
    for evaluation in ledger["evaluations"]:
        result = evaluation["result"]
        assert result["num_evaluated_episodes"] == 5
        assert result["num_crashed_episodes"] == 0
        assert {ep["object_count"] for ep in result["per_episode"]} == {0, 1, 2, 3, 4}
    assert list(output.rglob("*.gif"))
    metadata = json.loads((output / "run/policy/workspace/env_spaces.json").read_text())
    assert "imports" not in metadata and "description" not in metadata
    assert "approach.py" not in (
        output / "run/editor/workspace/CLAUDE.md"
    ).read_text().replace("test_approach.py", "")


@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_policy_service_telemetry(tmp_path, monkeypatch, runtime, enabled):
    """The strict agent's resets are recorded by its private environment service."""
    assets = os.environ.get("ROBOCODE_EXCHANGE_TEST_ARTIFACTS")
    if not assets:
        pytest.skip(
            "Set ROBOCODE_EXCHANGE_TEST_ARTIFACTS for telemetry container checks"
        )
    assets = Path(assets)
    root = Path(__file__).resolve().parents[2]
    if runtime == "docker":
        monkeypatch.setattr(
            docker_sandbox, "_build_docker_auth_args", lambda _: nullcontext(([], {}))
        )
        config_type = docker_sandbox.DockerSandboxConfig
        image = "robocode-sandbox:multi-agent-runtime"
        options = {"docker_image": image}
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
        image = str(root / "robocode-sandbox.sif")
        options = {
            "sif_path": Path(image),
            "strict_sif_path": root / "robocode-strict-blackbox.sif",
        }
    policy = config_type(
        sandbox_dir=tmp_path,
        model="fixture-model",
        blackbox=True,
        blackbox_strict=True,
        telemetry=enabled,
        **options,
    )
    editor = config_type(sandbox_dir=tmp_path, **options)
    config = ExchangeConfig(
        backend=runtime,
        environment_image=image,
        evaluation_image="unused",
        runtime_source=assets / "service-runtime",
        eval_seeds=[1],
        eval_counts=[0],
    )
    backend = CLIBackend("claude")
    exchange = Exchange(
        tmp_path / "telemetry",
        load_environment_artifact(assets / (runtime + "-artifact")),
        config,
        policy,
        editor,
        backend,
        backend,
    )
    assert exchange._policy_turn("p0") is not None  # pylint: disable=protected-access
    sink = exchange.policy.directory / "telemetry/events.jsonl"
    assert not (exchange.policy.workspace / "telemetry").exists()
    if enabled:
        events = [json.loads(line) for line in sink.read_text().splitlines()]
        assert any(event["kind"] == "reset" for event in events)
        assert all(event["run"] == "p0" for event in events)
    else:
        assert not sink.exists()
