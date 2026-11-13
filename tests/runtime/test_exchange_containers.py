"""Real P0 -> rejected E -> repaired E1 -> resumed P1, without model inference."""

import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.utils import apptainer_sandbox, docker_sandbox
from robocode.utils.model_broker import BrokerUpstream
from tests.runtime.test_agent_turn_containers import FixtureBackend


class ExchangeBackend(FixtureBackend):
    """Replace model inference only, with role-specific real work in each container."""

    def build_cli_cmd(self, config, **kwargs):
        command = self.delegate.build_cli_cmd(config, **kwargs)
        python = (
            "/opt/robocode-strict/bin/python"
            if config.blackbox
            else "/robocode/.venv/bin/python"
        )
        return [python, "-B", "/sandbox/exchange_cli.py", self.name, *command]


@pytest.mark.parametrize("cycles", [1, 3])
@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("access", ["blackbox", "whitebox"])
def test_exchange(tmp_path, monkeypatch, runtime, provider, access, cycles):
    """Exercise rejection, repair, revision attribution and native continuation."""
    assets = os.environ.get("ROBOCODE_EXCHANGE_TEST_ARTIFACTS")
    if not assets:
        pytest.skip(
            "Set ROBOCODE_EXCHANGE_TEST_ARTIFACTS to prepared variable-count assets"
        )
    assets = Path(assets)
    root = Path(__file__).resolve().parents[2]
    artifact = load_environment_artifact(assets / (runtime + "-artifact"))
    if runtime == "docker":
        monkeypatch.setattr(
            docker_sandbox, "_build_docker_auth_args", lambda _: nullcontext(([], {}))
        )
        config_type = docker_sandbox.DockerSandboxConfig
        environment_image = "robocode-sandbox:multi-agent-runtime"
        evaluation_image = "robocode-evaluation:multi-agent"
        options = {"docker_image": environment_image}
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
        environment_image = str(root / "robocode-sandbox.sif")
        evaluation_image = str(root / "robocode-evaluation.sif")
        options = {
            "sif_path": Path(environment_image),
            "strict_sif_path": root / "robocode-strict-blackbox.sif",
        }
    common = {
        "sandbox_dir": tmp_path,
        "model": "fixture-model",
        "max_budget_usd": 2,
        "init_files": {
            "exchange_cli.py": root / "tests/utils/fixtures/exchange_cli.py"
        },
    }
    config = ExchangeConfig(
        backend=runtime,
        environment_image=environment_image,
        evaluation_image=evaluation_image,
        runtime_source=assets / "service-runtime",
        eval_seeds=[1000],
        eval_counts=[0],
        editor_attempts=2,
        cycles=cycles,
        policy_access=access,
        render=False,
    )
    exchange = Exchange(
        tmp_path / "exchange",
        artifact,
        config,
        config_type(**common, blackbox=True, blackbox_strict=True, **options),
        config_type(**common, **options),
        ExchangeBackend(provider),
        ExchangeBackend(provider),
    )
    ledger = exchange.run()
    assert ledger["status"] == "completed", ledger
    assert len(ledger["candidates"]) == 2 * cycles
    bad, good = ledger["candidates"][0], ledger["candidates"][1]
    assert bad["status"] == "rejected"
    assert "SyntaxError" in bad["validation"]["error"]
    assert good["status"] == "accepted"
    assert (
        bad["parent_environment_id"]
        == good["parent_environment_id"]
        == artifact.artifact_id
    )
    assert (
        ledger["candidates"][-1]["environment_artifact_id"]
        == ledger["active_environment_id"]
    )
    assert bad["commit"] != good["commit"]
    assert ledger["turns"][-1]["resumed"]
    assert ledger["turns"][2]["resumed"]
    assert ledger["turns"][-1]["environment_id"] == ledger["active_environment_id"]
    evaluations = ledger["evaluations"]
    assert len(evaluations) == 1 + 3 * cycles
    results = [e["result"] for e in evaluations]
    assert results[0]["policy_artifact_id"] == results[1]["policy_artifact_id"]
    assert results[2]["policy_artifact_id"] == results[3]["policy_artifact_id"]
    assert results[0]["policy_artifact_id"] != results[2]["policy_artifact_id"]
    assert (
        results[0]["environment_artifact_id"]
        == results[3]["environment_artifact_id"]
        == artifact.artifact_id
    )
    assert (
        results[1]["environment_artifact_id"]
        == results[2]["environment_artifact_id"]
        == good["environment_artifact_id"]
    )
    assert set(good["validation"]["contract"]["observations"]) == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    for cycle in range(1, cycles + 1):
        rejected, accepted = ledger["candidates"][2 * (cycle - 1) : 2 * cycle]
        parent = (
            artifact.artifact_id
            if cycle == 1
            else ledger["candidates"][2 * cycle - 3]["environment_artifact_id"]
        )
        assert rejected["status"] == "rejected"
        assert accepted["status"] == "accepted"
        assert (
            rejected["parent_environment_id"]
            == accepted["parent_environment_id"]
            == parent
        )
        before, after, baseline = results[3 * cycle - 2 : 3 * cycle + 1]
        assert (
            before["environment_artifact_id"]
            == after["environment_artifact_id"]
            == accepted["environment_artifact_id"]
        )
        assert (
            before["policy_artifact_id"]
            == results[0 if cycle == 1 else 3 * cycle - 4]["policy_artifact_id"]
        )
        assert after["policy_artifact_id"] == baseline["policy_artifact_id"]
        assert baseline["environment_artifact_id"] == artifact.artifact_id
    source = exchange.active.path / "source"
    # Every file in the filtered view must survive the editor's Git/export.
    view_source = exchange.directory / "views/e1/source"
    names = {p.relative_to(view_source) for p in view_source.rglob("*") if p.is_file()}
    assert names == {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    for name in names - {Path("src/kinder/envs/kinematic2d/base_env.py")}:
        assert (source / name).read_bytes() == (view_source / name).read_bytes()
    assert (source / "src/kinder/envs/kinematic2d/base_env.py").read_text().count(
        "dx *= 0.5"
    ) == cycles
    assert not (source / "exchange_cli.py").exists()
    load_environment_artifact(artifact.path)
    assert json.loads((exchange.directory / "exchange.json").read_text()) == json.loads(
        json.dumps(ledger)
    )
