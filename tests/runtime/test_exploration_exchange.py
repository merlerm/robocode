"""Shared allowance and matched source-access starts for exploratory experiments."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.exploration_exchange import Allowance, make_exchange
from robocode.runtime.artifacts import export_environment
from robocode.runtime.snapshots import snapshot_runtime
from robocode.runtime.whitebox import create_whitebox_source


def test_allowance_reserves_headroom_and_records_actual_cost(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(dict(ceiling_usd=25, reserve_usd=5, turns=[])))
    budget = Allowance(path)
    budget.before_turn(20)
    budget.record(tmp_path, dict(turn_id="e1", cost_usd=3.5, allowance_usd=20))
    budget.before_turn(16.5)
    with pytest.raises(RuntimeError, match="allowance exhausted"):
        budget.before_turn(16.51)
    assert json.loads(path.read_text())["turns"][0]["cost_usd"] == 3.5


def test_seeded_start_keeps_code_source_and_fresh_session(tmp_path):
    source = tmp_path / "source"
    (source / "src/kinder").mkdir(parents=True)
    (source / "src/kinder/__init__.py").write_text("VALUE=1\n")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@localhost",
            "commit",
            "-qm",
            "Initial",
        ],
    ]:
        subprocess.run(["git", "-C", str(source), *args], check=True)
    artifact = export_environment(
        source,
        "HEAD",
        tmp_path / "artifact",
        package="kinder",
        config={"base_steps": 3},
        runtime_image="sha256:" + "1" * 64,
        dependencies={},
    )
    runtime = tmp_path / "runtime"
    snapshot_runtime(runtime)
    view = create_whitebox_source(artifact, runtime, tmp_path / "view")
    seed = tmp_path / "original-policy"
    seed.mkdir()
    (seed / "approach.py").write_text("VALUE=2\n")
    (seed / "helper.py").write_text("VALUE=3\n")
    provenance = seed / "source-provenance.json"
    provenance.write_text(
        json.dumps(
            dict(artifact_id=artifact.artifact_id, whitebox_source_id=view.source_id)
        )
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            dict(valid=True, environment_artifact_id=artifact.artifact_id, contract={})
        )
    )
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps(dict(ceiling_usd=250, reserve_usd=10, turns=[])))
    args = SimpleNamespace(
        output=tmp_path / "run",
        artifact=artifact.path,
        seed_policy=seed / "approach.py",
        policy_source=view.path,
        initial_validation=validation,
        budget_ledger=budget,
        cycles=2,
        editor_attempts=1,
        editor_budget=8,
        policy_budget=5,
        no_editor_pacing=True,
        editor_instruction_file=None,
        condition="test",
        editor_only=False,
    )
    exchange = make_exchange(args)
    assert exchange._policy_turn("p0") == exchange.seed_policy
    assert exchange.policy_sources[exchange.seed_policy].source_id == view.source_id
    assert exchange.policy_config.init_files["helper.py"].read_text() == "VALUE=3\n"
    assert not (exchange.policy.directory / "session.json").exists()
    assert exchange._validate(artifact, "e0")["valid"]
    assert exchange.editor_pacing is False
    validation.write_text(
        json.dumps(dict(valid=True, environment_artifact_id="wrong", contract={}))
    )
    with pytest.raises(AssertionError):
        exchange._validate(artifact, "e0")
    provenance.write_text(
        json.dumps(dict(artifact_id=artifact.artifact_id, whitebox_source_id="wrong"))
    )
    args.output = tmp_path / "wrong-source-run"
    with pytest.raises(AssertionError):
        make_exchange(args)
