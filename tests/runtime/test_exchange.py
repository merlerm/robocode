"""Coordinator stop conditions preserve explicit submission semantics."""

# pylint: disable=missing-function-docstring,unused-import,redefined-outer-name,protected-access
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from robocode.runtime import exchange as exchange_module
from robocode.runtime.candidates import CandidateSourceError
from robocode.runtime.exchange import Exchange, ExchangeConfig
from robocode.utils import rate_limit
from robocode.utils.docker_sandbox import DockerSandboxConfig
from tests.runtime.test_agent_session import setup, write_submission
from tests.runtime.test_whitebox import fixture_source


def make_exchange(source, tmp_path):
    _, artifact, runtime, _ = source
    config = ExchangeConfig(
        backend="docker",
        environment_image="image",
        evaluation_image="eval",
        runtime_source=runtime,
        eval_seeds=[1],
    )
    sandbox = DockerSandboxConfig(
        sandbox_dir=tmp_path, blackbox=True, blackbox_strict=True
    )
    backend = SimpleNamespace(name="claude")
    return Exchange(
        tmp_path / "exchange", artifact, config, sandbox, sandbox, backend, backend
    )


@pytest.mark.parametrize("outcome", ["unsubmitted", "best_effort", "rejected"])
def test_only_explicit_submission_advances(source, tmp_path, outcome):
    exchange = make_exchange(source, tmp_path)
    assert not exchange._record_turn({"outcome": outcome})
    assert exchange.ledger["status"] == "stopped_without_submission"
    assert exchange.ledger["active_environment_id"] == source[1].artifact_id


def test_failure_records_and_raises(source, tmp_path, monkeypatch):
    exchange = make_exchange(source, tmp_path)

    def fail(*_args):
        raise RuntimeError("image unavailable")

    monkeypatch.setattr(exchange, "_validate", fail)
    with pytest.raises(RuntimeError, match="image unavailable"):
        exchange.run()
    assert exchange.ledger["status"] == "failed"
    assert exchange.ledger["active_environment_id"] == source[1].artifact_id


def test_missing_source_is_rejected_and_repaired_in_same_session(
    source, tmp_path, monkeypatch
):
    exchange = make_exchange(source, tmp_path)
    exchange.config = replace(exchange.config, editor_attempts=2)
    binding = SimpleNamespace()
    monkeypatch.setattr(
        exchange_module, "policy_service", lambda *_a, **_kw: nullcontext(binding)
    )
    seen = []

    def turn(name, config, _backend, *, environment_id):
        seen.append((name, config.prompt, environment_id))
        return {"turn_id": name, "commit": name, "outcome": "submitted"}

    monkeypatch.setattr(exchange.editor, "run_turn", turn)

    def export(_workspace, commit, parent, _destination):
        if commit.endswith("a0"):
            raise CandidateSourceError("missing source/src")
        return replace(parent, artifact_id="repaired-candidate")

    monkeypatch.setattr(exchange_module, "export_candidate", export)
    monkeypatch.setattr(exchange, "_validate", lambda *_a: {"valid": True})
    assert exchange._editor_turns(tmp_path / "policy.py", 1, {})
    bad, good = exchange.ledger["candidates"][0], exchange.ledger["candidates"][1]
    assert bad["status"] == "rejected"
    assert good["status"] == "accepted"
    assert seen[0][2] == seen[1][2] == source[1].artifact_id
    assert "missing source/src" in seen[1][1]
    assert exchange_module.HELD_OUT_NOTICE in seen[0][1]
    assert exchange_module.HELD_OUT_NOTICE in seen[1][1]
    assert exchange.ledger["active_environment_id"] == "repaired-candidate"


def test_editor_prompt_contains_concrete_submission_contract(tmp_path, monkeypatch):
    session, config, backend = setup(tmp_path)
    config = replace(
        config, output_filename="", system_prompt="", prompt="Edit the simulator."
    )

    def run(active, _backend):
        assembled = backend.stdin_text(active)
        assert "Edit the simulator." in assembled
        assert "write .agent_submission.json containing" in assembled
        assert '"turn_id": "e1"' in assembled
        assert '"commit": "<full HEAD SHA>"' in assembled
        assert '"retained": false' in assembled
        (tmp_path / "assembled-editor-prompt.txt").write_text(assembled)
        return write_submission(active)

    monkeypatch.setattr(rate_limit, "_run_active_sandbox", run)
    record = session.run_turn("e1", config, backend, environment_id="e0")
    assert record["outcome"] == "submitted"


def test_whitebox_turns_pin_dependencies_and_transfer_without_rebinding(
    source, tmp_path, monkeypatch
):
    import json
    import subprocess

    from robocode.runtime.artifacts import export_environment

    repo, artifact, runtime, _ = source
    exchange = make_exchange(source, tmp_path)
    exchange.policy_config = replace(
        exchange.policy_config, blackbox=False, blackbox_strict=False
    )
    binding = SimpleNamespace(
        metadata={
            "action_space": {
                "type": "Box",
                "low": [-1],
                "high": [1],
                "shape": [1],
                "dtype": "float32",
            },
            "observation_space": {
                "type": "Box",
                "low": [-1],
                "high": [1],
                "shape": [1],
                "dtype": "float32",
            },
            "description": "Training task",
        },
        port=1234,
        token="fixture",
        client_host=lambda _: "127.0.0.1",
    )
    monkeypatch.setattr(
        exchange_module, "environment_service", lambda *_a, **_kw: nullcontext(binding)
    )
    seen = []

    def submit(name, config, _backend, *, environment_id):
        assert config.whitebox_source is not None
        assert not config.blackbox and not config.blackbox_strict
        assert "make_training_environment" in config.prompt
        assert "environment is a BLACK BOX" not in config.prompt
        assert exchange_module.HELD_OUT_NOTICE in config.prompt
        seen.append(config.whitebox_source)
        folder = exchange.policy.directory / "turns" / name / "artifact"
        folder.mkdir(parents=True)
        (folder / "approach.py").write_text("policy = True\n")
        return {"outcome": "submitted", "turn_id": name}

    monkeypatch.setattr(exchange.policy, "run_turn", submit)
    p0 = exchange._policy_turn("p0")
    (repo / "src/kinder/__init__.py").write_text("VALUE = 2\n")
    for args in [
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "E1",
        ],
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    edited = export_environment(
        repo,
        "HEAD",
        tmp_path / "e1",
        package="kinder",
        config={},
        runtime_image=artifact.manifest["runtime_image"],
        dependencies={},
    )
    exchange.active = edited
    p1 = exchange._policy_turn("p1")
    assert seen[0].source_id != seen[1].source_id
    assert seen[0].manifest["artifact_id"] == artifact.artifact_id
    assert seen[1].manifest["artifact_id"] == edited.artifact_id
    calls = []

    def evaluate(policy, environment, output, **kwargs):
        calls.append((policy, environment, kwargs))
        return {}

    monkeypatch.setattr(exchange_module, "evaluate_in_container", evaluate)
    exchange._evaluate(p0, edited, "p0-e1")
    exchange._evaluate(p1, artifact, "p1-e0")
    assert calls[0][2]["whitebox_source"] == seen[0]
    assert calls[1][2]["whitebox_source"] == seen[1]
    assert all(
        call[2]["evaluation_image"] == exchange.config.environment_image
        for call in calls
    )
    assert (
        json.loads((exchange.policy.directory / "turns/p0/turn.json").read_text())[
            "policy_source_environment_id"
        ]
        == artifact.artifact_id
    )
