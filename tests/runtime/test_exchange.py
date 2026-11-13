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
