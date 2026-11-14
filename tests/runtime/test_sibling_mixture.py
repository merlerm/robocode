"""Reference source patches must reconstruct the exact alternate files."""

import shutil
import subprocess
import sys
from types import SimpleNamespace

from experiments import sibling_mixture
from experiments.sibling_mixture import source_patch
from robocode.runtime.snapshots import file_manifest


def test_patch_reconstructs_no_newline_empty_and_binary_files(tmp_path):
    current, member = tmp_path / "current", tmp_path / "member"
    current.mkdir()
    (current / "code.py").write_text("VALUE = 1")
    (current / "deleted-empty").touch()
    (current / "asset.bin").write_bytes(b"\x00\xffbefore")
    shutil.copytree(current, member)
    (member / "code.py").write_text("VALUE = 2")
    (member / "deleted-empty").unlink()
    (member / "added-empty").touch()
    (member / "asset.bin").write_bytes(b"\x00\xffafter")
    patch = tmp_path / "member.patch"
    patch.write_text(source_patch(current, member))
    subprocess.run(["git", "apply", "-p1", str(patch)], cwd=current, check=True)
    assert file_manifest(current) == file_manifest(member)


def test_e0_control_imports_evaluation_siblings_only_after_policy_turn(
    monkeypatch, tmp_path
):
    policy, seed = tmp_path / "p1.py", tmp_path / "p0.py"
    events = []
    exchange = SimpleNamespace(
        archive=["e0"],
        seed_policy=seed,
        ledger={"experiment": {"source_diff_base_id": "e0-view"}, "candidates": []},
        policy_sources={policy: SimpleNamespace(source_id="e0-view")},
        _save=lambda: None,
    )

    def prepare(_exchange, probes, *, base_only):
        assert not probes and base_only
        return True

    def policy_turn(name):
        assert exchange.archive == ["e0"]
        assert not exchange.ledger["candidates"]
        events.append("policy")
        return policy

    def import_evaluation_siblings(_exchange, probes):
        assert "policy" in events and len(probes) == 1
        events.append("import-evaluation")
        exchange.archive.append("e1")
        exchange.ledger["candidates"].append({"status": "accepted"})
        return [{"probe": str(probes[0])}]

    def evaluate(_exchange, _policy, artifact, _name):
        events.append("evaluate-" + artifact)

    exchange._policy_turn = policy_turn
    monkeypatch.setattr(sibling_mixture, "make_exchange", lambda args: exchange)
    monkeypatch.setattr(sibling_mixture, "import_siblings", prepare)
    monkeypatch.setattr(sibling_mixture, "add_siblings", import_evaluation_siblings)
    monkeypatch.setattr(sibling_mixture.Exchange, "_evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "artifact",
            "output",
            "--seed-policy",
            "seed",
            "--policy-source",
            "view",
            "--initial-validation",
            "validation",
            "--budget-ledger",
            "budget",
            "--base-only",
            "--evaluation-probe",
            "probe",
        ],
    )
    sibling_mixture.main()
    assert events == [
        "evaluate-e0",
        "policy",
        "evaluate-e0",
        "import-evaluation",
        "evaluate-e1",
        "evaluate-e1",
    ]
    assert exchange.ledger["status"] == "completed_sibling_adaptation"
    assert exchange.ledger["candidates"][0]["evaluation_only"]
