"""Policy submissions must be stable and must not expose development credentials."""

import pytest

from robocode.runtime.snapshots import snapshot_policy, verify_snapshot


def test_snapshot_pins_helpers_and_data_without_transport(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name, text in {
        "approach.py": "import helper",
        "helper.py": "x=1",
        "weights.bin": "data",
        "env_spaces.json": "secret token",
    }.items():
        (source / name).write_text(text)
    frozen = tmp_path / "frozen"
    manifest = snapshot_policy(source / "approach.py", frozen)
    assert set(manifest["files"]) == {"approach.py", "helper.py", "weights.bin"}
    (source / "helper.py").write_text("x=2")
    verify_snapshot(frozen, manifest["id"])
    (frozen / "helper.py").write_text("x=3")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_snapshot(frozen, manifest["id"])


def test_snapshot_rejects_links_outside_submission(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "approach.py").write_text("pass")
    (source / "helper.py").symlink_to(tmp_path / "private.py")
    with pytest.raises(ValueError, match="symlinks"):
        snapshot_policy(source / "approach.py", tmp_path / "frozen")


def _sandbox(root, transcript):
    """Write identical policy code next to per-run agent and harness state."""
    for name, text in {
        "approach.py": "from helper import act",
        "helper.py": "def act(obs): return 0",
        "data/table.json": "[1, 2]",
        "agent_log.txt": transcript,
        ".agent_sessions/codex/rollout.jsonl": transcript,
        ".agent_sessions/codex/budget_status.json": transcript,
        ".agent_home/projects/session.jsonl": transcript,
        ".agent_prompts/system_prompt.md": transcript,
        ".agent_submission.json": transcript,
        "opencode.json": transcript,
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root / "approach.py"


def test_snapshot_excludes_agent_state(tmp_path):
    """Agent state stays out, so identical code gives identical identities."""
    left = snapshot_policy(_sandbox(tmp_path / "a", "plan A"), tmp_path / "a_frozen")
    right = snapshot_policy(_sandbox(tmp_path / "b", "plan B"), tmp_path / "b_frozen")
    expected = {"approach.py", "helper.py", "data/table.json"}
    assert set(left["files"]) == expected
    assert {p.name for p in (tmp_path / "a_frozen").iterdir()} == {
        "approach.py",
        "helper.py",
        "data",
    }
    assert left == right
