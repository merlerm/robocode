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
