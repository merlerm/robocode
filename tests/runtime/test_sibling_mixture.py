"""Reference source patches must reconstruct the exact alternate files."""

import shutil
import subprocess

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
