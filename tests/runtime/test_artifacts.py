"""Environment revisions must reproduce code, inputs and import origins."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robocode.runtime.artifacts import export_environment, load_environment_artifact

_IMAGE = "sha256:" + "1" * 64


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture(name="repo")
def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Artifact test")
    _git(repo, "config", "user.email", "artifact-test@example.invalid")
    package = repo / "src" / "sample_env"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (repo / ".gitignore").write_text("ignored.py\nassets/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "E0")
    return repo


def _export(repo: Path, destination: Path, **kwargs):
    options = {
        "package": "sample_env",
        "config": {"limit": 10, "name": "test"},
        "runtime_image": _IMAGE,
        "dependencies": {},
    }
    options.update(kwargs)
    return export_environment(repo, "HEAD", destination, **options)


def _run(artifact, code: str, extra_path: Path | None = None):
    root = Path(__file__).resolve().parents[2] / "src"
    paths = [str(extra_path or artifact.import_root), str(root)]
    return subprocess.run(
        [sys.executable, "-B", "-c", code, str(artifact.path)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(paths)),
        cwd=artifact.path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_export_uses_commit_not_working_tree(repo, tmp_path):
    """Dirty, staged, ignored and untracked editor files cannot change E0."""
    first = _export(repo, tmp_path / "first")
    (repo / "src/sample_env/__init__.py").write_text("VALUE = 999\n")
    (repo / "src/sample_env/untracked.py").write_text("VALUE = 999\n")
    (repo / "src/sample_env/ignored.py").write_text("VALUE = 999\n")
    _git(repo, "add", "src/sample_env/__init__.py")
    second = _export(repo, tmp_path / "second")
    assert first.artifact_id == second.artifact_id
    assert (second.import_root / "sample_env/__init__.py").read_text() == "VALUE = 1\n"
    assert not (second.import_root / "sample_env/untracked.py").exists()
    assert not (second.import_root / "sample_env/ignored.py").exists()
    assert not (second.path / "source/.git").exists()
    assert load_environment_artifact(second.path, expected_id=first.artifact_id)
    with pytest.raises(FileExistsError):
        _export(repo, first.path)


def test_revision_round_trip_runs_frozen_code(repo, tmp_path):
    """E0 -> E1 -> E0 loads the requested source in fresh interpreters."""
    first = _export(repo, tmp_path / "first")
    (repo / "src/sample_env/__init__.py").write_text("VALUE = 2\n")
    _git(repo, "commit", "-qam", "E1")
    second = _export(repo, tmp_path / "second")
    assert first.artifact_id != second.artifact_id
    code = """import sys
import sample_env
from robocode.runtime.artifacts import load_environment_artifact
from pathlib import Path
artifact = load_environment_artifact(Path(sys.argv[1]))
assert sample_env.__file__ in artifact.verify_imports()['sample_env']
print(sample_env.VALUE)
"""
    for artifact, value in [(first, "1"), (second, "2"), (first, "1")]:
        result = _run(artifact, code)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == value
    with pytest.raises(ValueError, match="identity mismatch"):
        load_environment_artifact(second.path, expected_id=first.artifact_id)


def test_config_order_and_source_location_do_not_change_identity(repo, tmp_path):
    first = _export(repo, tmp_path / "first")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    second = _export(clone, tmp_path / "second", config={"name": "test", "limit": 10})
    assert first.artifact_id == second.artifact_id


@pytest.mark.parametrize("changed", ["config", "asset", "dependency", "image"])
def test_identity_covers_every_declared_input(repo, tmp_path, changed):
    asset = tmp_path / "mesh.dat"
    asset.write_bytes(b"original mesh")
    lock = tmp_path / "uv.lock"
    lock.write_text("dependency-version = 1\n")
    options = {"assets": {"assets/mesh.dat": asset}, "dependencies": {"uv.lock": lock}}
    first = _export(repo, tmp_path / "first", **options)
    if changed == "config":
        options["config"] = {"limit": 11, "name": "test"}
    elif changed == "asset":
        asset.write_bytes(b"changed mesh")
    elif changed == "dependency":
        lock.write_text("dependency-version = 2\n")
    else:
        options["runtime_image"] = "sha256:" + "2" * 64
    second = _export(repo, tmp_path / "second", **options)
    assert first.artifact_id != second.artifact_id
    assert (first.path / "source/assets/mesh.dat").read_bytes() == b"original mesh"
    assert (
        first.path / "dependencies/uv.lock"
    ).read_text() == "dependency-version = 1\n"
    load_environment_artifact(first.path)


@pytest.mark.parametrize(
    "tamper", ["modify", "add", "delete", "executable", "manifest"]
)
def test_tampering_is_rejected(repo, tmp_path, tamper):
    artifact = _export(repo, tmp_path / "artifact")
    code = artifact.import_root / "sample_env/__init__.py"
    if tamper == "modify":
        code.write_text("VALUE = 999\n")
    elif tamper == "add":
        (code.parent / "shadow.py").write_text("VALUE = 999\n")
    elif tamper == "delete":
        code.unlink()
    elif tamper == "executable":
        code.chmod(0o755)
    else:
        manifest = artifact.path / "manifest.json"
        data = json.loads(manifest.read_text())
        data["config"]["limit"] = 999
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_environment_artifact(artifact.path)


def test_wrong_and_mixed_import_origins_are_rejected(repo, tmp_path):
    artifact = _export(repo, tmp_path / "artifact")
    code = """import sys
from pathlib import Path
import sample_env
from robocode.runtime.artifacts import load_environment_artifact
load_environment_artifact(Path(sys.argv[1])).verify_imports()
"""
    result = _run(artifact, code, extra_path=repo / "src")
    assert result.returncode != 0 and "loaded from" in result.stderr
    # A correct top-level import must not conceal a cached old submodule.
    mixed = code.replace(
        "load_environment_artifact(Path(sys.argv[1])).verify_imports()",
        "import types\nold = types.ModuleType('sample_env.cached')\n"
        f"old.__file__ = {str(repo / 'src/sample_env/__init__.py')!r}\n"
        "sys.modules['sample_env.cached'] = old\n"
        "load_environment_artifact(Path(sys.argv[1])).verify_imports()",
    )
    result = _run(artifact, mixed)
    assert result.returncode != 0 and "sample_env.cached loaded from" in result.stderr


def test_assets_cannot_replace_source_or_escape(repo, tmp_path):
    asset = tmp_path / "asset"
    asset.write_bytes(b"not code")
    with pytest.raises(FileExistsError, match="replace committed source"):
        _export(
            repo, tmp_path / "overwrite", assets={"src/sample_env/__init__.py": asset}
        )
    with pytest.raises(ValueError, match="relative path"):
        _export(repo, tmp_path / "escape", assets={"../outside": asset})
    assert not (tmp_path / "escape/outside").exists()


def test_symlinks_and_submodule_parents_fail_loudly(repo, tmp_path):
    (repo / "link").symlink_to("src/sample_env/__init__.py")
    _git(repo, "add", "link")
    _git(repo, "commit", "-qm", "link")
    with pytest.raises(ValueError, match="Unsupported Git archive entry"):
        _export(repo, tmp_path / "symlink")
    _git(repo, "rm", "link")
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        _git(repo, "rev-parse", "HEAD"),
        "nested",
    )
    _git(repo, "commit", "-qm", "gitlink")
    with pytest.raises(ValueError, match="submodule parent"):
        _export(repo, tmp_path / "gitlink")


def test_runtime_image_must_be_content_identified(repo, tmp_path):
    with pytest.raises(ValueError, match="resolved sha256"):
        _export(repo, tmp_path / "tag", runtime_image="image:latest")
    assert not (tmp_path / "tag").exists()


def test_manifest_symlink_is_rejected(repo, tmp_path):
    artifact = _export(repo, tmp_path / "artifact")
    manifest = artifact.path / "manifest.json"
    outside = tmp_path / "external.json"
    manifest.rename(outside)
    manifest.symlink_to(outside)
    with pytest.raises(ValueError, match="manifest must not be a symlink"):
        load_environment_artifact(artifact.path)


def test_special_asset_is_rejected(repo, tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="Expected an asset file or directory"):
        _export(repo, tmp_path / "artifact", assets={"assets/pipe": fifo})


def test_namespace_packages_require_all_paths_inside_export(repo, tmp_path):
    namespace = repo / "src/sample_env/nested"
    namespace.mkdir()
    (namespace / "module.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "namespace")
    artifact = _export(repo, tmp_path / "artifact")
    code = """import sys
from pathlib import Path
import sample_env.nested.module
from robocode.runtime.artifacts import load_environment_artifact
artifact = load_environment_artifact(Path(sys.argv[1]))
origins = artifact.verify_imports()
assert origins['sample_env.nested'] == [str(artifact.import_root / 'sample_env/nested')]
"""
    result = _run(artifact, code)
    assert result.returncode == 0, result.stderr
    result = _run(
        artifact,
        code
        + "sample_env.nested.__path__.append('/outside')\nartifact.verify_imports()",
    )
    assert result.returncode != 0 and "sample_env.nested loaded from" in result.stderr
