"""Reproduce E0 -> E1 -> E0 using frozen KinDER exports and a saved policy.

Run with the RoboCode interpreter and PYTHONPATH=src:., passing a fresh output path and
an existing approach.py. This is a manual probe, not the experiment runner.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from robocode.runtime.artifacts import export_environment, load_environment_artifact
from robocode.utils.apptainer_environment import clean_apptainer_env
from robocode.utils.apptainer_sandbox import _apptainer_exec_prefix

EDIT_FILE = "src/kinder/envs/kinematic2d/motion2d.py"
STEP_EDIT = """    def step(self, action):
        # A spatial drag zone changes dynamics without changing the task or spaces.
        assert self._current_state is not None
        robot = self._current_state.get_objects(CRVRobotType)[0]
        if self._current_state.get(robot, "x") >= 1.2:
            action = np.array(action, copy=True)
            action[:2] *= 0.25
        return super().step(action)

"""
DESCRIPTION = " Translation is reduced to one quarter while the robot is at x >= 1.2."


def git(repo: Path, *args: str) -> str:
    """Run Git in the specified checkout."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def digest(path: Path) -> str:
    """Hash a file without loading it into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: Any) -> None:
    """Save JSON evidence."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Create revisions, execute three fresh containers and check the evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True)
    probe = output / "probe"
    probe.mkdir()
    shutil.copyfile(Path(__file__).with_name("worker.py"), probe / "worker.py")
    shutil.copyfile(args.policy, probe / "approach.py")
    kinder = repo / "third-party/kindergarden"
    base = git(kinder, "rev-parse", "HEAD")
    edited = output / "editor-worktree"
    git(kinder, "worktree", "add", "--detach", str(edited), base)
    path = edited / EDIT_FILE
    source = path.read_text()
    marker = "    def _sample_initial_state(self) -> ObjectCentricState:\n"
    assert source.count(marker) == 1
    source = source.replace(marker, STEP_EDIT + marker)
    marker = "This variant has no narrow passages (open space)."
    assert source.count(marker) == 1
    path.write_text(source.replace(marker, marker + DESCRIPTION))
    git(edited, "add", EDIT_FILE)
    git(
        edited,
        "-c",
        "user.name=Runtime verification",
        "-c",
        "user.email=runtime-check@example.invalid",
        "commit",
        "-m",
        "Manual round-trip probe: add a spatial drag zone",
    )
    changed = git(edited, "rev-parse", "HEAD")
    (output / "environment.patch").write_text(git(edited, "diff", base, changed) + "\n")
    git(
        edited,
        "bundle",
        "create",
        str(output / "environment-change.bundle"),
        f"{base}..HEAD",
    )
    image = repo / "robocode-sandbox.sif"
    image_id = "sha256:" + digest(image)
    config = {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    }
    manifest = {
        "robocode_commit": git(repo, "rev-parse", "HEAD"),
        "base_environment_commit": base,
        "edited_environment_commit": changed,
        "runtime_image": image_id,
        "policy_sha256": digest(probe / "approach.py"),
        "seeds": list(range(10)),
        "horizon": 200,
    }
    save(output / "run-manifest.json", manifest)
    artifacts = []
    results = []
    commands = []
    for label, revision in [("e0-before", base), ("e1", changed), ("e0-return", base)]:
        artifact = export_environment(
            edited,
            revision,
            output / (label + "-artifact"),
            package="kinder",
            config=config,
            runtime_image=image_id,
            dependencies={
                "uv.lock": repo / "uv.lock",
                "pyproject.toml": repo / "pyproject.toml",
            },
        )
        artifacts.append(artifact)
        phase = output / label
        phase.mkdir()
        shutil.copyfile(
            artifact.path / "manifest.json", phase / "artifact-manifest.json"
        )
        cmd = _apptainer_exec_prefix() + [
            "--bind",
            f"{artifact.path}:/artifact:ro",
            "--bind",
            f'{repo / "src"}:/robocode/src:ro',
            "--bind",
            f"{probe}:/probe:ro",
            "--bind",
            f"{phase}:/sandbox",
            "--env",
            "PYTHONPATH=/artifact/source/src:/robocode/src",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "MUJOCO_GL=osmesa",
            str(image),
            "/robocode/.venv/bin/python",
            "-B",
            "/probe/worker.py",
            artifact.artifact_id,
        ]
        commands.append(cmd)
        save(phase / "command.json", cmd)
        print(
            f"START {label} artifact={artifact.artifact_id} commit={revision}",
            flush=True,
        )
        with (phase / "container.log").open("w") as log:
            process = subprocess.run(
                cmd,
                env=clean_apptainer_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        print((phase / "container.log").read_text(), flush=True)
        process.check_returncode()
        load_environment_artifact(artifact.path, expected_id=artifact.artifact_id)
        results.append(json.loads((phase / "result.json").read_text()))
        (phase / "ready.json").unlink()
    assert (
        artifacts[0].artifact_id == artifacts[2].artifact_id != artifacts[1].artifact_id
    )
    before, changed_result, returned = results[0], results[1], results[2]
    assert before == returned
    trace_before = (output / "e0-before/transitions.jsonl").read_bytes()
    trace_after = (output / "e0-return/transitions.jsonl").read_bytes()
    trace_changed = (output / "e1/transitions.jsonl").read_bytes()
    assert trace_before == trace_after != trace_changed
    meta = [
        json.loads((output / label / "metadata.json").read_text())
        for label in ["e0-before", "e1", "e0-return"]
    ]
    assert meta[0] == meta[2]
    for key in ["action_space", "observation_space"]:
        assert meta[0][key] == meta[1][key]
    assert meta[0]["description"] != meta[1]["description"]
    for frame in before["renders"]:
        assert digest(output / "e0-before" / before["renders"][frame]) == digest(
            output / "e0-return" / returned["renders"][frame]
        )
    assert digest(output / "e0-before" / before["renders"]["step30"]) != digest(
        output / "e1" / changed_result["renders"]["step30"]
    )
    assert all(row["terminated"] for result in results for row in result["episodes"])
    assert all(
        a["steps"] < b["steps"]
        for a, b in zip(before["episodes"], changed_result["episodes"])
    )
    negative = output / "wrong-revision"
    negative.mkdir()
    cmd = commands[1].copy()
    cmd[cmd.index(str(output / "e1") + ":/sandbox")] = str(negative) + ":/sandbox"
    cmd[-1] = artifacts[0].artifact_id
    save(negative / "command.json", cmd)
    with (negative / "container.log").open("w") as log:
        process = subprocess.run(
            cmd,
            env=clean_apptainer_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    assert process.returncode != 0
    assert (
        "Environment artifact identity mismatch"
        in (negative / "container.log").read_text()
    )
    assert (
        not (negative / "ready.json").exists()
        and not (negative / "result.json").exists()
    )
    summary = {
        "status": "passed",
        "e0_identity_restored": True,
        "e0_transitions_byte_identical": True,
        "e0_metadata_restored": True,
        "e0_render_bytes_restored": True,
        "e1_spaces_unchanged": True,
        "e1_description_updated": True,
        "e1_step30_render_changed": True,
        "wrong_revision_rejected_before_service_start": True,
        "policy_sha256_unchanged": len({r["policy_sha256"] for r in results}) == 1,
        "phases": {
            label: {
                "artifact_id": result["artifact_id"],
                "successes": sum(row["terminated"] for row in result["episodes"]),
                "mean_steps": sum(row["steps"] for row in result["episodes"]) / 10,
            }
            for label, result in zip(["e0-before", "e1", "e0-return"], results)
        },
    }
    save(output / "summary.json", summary)
    print("ROUND_TRIP_PASSED " + json.dumps(summary, sort_keys=True), flush=True)
    assert git(edited, "status", "--porcelain") == ""
    git(kinder, "worktree", "remove", str(edited))
    for artifact in artifacts:
        shutil.rmtree(artifact.path)
    print(
        "REMOVED_TEMPORARY_WORKTREE_AND_EXPORTS; kept manifests, commit bundle, "
        "source patch, policy, traces, renders and logs",
        flush=True,
    )


if __name__ == "__main__":
    main()
