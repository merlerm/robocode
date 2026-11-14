"""Adapt frozen P0 once to independently edited sibling environments and E0."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from experiments.exploration_exchange import make_exchange, write_json
from robocode.runtime.artifacts import load_environment_artifact
from robocode.runtime.exchange import Exchange
from robocode.runtime.whitebox import create_whitebox_source


def source_patch(current, other):
    """Git preserves missing newlines, empty files and binary changes."""
    with tempfile.TemporaryDirectory(prefix="robocode-sibling-diff-") as temporary:
        root = Path(temporary)
        for source, name in [(current, "current"), (other, "member")]:
            shutil.copytree(
                source,
                root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--binary",
                "--no-renames",
                "--no-ext-diff",
                "--no-textconv",
                "--src-prefix=",
                "--dst-prefix=",
                "current",
                "member",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in (0, 1), result.stderr
        return result.stdout


def import_siblings(exchange, probes, *, base_only=False):
    """Keep accepted artifacts separate; record rejected siblings without training on them."""
    sources = []
    for probe in probes:
        ledger = json.loads((probe / "run/exchange.json").read_text())
        assert ledger["status"] != "running"
        assert ledger["initial_environment_id"] == exchange.initial.artifact_id
        assert (
            ledger["experiment"]["seed_policy_id"]
            == exchange.ledger["experiment"]["seed_policy_id"]
        )
        accepted = [c for c in ledger["candidates"] if c["status"] == "accepted"]
        sources.append(
            dict(probe=str(probe), status=ledger["status"], accepted=len(accepted))
        )
        assert len(accepted) <= 1
        for candidate in accepted:
            name = f"e{len(exchange.archive)}-imported"
            original = probe / "run/environments" / candidate["turn_id"]
            destination = exchange.directory / "environments" / name
            shutil.copytree(original, destination)
            artifact = load_environment_artifact(
                destination, expected_id=candidate["environment_artifact_id"]
            )
            exchange.archive.append(artifact)
            exchange.ledger["candidates"].append(
                dict(
                    candidate,
                    turn_id=name,
                    imported_from=str(original),
                    source_probe=str(probe),
                    source_turn_id=candidate["turn_id"],
                )
            )
    exchange.ledger["experiment"].update(
        independent_probes=sources, adaptation_only=True
    )
    if len(exchange.archive) == 1 and not base_only:
        exchange.ledger["status"] = "stopped_without_accepted_sibling"
        exchange._save()
        return False
    exchange.active = exchange.archive[-1]
    exchange.ledger["active_environment_id"] = exchange.active.artifact_id
    views = [
        create_whitebox_source(
            a,
            exchange.config.runtime_source,
            exchange.directory / "archive-views" / f"e{i}",
        )
        for i, a in enumerate(exchange.archive)
    ]
    current = views[-1].path / "source"
    exchange.ledger["experiment"]["source_diff_base_id"] = views[-1].source_id
    folder = exchange.directory / "archive-diffs"
    folder.mkdir()
    index = []
    for i, view in enumerate(views):
        (folder / f"e{i}.patch").write_text(source_patch(current, view.path / "source"))
        index.append(
            dict(
                member=f"e{i}",
                environment_artifact_id=view.manifest["artifact_id"],
                source_id=view.source_id,
                config=view.manifest["environment_config"],
            )
        )
    write_json(folder / "members.json", index)
    files = dict(exchange.policy_config.init_files)
    files.update({"archive_sources/" + p.name: p for p in folder.iterdir()})
    exchange.policy_config = replace(exchange.policy_config, init_files=files)
    exchange.policy_instruction = (
        "The training distribution contains the members listed in archive_sources/members.json. "
        "Any edited members were created independently from the same original, not by applying edits sequentially. "
        "archive_sources/members.json describes the members. The eN.patch files give the full "
        "source differences FROM the currently mounted /whitebox/source TO each member. "
        "An empty patch means the mounted source already matches that member. These are reference "
        "copies: changing them does not change training. Use the service for mixture rollouts. "
        "Adapt the existing approach.py to work across the mixture without a revision label."
    )
    exchange._save()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    for name in ["seed-policy", "policy-source", "initial-validation", "budget-ledger"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--probe", type=Path, action="append", default=[])
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--condition", default="independent-editor-mixture")
    parser.add_argument("--policy-budget", type=float, default=5)
    parser.set_defaults(
        cycles=1,
        editor_attempts=1,
        editor_budget=8,
        no_editor_pacing=True,
        editor_instruction_file=None,
        editor_only=False,
    )
    args = parser.parse_args()
    assert (
        bool(args.probe) != args.base_only
    ), "Choose sibling probes or the E0-only control"
    os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    os.environ.pop("ROBOCODE_CLAUDE_AUTH_ALLOW_PARALLEL", None)
    exchange = make_exchange(args)
    try:
        if not import_siblings(
            exchange, [p.resolve() for p in args.probe], base_only=args.base_only
        ):
            return
        for i, artifact in enumerate(exchange.archive):
            Exchange._evaluate(exchange, exchange.seed_policy, artifact, f"p0-e{i}")
        policy = exchange._policy_turn("p1")
        if policy is None:
            return
        assert (
            exchange.policy_sources[policy.resolve()].source_id
            == exchange.ledger["experiment"]["source_diff_base_id"]
        )
        for i, artifact in enumerate(exchange.archive):
            Exchange._evaluate(exchange, policy, artifact, f"p1-e{i}")
        exchange.ledger["status"] = "completed_sibling_adaptation"
        exchange._save()
    except Exception as error:
        exchange.ledger.update(status="failed", error=repr(error))
        exchange._save()
        raise


if __name__ == "__main__":
    main()
