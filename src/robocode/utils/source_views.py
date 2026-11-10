"""Exclude solution surfaces from agent-visible source trees."""

import shutil
from pathlib import Path


def copy_environment_source(
    kindergarden: Path, dest: Path, *, blackbox: bool = False
) -> None:
    """Copy ``kindergarden`` to *dest*, skipping ``tests/``, ``docs/`` and ``demos/``.

    ``demos/`` holds recorded solutions. Blackbox also skips ``kinder/envs/``;
    the installable skeleton stays so ``uv sync --frozen`` succeeds.
    """
    skip = ("tests", "docs", "demos") + (("envs",) if blackbox else ())
    shutil.copytree(
        kindergarden,
        dest,
        ignore=shutil.ignore_patterns(*skip),
    )


def copy_runtime_source(
    src: Path, dest: Path, *, keep_primitives: bool = False, blackbox: bool = False
) -> None:
    """Hide oracles; agents also lose baselines and ungranted primitives."""
    if keep_primitives:
        skip: tuple[str, ...] = ("oracles",)
    else:
        skip = ("oracles", "primitives", "primitive_descriptions.py", "approaches")
    if blackbox:
        skip += ("environments",)
    # Bytecode retains the same withheld source strings even after .py is omitted.
    skip += ("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*skip))
    if "approaches" in skip:
        approaches = dest / "robocode" / "approaches"
        approaches.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            src / "robocode" / "approaches" / "base_approach.py",
            approaches / "base_approach.py",
        )
