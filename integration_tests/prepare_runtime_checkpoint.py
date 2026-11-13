"""Export the frozen inputs used by the opt-in runtime acceptance tests."""

import argparse
from pathlib import Path

import yaml

from robocode.runtime.artifacts import export_environment
from robocode.runtime.containers import image_identity
from robocode.runtime.snapshots import snapshot_runtime
from robocode.runtime.whitebox import create_whitebox_source


def main() -> None:
    """Create a new evidence directory using the local checkpoint images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--backend", choices=["docker", "apptainer", "all"], default="all"
    )
    parser.add_argument("--environment-config", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    snapshot_runtime(root / "service-runtime")
    config = {
        "_target_": "robocode.environments.kinder_geom2d_env.KinderGeom2DEnv",
        "env_id": "kinder/Motion2D-p0-v0",
    }
    if args.environment_config is not None:
        config = yaml.safe_load(args.environment_config.read_text(encoding="utf-8"))
    for backend, image in [
        ("docker", "robocode-sandbox:multi-agent-runtime"),
        ("apptainer", str(repo / "robocode-sandbox.sif")),
    ]:
        if args.backend not in {backend, "all"}:
            continue
        artifact = export_environment(
            repo / "third-party/kindergarden",
            "HEAD",
            root / (backend + "-artifact"),
            package="kinder",
            config=config,
            runtime_image=image_identity(backend, image),
            dependencies={
                "uv.lock": repo / "uv.lock",
                "pyproject.toml": repo / "pyproject.toml",
            },
        )
        view = create_whitebox_source(
            artifact, root / "service-runtime", root / (backend + "-whitebox")
        )
        print(backend, artifact.artifact_id, view.source_id, flush=True)


if __name__ == "__main__":
    main()
