"""Check that a strict blackbox image was built from this checkout.

The strict image bakes the render server and client. Launchers call
``verify_strict_image`` before starting an agent container, so a stale image fails
with rebuild instructions rather than inside the container.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from robocode.utils.apptainer_environment import clean_apptainer_env

# Package-relative sources that docker/Dockerfile.strict-blackbox COPYs into
# STRICT_RENDER_DIR. Keep this list and that COPY line identical (a test compares them).
STRICT_RENDER_SOURCES: tuple[str, ...] = ("mcp/strict_server.py", "utils/env_client.py")
STRICT_RENDER_DIR = "/opt/robocode-render"
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_VERIFIED_IMAGES: set[tuple[str, str]] = set()

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class StrictImageMismatchError(RuntimeError):
    """The strict image bakes render sources that differ from this checkout."""


def _run_hash_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=clean_apptainer_env() if cmd[0] == "apptainer" else None,
    )


def verify_strict_image(
    backend: str, image: str, run: CommandRunner = _run_hash_command
) -> None:
    """Raise unless *image* bakes byte-identical copies of the host render sources.

    Hashes the baked files inside the image once per (backend, image) per process.
    """
    if (backend, image) in _VERIFIED_IMAGES:
        return
    baked = [f"{STRICT_RENDER_DIR}/{Path(rel).name}" for rel in STRICT_RENDER_SOURCES]
    if backend == "docker":
        cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none"]
        cmd += ["--entrypoint", "sha256sum", image, *baked]
        rebuild = "bash docker/build_strict_blackbox.sh"
    elif backend == "apptainer":
        cmd = ["apptainer", "exec", "--containall", "--no-home", "--cleanenv"]
        cmd += ["--pwd", "/", image, "sha256sum", *baked]
        rebuild = "bash docker/build_strict_blackbox_sif.sh"
    else:
        raise ValueError(f"Unsupported strict container backend: {backend}")
    result = run(cmd)
    # sha256sum prints "<digest>  <path>" for each readable file, even on failure.
    image_digests = {
        path: digest
        for digest, path in (
            line.split(maxsplit=1) for line in result.stdout.splitlines() if line
        )
    }
    problems = []
    for rel, path in zip(STRICT_RENDER_SOURCES, baked):
        host = hashlib.sha256((_PACKAGE_DIR / rel).read_bytes()).hexdigest()
        if path not in image_digests:
            problems.append(f"src/robocode/{rel} (missing from image)")
        elif image_digests[path] != host:
            problems.append(f"src/robocode/{rel} (differs)")
    if problems:
        stderr = result.stderr.strip()
        raise StrictImageMismatchError(
            f"Strict {backend} image {image} was not built from this checkout "
            f"({_PACKAGE_DIR}): " + ", ".join(problems) + ". Rebuild it with "
            f"`{rebuild}`, or set ROBOCODE_STRICT_IMAGE (Docker) or strict_sif_path "
            "(Apptainer) to an image built from this checkout."
            + (f"\n{backend} stderr: {stderr}" if stderr else "")
        )
    _VERIFIED_IMAGES.add((backend, image))
