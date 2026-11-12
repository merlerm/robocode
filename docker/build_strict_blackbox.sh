#!/usr/bin/env bash
# Build the strict blackbox Docker image under the tag the code uses by default
# (STRICT_BLACKBOX_IMAGE in src/robocode/utils/strict_blackbox.py).
# ROBOCODE_STRICT_IMAGE selects another tag for both the build and the runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ROBOCODE_STRICT_IMAGE:-robocode-strict-blackbox:multi-agent-runtime}"

echo "Building ${IMAGE} from ${REPO_ROOT} ..."
docker build \
    --tag "${IMAGE}" \
    --file "${REPO_ROOT}/docker/Dockerfile.strict-blackbox" \
    --build-arg "USER_UID=$(id -u)" \
    --build-arg "USER_GID=$(id -g)" \
    "${REPO_ROOT}"
echo "Done. Image tagged: ${IMAGE}"
