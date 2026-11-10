#!/bin/bash
# Artifact runs use the pinned image's dependencies; never resync the checkout.
set -euo pipefail
/usr/local/bin/init-firewall.sh
export HOME=/home/node USER=node LOGNAME=node
exec /usr/bin/setpriv --reuid=node --regid=node --init-groups \
    --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs -- "$@"
