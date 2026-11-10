# Manual environment round trip, 2026-11-10

The same saved Motion2D policy solved all 10 seeds in each phase:

| Revision | Mean steps | Solved |
|---|---:|---:|
| E0 original | 39.3 | 10/10 |
| E1 spatial drag | 108.8 | 10/10 |
| E0 restored | 39.3 | 10/10 |

E1 adds transition code: once robot x >= 1.2, multiply translational actions by
0.25. Goal, reward, spaces and policy remain unchanged. The first differing
transition for seed 0 is step 20: dx is 0.0500 in E0 and 0.0125 in E1. Earlier
states/actions match. All 10 seeds exhibit the expected quarter displacement at
the first divergence.

The E0 return restores the same artifact identity, metadata, byte-identical
transition logs and saved PNGs. Each phase starts a fresh Apptainer container and
uses the existing strict RPC service. Loaded KinDER module paths are verified
inside the frozen export before and after serving. A negative test mounts E1
while requesting E0 and fails before service startup. No model calls.

This is a manual harness with service and client in the same container. It does
not test filesystem isolation between them or integrate the normal experiment
runner with separate environment/evaluation containers. Those remain next work.
These seeds are 0–9; the earlier 39.1-step smoke result used a different eval seed.

## Inspect locally

`outputs/environment-round-trip-2026-11-10/index.html` contains synchronized
videos, a step scrubber, trajectory plot, every phase's logs, per-seed scores and
links to manifests, exact commands, source patch and unchanged policy.

Videos replay recorded observations using the unchanged E0 renderer; they do
not rerun the policy. Nine reference frames match the original service PNGs
exactly. Browser playback was not exercised because no browser was connected;
control logic, links and H.264 files were checked separately.

The evidence directory retains all transition logs, artifacts' manifests, the
committed E1 edit as a Git bundle, replay/report scripts, videos and process logs.
Large temporary exports, editor worktree and intermediate video frames are removed.
The local vault also holds a linked report and a portable ZIP of the evidence.

## Reproduce

With the pinned KinDER submodule and local `robocode-sandbox.sif` available:

```sh
PYTHONPATH=src:. /path/to/python docs/runtime/round_trip/run.py \
  outputs/new-round-trip /path/to/saved/approach.py
```

Use a fresh destination. The script records source commits, SIF SHA256, policy
SHA256, configuration, seeds and exact commands. It verifies the round trip and
removes its temporary Git worktree and exported trees after success. Failures
propagate with logs retained. Scoped formatting, mypy and pylint pass; pylint
emits existing legacy configuration warnings.
