# Whitebox verification, 2026-11-10

Whitebox Codex synthesis works in the fork's isolated Apptainer runtime. The live
agent read environment source, tested and revised its policy, then completed with
10/10 held-out Motion2D Easy episodes solved, zero crashes and 39.1 mean steps.
Both determinism checks passed. Recorded token cost was $1.0189664 under the $2
cap; synthesis took 271.8 seconds. No API errors or permission denials were reported.

Source: `e54813e` on `multi-agent-runtime`; KinDER `5b2dbac`. The model was
`gpt-5.6-sol`, medium, Codex CLI 0.153.4. Settings match the preceding blackbox
smoke except `approach.blackbox=false` and `approach.blackbox_strict=false`:
no primitives, replicate seed 42, eval seed 314159, 10 evaluation tasks, 1000-step
cap, 60-second policy-time limit, two determinism checks, no evaluation videos.
Cost is the repository's estimate using the existing ChatGPT login, not an invoice.

## Evidence

Local artifacts: `outputs/whitebox-verification-2026-11-10/`.

- `live/stream.jsonl:16`: the agent reads `motion2d.py` and the base environment.
- `live/stream.jsonl:30`: its executed development test reports 1000 seeds passed.
- `live/results.json`: independent held-out results and generation accounting.
- Final sandbox policy commit: `1c971fc1706793e105012e0fe08187dbd5f3666d`.
- `probe.log`: no-model checks through the real Apptainer launcher, source mounts,
  prepared dependencies and isolated transport. Source is readable; oracle code
  and recorded demos are absent. Motion2D reset/step/state work. Both render tools
  produce valid PNGs. Packing3D (PyBullet) and ConstrainedCupboard3D (MuJoCo, plain
  background) reset, step and render successfully. Rendered images are retained.
- The live agent did not call a render tool; rendering evidence comes from the
  separate container probe. Its MCP server did start successfully.
- 354 tests passed, no skips: 194 approach/prompt/environment/render checks,
  51 Docker checks, and 109 backend/transport/accounting/resume checks. These
  include actual Docker and strict Apptainer containers; Claude continuation uses
  a fake CLI and is not live Claude verification.

## Setup and fix

Initialized the pinned KinDER, kinder-baselines and ss-pybullet submodules. Copied
existing MimicLabs assets into the fork: isolated agents cannot download them.
The original checkout and its interpreter were not modified.

Fixed `docker/Dockerfile` to retain installed dependency metadata and propagate
installation failures. Previously it deleted all dist-info/egg-info/pth files,
forcing dependency reinstalls, and a trailing `true` hid installer failures.
The actual container checks passed after this change.

The local image was built with the frozen lockfile, seeded from existing CLIs
and simulator packages to avoid slow downloads. Updated package files were checked
against their installed RECORD hashes. An obsolete overlapping PyBullet package
in the old seed was removed before final synchronization. A clean network-only
build was not completed. `Dockerfile.seeded`, build logs, image hashes, CLI
versions and seed provenance are retained with the outputs.

Ready-to-use local images: `robocode-sandbox:multi-agent-runtime` (also tagged
`robocode-sandbox:latest`) and the fork's ignored `robocode-sandbox.sif`. Source
mounts remain runtime inputs. Scratch, test directories and the prepared-venv
cache were removed; the next run will recreate that cache. No task processes remain.

`run.py` records the exact live command; recreate its short disk-backed
`outputs/wb-tmp` directory before using it and choose a fresh output directory.
Do not use a long TMPDIR: the broker's Unix socket has an OS path-length limit.
`probe.py` records the non-model checks. Logs and policy history remain local and
ignored; only the build fix and this report are committed.

## Limits

This verifies the exercised whitebox path, not every task/primitive combination.
The live model run used Apptainer; Docker has real container checks but no live
model call in this verification. Claude quota/access and cluster validation remain
pending. Final evaluation still runs on the host. Revision-bound environment and
evaluation containers and durable coordinated agent turns remain outstanding in
[checkpoint 1](checkpoint-1.md).
