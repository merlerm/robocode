# Frozen-policy evaluation containers

The artifact-backed runner now loads and evaluates generated policy code only
inside an evaluation container. `AgenticApproach.synthesize()` returns files;
`train()` retains the existing local-load behavior for fixed-environment runs.
This path is independent of whether Claude or Codex produced the submission.

## Layout

- Host: orchestrates synthesis, snapshots files, schedules held-out seeds and saves results.
- Development environment: frozen simulator service used during synthesis.
- Policy agent: existing isolated synthesis sandbox, with no held-out seed access.
- Evaluation environment: a fresh service at the same verified artifact revision.
- Evaluator: frozen policy plus generic scoring code; environment operations use RPC.

Evaluation receives no simulator source, artifact mount, agent credentials, Docker
socket or host network. A fixed Unix-socket relay reaches only its environment
service. Policy and harness mounts are read-only; only evaluation output and `/tmp`
are writable. This preserves the blackbox source boundary. It does not claim to
make the in-process scorer tamper-proof against deliberately malicious policy code.

## Images and invocation

Build the generic evaluator from the repository root:

```sh
docker build -t robocode-evaluation:multi-agent -f docker/Dockerfile.evaluation docker
```

For Apptainer, save that image with `docker save`, then run `apptainer build` from
the `docker-archive://` tar. Put `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` on disk,
not this machine's RAM-backed `/tmp`. The verified local SIF is
`robocode-evaluation.sif`. Both are reusable images, separate from simulator images.

Alongside the existing artifact settings, pass:

```text
environment_runtime.evaluation_image=robocode-evaluation:multi-agent
```

For Apptainer use the absolute SIF path. The runner resolves the evaluation image
before synthesis to catch missing images/configuration before spending agent budget.
Docker launches the resolved image ID; SIF content identity is checked again after
execution. Actual image identities, rather than mutable names, enter results.

## Frozen inputs and outputs

`evaluation/policy/` snapshots policy files, helpers and data. Development transport
files, agent instructions and caches are excluded. Symlinks fail loudly. Its
`policy.json` hashes filenames and bytes. `evaluation/runtime/` contains only the
explicit generic harness files, hashed in `runtime-source.json`; no RoboCode
simulator wrappers enter this tree. Both snapshots are retained for inspection.

`evaluation/output/` contains raw results and videos. The ordinary experiment
`results.json` and `videos/` remain available to existing consumers. Results,
episodes and determinism replays identify policy and environment artifacts; run
metadata also identifies both runtime sources and images. Tagged `evaluation.log`,
`log-context.json`, `command.json`, and the separate environment-service logs make
process origins explicit. Synthesis cannot access the evaluation input directory.

## Timing and failures

The evaluator calls the existing episode implementation and shared scoring/replay
routines. Policy reset/action/update time is charged to the existing policy budget;
simulator work stays outside it. The 600-second episode wall backstop remains.
Policy loading has a 60-second startup bound; a batch has a finite outer deadline.
Timeouts and episode crashes keep their established scores. Loading failures and
container failures abort loudly. The shared launcher kills/reaps owned workers and
removes relays on normal completion, failure or interruption.

## Scope and verification

Current support remains generalized strict-blackbox agentic synthesis, Box spaces,
no primitives, and no approach-history replay. Fixed-environment and per-instance
paths retain their existing behavior. Artifact whitebox source mounting, live
Claude continuation, cluster verification and alternating editor turns remain work.

Evidence is retained in `outputs/evaluation-container-2026-11-10/`. Both backends
replay the saved Motion2D policy at 10/10 solved, 39.1 mean steps, and full agreement
on two determinism replay pairs. The host-load guard makes any host policy loading
fail and checks that KinDER was not imported. Initial runs also save videos.
Live checks cover hidden source/host files, read-only submissions, policy timeouts,
native worker exits, load errors, tagged tracebacks and cleanup. Snapshot checks
cover helper/data changes and symlink rejection. No live model calls were used.
