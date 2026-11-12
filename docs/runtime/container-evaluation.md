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

Blackbox evaluation receives no simulator source, artifact mount, agent credentials, Docker
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

Current support covers generalized strict-blackbox and whitebox agentic synthesis
on fixed (Box) and variable-count (object-centric) environments, with no primitives
and no approach-history replay. Whitebox evaluation uses
[revision-bound source and simulator images](whitebox-artifacts.md). Per-instance
paths retain their existing behavior. Live Claude continuation, cluster verification
and alternating editor turns remain work.

## Observations and object counts

A policy is scored with the observation view its author had during synthesis. In
strict blackbox, it receives the strict `env_client` view: a Box with no conversion
helpers, or the client's object-centric space (`types`, `type_features`,
`get_type`) and states. In whitebox, the evaluator instantiates the artifact
configuration beside the pinned source and passes that environment's own observation
space, so Box spaces keep `devectorize`/`vectorize`, and object-centric states are
rebuilt as the environment's state class and types. The service's float32 state
codec still applies to those states.

A service that reports `counts` makes the runner-side `RemoteEnvironment` a
`VariableCountEnv`. The runner cycles its `eval_counts` over the held-out seeds, as
on the host path, and passes that schedule to the evaluator. Each episode pins its
count and uses the service's per-count step limit; determinism replays use the same
counts. Results carry `by_count`, `largest_count_all_solved`,
`largest_count_any_solved`, per-episode `object_count`, `count_regimes`,
`design_count_solve_rate` and `held_out_count_solve_rate`.

Evidence is retained in `outputs/evaluation-container-2026-11-10/`. Both backends
replay the saved Motion2D policy at 10/10 solved, 39.1 mean steps, and full agreement
on two determinism replay pairs. The host-load guard makes any host policy loading
fail and checks that KinDER was not imported. Initial runs also save videos.
Live checks cover hidden source/host files, read-only submissions, policy timeouts,
native worker exits, load errors, tagged tracebacks and cleanup. Snapshot checks
cover helper/data changes and symlink rejection. No live model calls were used.
