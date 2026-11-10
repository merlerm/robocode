# Whitebox source tied to an environment revision

The artifact runner now supports generalized whitebox KinDER policies as well as
strict blackbox policies. Both use an isolated evaluator and a separate environment
service. The host never loads the submitted policy or simulator.

Whitebox agents can read the current environment code, instantiate it locally,
and use `env_client.make_env()` for authoritative service rollouts. Submitted
policies may import the matching simulator, but cannot depend on `env_client`.

## Source and execution

The runner freezes RoboCode's wrapper/runtime source once in `service-runtime/`.
Both environment services use that snapshot. `whitebox-source/` contains filtered
copies of the verified environment artifact and wrapper source, excluding the
existing oracle, baseline, primitive and demonstration surfaces.

`view.json` records the environment artifact, image, wrapper snapshot, file hashes,
and bootstrap hash. Its content identity is checked before synthesis and evaluation.
Mismatched environment bindings, source changes, wrapper changes or images fail
loudly. Each new environment revision needs a new view and a fresh container.

The policy sandbox mounts this view read-only. It uses the artifact image's pinned
dependencies without reinstalling from the working checkout. A small bootstrap
verifies the mounted files and actual KinDER import origins before launching either
Claude or Codex, saving `sandbox/source-provenance.json`. Telemetry preserves these
import paths. Evaluation uses the same source view and image, with a separate RPC
connection to a fresh environment service. Results contain `whitebox_source_id`,
actual import paths, and the existing policy/environment identities.

Strict blackbox synthesis and evaluation receive no whitebox view. The generic
blackbox evaluator image remains independent of simulator dependencies.

## Invocation

Use the existing artifact runner with:

```text
approach.blackbox=false
approach.blackbox_strict=false
primitive_level=none
environment_runtime=artifact
environment_runtime.artifact=<export directory>
environment_runtime.expected_id=<artifact identity>
environment_runtime.image=<artifact simulator image>
environment_runtime.evaluation_image=<same simulator image>
```

Set `approach.container_backend=docker` or `apptainer`; the latter takes absolute
SIF paths. Docker resolves an immutable image ID. SIF bytes are hashed. Box spaces,
generalized synthesis, no primitives and no approach-history replay are the current
artifact pilot's scope. Existing fixed-environment modes keep their previous paths.

## Verification

`tests/runtime/test_whitebox.py` checks frozen views, committed E0/E1 edits, stale
handoffs, tampered files/bootstrap/wrappers, and blackbox rejection. The opt-in
`test_whitebox_containers.py` uses `ROBOCODE_WHITEBOX_TEST_ARTIFACTS`, containing
`{docker,apptainer}-artifact`, `{docker,apptainer}-whitebox` and `service-runtime`.
It runs real launchers with Claude/Codex setup and a deterministic probe replacing
model inference. It checks read-only source, identical local/server reset and step,
rendering, telemetry, provenance, and evaluator imports/local simulation.

Saved whitebox-policy replays on both backends reproduce 10/10 solved, 39.1 mean
steps and full agreement on two determinism replay pairs, with host policy loading
forbidden and no host KinDER imports. Evidence, commands, logs and videos are in
`outputs/whitebox-artifacts-2026-11-10/`. No paid model requests were made. Cluster
execution and live Claude continuation are still separate acceptance checks.
