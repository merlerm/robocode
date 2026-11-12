# Runtime refactor checkpoint validation

This checkpoint retains the one-generation workflow. It adds versioned environment
services, isolated evaluation and pinned whitebox source; it does not add alternating
agent turns. The acceptance tests cover the implemented artifact pilot and the
existing fixed-environment approaches separately.

## Reproduce the checks

Initialize declared dependencies with `git submodule update --init --recursive`.
Use the project Python environment and the fork's `src` on `PYTHONPATH`. On this
machine the interpreter is the original checkout's existing venv; do not modify it.
Its optional PDDLStream planner can be imported directly from the fork:

```sh
export PYTHONPATH=src:.:third-party/kinder-baselines/kinder-pddlstream-planning/src
unit_tmp=$(mktemp -d /tmp/rc-checkpoint.XXXXXX)
python -m pytest tests -q --basetemp="$unit_tmp"
```

Small socket-based fixtures need a short path: Unix sockets have a roughly
108-byte pathname limit. Remove that task-owned temporary directory after preserving
the test log and checking that no test still uses it. Large exports stay on disk.

Use the host's supported graphics backend. This host supports EGL, while the
container images provide OSMesa. Forcing `MUJOCO_GL=osmesa` on the host without
libOSMesa makes MuJoCo imports and environment registration fail.

Build the Docker/SIF images described in the service and evaluator documentation,
then prepare a **new directory on disk** (not RAM-backed `/tmp`):

```sh
python integration_tests/prepare_runtime_checkpoint.py outputs/checkpoint-tests/assets
export ROBOCODE_SERVICE_TEST_ARTIFACTS=$PWD/outputs/checkpoint-tests/assets
export ROBOCODE_WHITEBOX_TEST_ARTIFACTS=$ROBOCODE_SERVICE_TEST_ARTIFACTS
export ROBOCODE_CHECKPOINT_TEST_ARTIFACTS=$ROBOCODE_SERVICE_TEST_ARTIFACTS
export ROBOCODE_TEST_APPTAINER_IMAGE=$PWD/robocode-strict-blackbox.sif
python -m pytest tests/runtime tests/utils/test_codex_budget.py \
  tests/integration_tests/test_red_team_sandbox.py -q \
  --basetemp=outputs/checkpoint-tests/containers
```

The preparation helper accepts `--backend apptainer` or `--backend docker`; select
that backend's cases with pytest `-k` when only one runtime is installed. Use fresh
artifact directories for repeated runs: the revision tests create committed test
repos and additional exports beside the base artifact. Preserve logs/manifests and
remove those test repos and large source exports after checking that no test still
uses them. No model inference is needed by these checks. The separate tests marked
`integration` remain opt-in because they can make paid calls.

## Acceptance coverage

| Boundary | Checks |
|---|---|
| Existing behavior | Generalized/per-instance agentic, CDL, GenPlan, best-of-k, random/oracle and planner regressions; primitives and variable-count environments |
| Full runner | Prompt/source binding, synthesis, MCP rendering, policy snapshot, evaluator and host-import guards on both runtimes |
| Providers | Claude/Codex setup through real launchers; deterministic CLI substitutes |
| Environment families | Motion2D, DynPushT2D physics and Packing3D; blackbox and whitebox evaluators; rendering and deterministic replays |
| Revision identity | Committed E0 → half-motion E1 → E0; local simulation and scoring server both execute the intended revision |
| Independence | Concurrent clients and simultaneously live E0/E1 services; closing E1 leaves E0 usable |
| Policy submission | Helpers and JSON data copied with the program; immutable policy/runtime mounts and per-episode identity metadata |
| Failure handling | Load errors, actual startup hang, rollout timeout, native crash, dead service and relay/container cleanup |
| Agent plumbing | Actual MCP initialize/list/render calls, telemetry, synthetic budget cutoff and retry/session tests |

The full-runner tests prohibit host loading of generated policy code and assert
that no KinDER modules entered the host interpreter. E1 is an actual source edit,
committed in a disposable editor repository; a new export/view and fresh evaluator
must observe its changed dynamics. These are infrastructure checks, not robustness
or adversarial-policy performance results.

## Findings and limits

The broader sweep found audit/benchmark scripts still using the retired
`env_server_running` API. They now use `environment_service` and `EnvironmentBinding`.
The 3D concurrency script uses a currently registered environment and checks error
replies. A Docker cleanup test now tolerates capitalization changes in CLI errors.

Artifact support remains generalized KinDER, Box spaces, no primitives and no
approach-history replay. Other modes retain their existing execution paths. LIBERO
requires its optional installation. Live provider behavior and cluster execution
remain distinct checks; fake CLI tests do not establish model behavior or live
Claude continuation.

Evidence and machine/toolchain metadata are retained in
`outputs/checkpoint-validation-2026-11-10/`. The vault's checkpoint validation report
records final case counts and the combined verdict across the broad suite and
focused live runs.

Final verification on 2026-11-10: **1,311 distinct test cases passed**, after
rechecking failures and completing the cases left by the first run's failure cap.
`verification-summary.json` combines JUnit results without counting reruns twice.
The optional LIBERO module was not collected because its dependencies are absent;
two paid integration tests were deselected. Initial failures and corrections remain
in the raw logs: host graphics configuration, overly long socket-fixture paths,
and the Docker error-message assertion described above.
