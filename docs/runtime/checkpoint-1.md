# Runtime parity checkpoint

Base: upstream/main d3d42dd12e74c00383ceeee5e75f16bd8c4abc09.
Development remote: merlerm/robocode. Upstream pushes disabled locally.

The original viewer checkout is dirty and is not part of this branch. Its
untracked StickButton variant is not a dependency of this work.

Acceptance: preserve generalized/per-instance approach behavior, source access
modes, rewards/steps/timeouts and model-backend behavior; isolate evaluation;
explicit environment bindings and revision provenance; render transport without
shared workspaces; durable agent continuation for Claude and Codex.

Execution plan:
1. Baseline and focused regression coverage.
2. Render transport, bounded requests and explicit service bindings.
3. Revision-bound isolated environment and evaluation execution.
4. Role-neutral persistent turns and backend parity checks.

CUDA and collaborative policy synthesis are deferred. Live Claude checks await
user-supplied quota/access; do not mark them verified from mocked tests.

## Verified layers

- Render RPC returns PNG bytes; each client writes its own render artifacts. The
  environment server no longer takes a policy workspace path. Concurrent clients
  retain separate files and repeat renders do not overwrite prior artifacts.
- Render/strict regressions: 61 passed, 2 skipped (strict Apptainer image absent).
  Tested Docker image: robocode-strict-blackbox:multi-agent-runtime. Set
  ROBOCODE_STRICT_IMAGE to this tag for fork runs and strict container tests.
  Scoped mypy and pylint passed (pylint reports legacy configuration warnings).

The existing paper image is deliberately not overwritten. Rebuild runtime images
when the protocol changes; old clients are not supported.


## Service ownership layer

The experiment runner now owns environment services through an ExitStack. Both
agentic implementations receive an EnvironmentBinding; finishing synthesis no
longer stops the service. Each service has a fresh directory, log and failure callback. Reusing a
directory raises FileExistsError; there is no lock or status-file layer. The runner still interrupts its current experiment on a
service crash; the reusable service no longer implicitly interrupts the main
thread. Telemetry retains the existing experiment sink and run ID.

RPCs have a finite absolute deadline (600 seconds by default, independently of
the existing evaluation policy-time limit). A timed-out or malformed exchange
closes the connection and is never retried. This includes trickling responses.
Service teardown terminates its process group and reaps its server process.

Validation: 142 affected regression tests passed, 2 strict Apptainer container
checks skipped because that image is unavailable. Two additional subprocess
checks passed for hung reset and step cleanup. Scoped mypy and pylint passed.
No live model calls were made.

## Outstanding checkpoint work

The full parity checkpoint is NOT complete. Environment code, metadata discovery,
and evaluation still execute on the host in this layer. Next implement committed
source exports with dependency/config/image identity and origin verification,
then containerized metadata and evaluation under the same source-access rules.
After that extract role-neutral durable turns and check both backend adapters,
including stale outputs, cumulative budgets and native continuation. Live Claude
verification still requires the access the user offered; a passing mock is not
backend parity. No adversarial objective or editor agent has been added yet.


## Research-code simplification and Apptainer check

Removed the unused status files and filesystem lease, shortened service comments,
and made the Docker/Apptainer/local address mapping explicit. Exceptions propagate;
the remaining catches handle failed-connection cleanup, server-crash reporting,
and terminating processes that may already have exited.

Apptainer 1.5.3 was already installed. Built the missing
`robocode-strict-blackbox.sif` from the local fork Docker image. The container tests
now use the actual Apptainer network namespace, supervisor and environment relay;
the previous render helper shared host networking. Both real Apptainer checks pass:
clean numerical dependencies, and state/policy rendering through the service.
Docker passes the same two probes. Across the focused suites, 164 tests passed
with no skips; scoped mypy and pylint passed. Most are existing regression tests.
No model API calls are involved.

The SIF is local and ignored. Build scratch is removed after validation. Repeat
these real-runtime tests on the target cluster: local results do not establish
that its node policy permits the same user/network namespaces.

## Live before/after smoke

[2026-11-10 smoke results](smoke-2026-11-10.md): Codex in strict Apptainer
completed on both the baseline and refactored code. Both solved 10/10 Motion2D
Easy episodes at 39.1 mean steps, with no crashes; total recorded cost $0.895646.
Replaying the same policy across revisions matched exactly. The after run also
used render_state successfully. This adds live Codex evidence, not Claude or
cluster verification, and does not close the outstanding checkpoint work.

## Whitebox verification

[2026-11-10 whitebox results](whitebox-smoke-2026-11-10.md): live Codex in
Apptainer read source, tested its policy and solved 10/10 held-out Motion2D
episodes at 39.1 mean steps for $1.0189664. Separate isolated probes passed
2D and PyBullet/MuJoCo simulation and both whitebox render tools. 354 relevant
tests passed with no skips. Fixed the whitebox Dockerfile's swallowed build
errors and removed unnecessary dependency-metadata deletion. Live Claude,
cluster execution and the outstanding architectural work remain pending.
