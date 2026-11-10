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
  ROBOCODE_TEST_STRICT_IMAGE to this tag when running strict container tests.
  Scoped mypy and pylint passed (pylint reports legacy configuration warnings).

The existing paper image is deliberately not overwritten. Rebuild runtime images
when the protocol changes; old clients are not supported.
