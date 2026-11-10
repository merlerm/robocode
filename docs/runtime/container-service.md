# Artifact-backed environment service

The experiment runner can now get descriptions/spaces and execute reset, step
and rendering from a frozen environment in a Docker or Apptainer container.
The host imports no KinDER code on this path. Generated-policy evaluation still
runs in a host subprocess; moving it into its own container is the next layer.

## Current supported path

Select `environment_runtime=artifact` and provide:

- `environment_runtime.artifact`: exported artifact directory.
- `environment_runtime.expected_id`: the expected content identity.
- `environment_runtime.image`: local Docker image reference or SIF path.
- Backend defaults to `approach.container_backend`; both must currently match.

This first runner integration supports generalized `AgenticApproach`, strict
blackbox, Box spaces and no primitives. The resolved environment configuration
must exactly match the artifact. A mismatch fails; there is no host fallback.
Whitebox, other approach lifecycles and approach-history replay fail explicitly
on this path. Existing fixed-environment runs keep their current behavior.

Whitebox needs matching source in both synthesis and policy evaluation. Since the
current evaluator imports generated code on the host, enabling it now could load
a policy against the wrong local simulator revision. Extend it after evaluation
is isolated; do not silently weaken the revision contract.

## Execution and provenance

The launcher verifies artifact contents and actual local image identity, then
starts the resolved image with no network, no capabilities and read-only source
mounts. A Unix socket exposes the environment API through a fixed host relay;
existing agent Docker/Apptainer transports consume the same binding. Policy
containers receive no environment artifact mount. Socket directories are small,
short-lived `/tmp` directories; simulator exports stay on the caller's filesystem.

Worker startup instantiates the configured environment, obtains descriptions and
spaces, checks imported module origins, then publishes metadata atomically.
Loaded origins are checked again for each successful RPC. The trusted RoboCode
runtime source is copied once into a read-only snapshot and hashed separately,
so editing the checkout cannot change a live worker. This hash supplements the
environment artifact identity; the artifact's lockfiles remain declared provenance,
not a request to install packages dynamically.

Each service saves `metadata.json`, `runtime-source.json`, redacted `command.json`
and `env_server.log`. On shutdown it archives the exact trusted source snapshot
as `runtime-source.tar.gz`; both backends' archived files were checked against
the recorded hash manifest. Results and per-episode records carry environment identity;
results also include image and trusted-runtime source identities. Determinism
records include their environment identity. Optional telemetry is retained under
the service and copied to the requested experiment sink on shutdown. For this
intermediate path it covers service operations, including remote evaluation calls.

Startup has a 60-second bound. A dead worker notifies the experiment owner.
Teardown removes the Docker container or terminates/reaps the Apptainer process
group, then releases relays and temporary source/socket directories. Client RPC
absolute deadlines remain unchanged.

## Verification, 2026-11-10

Local evidence: `outputs/container-service-2026-11-10/`.

| Check | Docker | Apptainer |
|---|---|---|
| Real service reset, step, render | Pass | Pass |
| Strict policy container cannot import/read environment source | Pass | Pass |
| Strict policy container reaches service for reset/step/render | Pass | Pass |
| Normal runner, saved policy, 10 episodes | 10/10 solved | 10/10 solved |
| Mean steps, same original eval seed | 39.1 | 39.1 |
| Determinism replay agreement, two episodes | 100% | 100% |
| No host KinDER imports during full runner | Pass | Pass |
| Wrong image/config and failed startup rejected | Pass | Pass |
| Caller exception and worker-death cleanup | Pass | Pass |

Original runner checks also generated episode GIFs. Final runner checks verify
per-episode/determinism identities and matching runtime-source hashes across
backends. No model calls were made. Live backend/cluster compatibility remains a
separate check; local Apptainer is not evidence about cluster namespace policy.

115 distinct checks passed across the affected artifact, service, runner, agentic
and transport suites, including eight live container checks. The first Docker
runner uncovered a teardown race, now fixed. The first worker-death test relied
on Docker CLI SIGTERM forwarding; it now kills the actual worker and verifies
owner notification. Docker also uses an init process to forward signals and reap
children. Initial failure logs are retained. Scoped formatting, type and lint
checks pass; pylint emits existing legacy configuration warnings.

To rerun live tests, export fresh artifacts for both local images into a directory
as `docker-artifact/` and `apptainer-artifact/`, then set
`ROBOCODE_SERVICE_TEST_ARTIFACTS` to that directory when running
`tests/runtime/test_container_service.py`. Keep pytest's base temp directory on
disk, not RAM-backed `/tmp`, when including real KinDER exports.

## Log origins

Every captured environment-container output line (including prints and tracebacks)
has a `[role=environment service=... backend=... instance=...]` prefix. The instance
is the Docker container name or an assigned Apptainer execution identifier.
`log-context.json` records the same fields, even when startup fails. Successful
service metadata also records this context and the actual Docker container ID.
Host launch/stop messages use `role=coordinator` and identify the target instance.
Existing policy-agent streams keep their provider format; their launch logs already
identify the Docker container name or Apptainer run ID.

Logging verification: both real reset/step/render probes and all eight live lifecycle
checks passed. The checks assert every captured line is tagged, including startup
tracebacks. Evidence: `outputs/container-log-tags-2026-11-10/`. Scoped mypy,
pylint and formatting checks pass (existing pylint configuration warnings remain).
