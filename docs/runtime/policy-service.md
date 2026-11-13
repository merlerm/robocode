# Frozen-policy service

`robocode.runtime.policy_service.policy_service` serves a frozen strict-blackbox
policy from the generic evaluation image, so another agent can call it like an API
without seeing its files. Callers use `robocode/utils/policy_client.py`, a
standalone file (standard library and numpy) meant to be copied into a workspace,
like `env_client.py`.

```python
with policy_service(
    policy, binding.metadata, service_dir,
    backend="docker", image="robocode-evaluation:multi-agent",
    blackbox_strict=True, seed=replicate_seed,
) as policy_binding:
    with PolicyClient(policy_binding.client_metadata()) as policy:
        policy.reset(state, info)
        action = policy.act(state)
        policy.update(next_state, reward, done, info)
```

## Lifetime and provenance

The host snapshots the submission with `snapshot_policy`, runs the strict import
check on it, and copies the source subset `POLICY_RUNTIME_FILES` (the evaluator's
generic files plus `runtime/policy_worker.py`). Both are hashed in `policy.json` and
`runtime-source.json`; the worker verifies both hashes before serving. The service
directory also holds `input/request.json`, a redacted `command.json`,
`log-context.json`, `metadata.json` (policy, runtime source and image identities,
Docker container ID) and the tagged log `policy_service.log`. Startup is bounded at
60 seconds. On exit the container is removed (Docker) or its process group is
terminated (Apptainer), the relay stops and the socket directory is deleted. The
image identity is checked again on every exit: a change is an error after normal use
and a note on the caller's own exception otherwise.

Only policies written in strict blackbox mode are accepted. `blackbox_strict=False`,
non-Box action spaces, and observation spaces other than Box or ObjectCentric fail
on the host before any container starts, as does a strict import violation.

## Protocol

JSON lines over the relay, one request and one reply at a time, each request
carrying the service token:

| Request | Reply | Policy call |
|---|---|---|
| `{"cmd": "reset", "state", "info"}` | `{"ok": true}` | `reset(state, info)` |
| `{"cmd": "act", "state"}` | `{"action": [...]}` | `get_action(state)` |
| `{"cmd": "update", "state", "reward", "done", "info"}` | `{"ok": true}` | `update(...)` if defined |

States use the environment service's wire encoding: `{"__ndarray__", "dtype"}` for
vectors and `{"__ocs__": payload}` for object-centric states. The client encodes
numpy vectors, real `relational_structs` states and `env_client` mirrors; a real
state's payload is byte-identical to `object_centric_codec`. The policy receives
exactly the evaluator's strict view: a vector cast to the observation dtype, or an
`env_client._ObjectCentricState`. Box states must be real numeric, have the declared
shape and stay finite in the observation dtype. Object-centric states may list only
declared types, each with its declared parent and feature names in order, parents
first; objects need unique names, a listed type and that type's number of finite
float32 features. `info` must decode as tagged JSON and `reward` must be finite. Spaces and the policy instance are built as in
`evaluation_worker.FrozenPolicy`, with the same replicate seed. `act` and `update`
before `reset` are protocol errors. Actions must have the action space's shape and
are returned as lists; the client returns an array of the action dtype.

## Process model and deadlines

The serving process never runs policy code. A connection's first request with
the right token forks one worker, in its own process group, that loads the policy,
holds its state across calls, and keeps no file descriptor of other connections.
Connections therefore share no policy instance and no loader state. The serving
process authenticates and validates every request before forwarding it and passes
on only well-formed replies. Any failure while parsing a request or a worker reply
closes that connection alone.

At most 8 connections may wait to authenticate (10 seconds each) and at most 8
workers may run; beyond that the service answers `busy`.

Loading has a 60-second bound and each call a configurable bound
(`request_timeout`, 30 seconds by default). Deadlines are checked before any worker
reply is read or forwarded, so a late reply is a timeout. A missed deadline kills
the worker's process group and answers `policy_timeout`. A client disconnect, an
error and termination also kill the whole group, including processes the policy
started, unless the policy moved them to another group.

Sends to a client or worker time out after 1 second. A client that stops reading
loses its connection and worker; each blocked send can delay other connections by that amount.

## Public error codes

Every failure is one reply `{"error": code}` followed by closing the connection.
Replies never contain exception text, tracebacks, paths or source lines; those go
to `policy_service.log` on the host.

| Code | Meaning |
|---|---|
| `policy_error` | The policy raised or exited on load, reset, act or update, or returned a malformed action |
| `policy_timeout` | Loading or a call exceeded its deadline |
| `protocol_error` | Malformed request, wrong state encoding or shape, call before `reset`, or a second request while one is pending |
| `unauthorized` | Missing or wrong token, or no request within 10 seconds |
| `busy` | Too many connections waiting to authenticate, or too many workers |

`PolicyServiceError.code` carries one of these, or `connection_closed` when the
connection failed locally. A client-side timeout is reported as `policy_timeout`.

## Isolation

The container has no network, no capabilities, a read-only root and only these
mounts: the runtime subset and policy snapshot (read-only), the request file
(read-only) and the socket directory. It receives no simulator source, artifact,
credentials or environment service. Live tests on both backends check this from
inside the policy, and that the host refuses strict import violations.

As for the environment service, the host relay listens on all interfaces for
Docker (so other containers can reach it) and on loopback for Apptainer; the token
guards it.

Call timing and chosen-state queries reveal behaviour by design: the service
hides code, not behaviour.

## Trust model

Policy workers and the supervisor share one container and user. This prototype
assumes the frozen policy does not deliberately expose its own source or impersonate
the service. The editor's access modes do not add a security boundary between them.

## Out of scope

Source-authored (whitebox) policies, the editor agent's sandbox, any coordinator
loop, and output queues: sends block for up to the 1-second send timeout. The fixed
relay drops a connection after 120 seconds without traffic in either direction.

## Verification

`tests/runtime/test_policy_service.py` runs the worker as a local process on a Unix
socket behind the production relay, with only the shipped files importable.
`tests/runtime/test_policy_service_containers.py` runs on both Docker and Apptainer
whenever the evaluation images exist; it needs no environment artifact.
`tests/runtime/test_evaluation_runtime_files.py` checks that the shipped subset is
import-closed.

The send-timeout regression fills a small socket buffer and forwards ordinary
three-element actions. It checks connection-local cleanup without requiring huge
action spaces. See `test_send_timeout_closes_only_the_affected_connection`.
