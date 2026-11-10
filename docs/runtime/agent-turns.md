# Persistent agent turns

`robocode.runtime.agent_session.AgentSession` is an opt-in coordinator API. It
reuses the existing Docker, Apptainer and local launchers. The normal experiment
runner still performs one generation; adversarial alternation is not wired in yet.

Each session owns `workspace/`, its native conversation store and a Git repository.
`run_turn()` runs one allowance and returns after the launcher removes the agent
container. The session is then idle. Recreate `AgentSession` and the backend object
from the same directory to resume later. Claude uses `--resume <session_id>`;
Codex uses `exec resume <session_id>`. IDs come from the actual CLI stream, including
retries, rather than selecting the latest conversation. Mid-tool pause is deferred.

```python
from dataclasses import replace
from robocode.runtime.agent_session import AgentSession

session = AgentSession(run_dir / "policy")
config = replace(policy_config, sandbox_dir=session.workspace, max_budget_usd=10)
first = session.run_turn("policy-0", config, backend, environment_id=e0.artifact_id)
# The coordinator evaluates the submission and lets the environment editor work.
# Before resuming, it installs the new permitted binding/source view and metadata.
resumed = AgentSession(run_dir / "policy")
second = resumed.run_turn("policy-1", updated_config, new_backend,
                          environment_id=e1.artifact_id)
```

`updated_config` must use the same workspace, provider/model and source-access
contract. Its budget is a fresh allowance. `init_files` only initializes the first
turn; it never overwrites edits on continuation or retry. The caller installs fresh
controller-owned environment transport metadata before each turn. The session
records the supplied environment ID; environment binding/image verification remains
the service/source-view layer's responsibility. Never upgrade a blackbox session to
whitebox or send it editor diffs/private logs.

## Submission and scoring

The turn prompt asks the agent to commit and write `.agent_submission.json`:

```json
{"turn_id": "policy-0", "commit": "<full HEAD SHA>", "retained": false}
```

A submission must name the current turn and HEAD, have no uncommitted non-ignored
files, and contain `output_filename` when one is required. Submitting the previous
handoff commit unchanged requires `retained: true`. A stale file or normal CLI exit
alone produces `outcome: unsubmitted`; it cannot advance the experiment as a new
successful generation. Claude and Codex use this same contract. Codex's older
policy-confidence marker is not used for coordinated turns.

Accepted handoffs get a committed-tree export under `turns/<id>/artifact/`, with a
content manifest and Git commit in the receipt. Generated instructions, native
sessions and environment transport files are excluded from new coordinated Git
workspaces. Policy helpers/data must be committed too. Policy turns require
`output_filename="approach.py"`; editor turns can require another file or leave
this empty. The session never imports the submitted code on the host. A structurally
valid submission still needs isolated evaluation before calling it a usable policy
or accepting an environment edit.

Budget/turn-limit stops preserve saved work using the existing auto-commit behavior,
but return `outcome: best_effort`, not `submitted`. The evaluator can inspect that
candidate as before. Stop reason and submission outcome are recorded independently.
An empty/missing required policy fails rather than silently selecting stale output.

## Budget and durable records

Each attempt is recorded separately, including retry/compaction costs. Retries use
only the remainder of the current turn; the next turn gets its own allowance.
Session spend is the sum of actual reported attempt costs. A hard cutoff can
overshoot by an in-flight request; the receipt records actual spend.

`session.json` holds provider/access identity, exact native session ID, completed
turn IDs, last selected commit and cumulative cost. Each `turns/<id>/turn.json`
holds allowance, environment ID, attempts, stop reason, outcome and artifact
identity. Each turn also retains its own slice of `stream.jsonl`. Coordinator state
is outside the agent-mounted workspace. Independent sessions need independent
root directories; there is no fixed number of agent slots.

A file lock rejects overlapping calls on the same session. Reused turn IDs fail.
A coordinator crash, invalid handoff, missing native ID or unknown spend leaves a
pending record and refuses automatic continuation. Inspect/reconcile its ledger
before further work; automatic crash recovery is deliberately not implemented.

## Verification

Tests exercise real Git submissions, retained/stale commits, retries, missing
context/accounting, immutable exports, independent sessions, overlapping calls and
best-effort cutoff. Container tests recreate the coordinator, backend and container
between turns for Claude/Codex on Docker/Apptainer. They retain production provider
setup, parsing, accounting, mounts and cleanup, substituting a deterministic CLI
fixture for inference. They verify exact-ID resume arguments, native-store and
workspace persistence, fresh allowances, cumulative costs and separate turn logs.
These are not live model-continuation results. Live Claude, live Codex continuation
and cluster execution remain separate acceptance checks.

```sh
export PYTHONPATH=src:.
export ROBOCODE_STRICT_IMAGE=robocode-strict-blackbox:multi-agent-runtime
export ROBOCODE_TEST_AGENT_TURNS=1
python -m pytest tests/runtime/test_agent_session.py \
  tests/runtime/test_agent_turn_containers.py -q
```

The container fixtures require the fork's strict Docker image and
`robocode-strict-blackbox.sif`. They make no model requests. Evidence is retained
under `outputs/agent-turns-2026-11-10/`.

Verified on 2026-11-10: **223 distinct passing cases**, no unresolved failures or
skips after the installed Apptainer budget test was enabled. The combined ledger
is `outputs/agent-turns-2026-11-10/verification-summary.json`. Scoped mypy passes;
pylint source score is 10/10 with existing obsolete-config diagnostics.
