# Policy/environment exchange

`experiments/run_exchange.py` runs strict-blackbox policy synthesis, a source-editing
adversary, and resumed synthesis. Each actor has its own persistent `AgentSession`.
The editor sees the frozen policy through its service, with optional read-only code.

```text
E0 -> policy P0 -> evaluate P0/E0
                      |
                 editor submits commit
                      |
          export source/ -> container validation
                      |
             reject ------ accept E1
               |               |
         repair same       evaluate P0/E1
         editor session        |
                          resume policy -> P1
                               |
                        evaluate P1/E1, P1/E0
```

The editor's submitted commit and accepted environment are separate. A rejected
candidate leaves the previous environment active. Its source, receipt and validation
error remain available for a bounded repair turn. A normal CLI exit or best-effort
budget stop does not count as submission. Infrastructure errors are recorded and
raised. A stopped or failed exchange retains its files for inspection; the CLI does
not automatically resume an interrupted coordinator process.

Validation instantiates the exported environment in its pinned container. It compares
action/observation spaces, count schedules, and reset/step observation layouts at every
declared count with E0. It checks import origins. This is a structural probe, not a
proof of solvability or preserved reward semantics. The editor is instructed to keep
the task solvable and may edit arbitrary simulator code within that constraint.

All evaluations use separate environment/evaluator containers. Every result identifies
both artifacts. `exchange.json` records turns, rejected and accepted revisions, and
scores. Detailed logs, receipts and renders remain under each operation's directory.
The policy receives only its environment API. Held-out evaluation requests and results
are not mounted into either actor. P1/E0 is a regression measurement; the policy does
not yet train against an archive of old environments.

## Run

Use a fresh output directory, prepared images and an exported E0 artifact. For the
variable-count Obstruction2D config:

```bash
PYTHONPATH=src:. python integration_tests/prepare_runtime_checkpoint.py \
  outputs/exchange-assets --backend docker \
  --environment-config experiments/conf/environment/obstruction2d_generalized.yaml
PYTHONPATH=src:. python experiments/run_exchange.py \
  outputs/exchange-assets/docker-artifact outputs/exchange-live \
  --model claude-opus-5-5 --budget-per-turn 2 --cycles 1 \
  --counts 0 1 2 3 4 --policy-access blackbox
```

One cycle and one editor attempt permits three $2 turns ($6 nominal model allowance).
`--editor-attempts` permits additional paid repair turns; it defaults to one.
`--cycles` limits exchanges. Seeds and counts are parallel lists. Use `--backend
apptainer` with the corresponding Apptainer artifact and local SIF files. The source
surface is the filtered simulator package; source-authored policies are not supported
by this pilot. Simulator assets are committed alongside code in the editor repo.

## Scripted validation

No model calls are made by these fixtures. They retain actual provider launchers,
stream parsers, session stores, services, Git exports and evaluators.

```bash
ROBOCODE_EXCHANGE_TEST_ARTIFACTS=$PWD/outputs/exchange-assets \
PYTHONPATH=src:. python -m pytest tests/runtime/test_exchange_containers.py -v
```

Prepare assets with `--backend all` for the full matrix: Docker/Apptainer,
Claude/Codex adapters and policy-blackbox/whitebox editors. Each case submits invalid
Python, repairs it in the same editor context, resumes P1, checks all five count
schemas, verifies image assets survived export, and compares revision IDs across the
four evaluations.
