# Environment editor workspace

An editor is an `AgentSession` with `output_filename=""`. It submits a Git commit,
just as a policy agent does, but edits the simulator under `source/`.

`prepare_editor_workspace(view, session.workspace)` copies a verified, filtered
`WhiteboxSource` into a fresh workspace. The regular launcher initializes its Git
repository. Upstream history, demos and oracle code are absent. Call preparation
once; resumed turns keep their edits and native CLI conversation.

Set `SandboxConfig.environment_editor = EditorBinding(view, policy_binding)`.
Both Docker and Apptainer use the environment artifact's image and dependencies:

| Inside the editor | Access |
| --- | --- |
| `/sandbox/source/` | Mutable simulator source, first on `PYTHONPATH` |
| `/robocode/src/` | Filtered runtime, read-only |
| `policy_service.json` | Untracked client connection metadata, refreshed each turn |
| `/frozen-policy/` | Read-only policy snapshot, only when `policy_source` is supplied |

For policy-whitebox, pass the policy service's frozen snapshot directory as
`policy_source`. Its hash must match the running policy. Policy-blackbox omits this
mount. Neither mode uses the editor's process to execute policy code:

```python
import json
from pathlib import Path
from robocode.utils.policy_client import PolicyClient

with PolicyClient(json.loads(Path("policy_service.json").read_text())) as policy:
    state, info = env.reset(seed=42)  # env uses the edited local simulator
    policy.reset(state, info)
    action = policy.act(state)
    state, reward, terminated, truncated, info = env.step(action)
    policy.update(state, reward, terminated or truncated, info)
```

The coordinator supplies the environment construction config and editor objective
in its prompt. For the first exchange, ask it to keep the task solvable; reward
and success code remain editable. Turn receipts record the starting environment,
source view, frozen policy, access mode and submitted commit.

Docker uses the existing firewall and host gateway. Apptainer uses a dedicated
policy relay inside its isolated network, alongside the model broker. Both skip
dependency reinstallation and retain the image's pinned environment.

This checkpoint provides editor turns, not candidate acceptance. Exporting and
validating a new environment revision, retaining rejected edits for repair, and
waking the policy agent belong to the next coordinator checkpoint.

## Verification

Prepare separate fixtures beneath an output directory:

```sh
python integration_tests/prepare_runtime_checkpoint.py outputs/editor-tests/assets/motion
python integration_tests/prepare_runtime_checkpoint.py outputs/editor-tests/assets/variable --environment-config experiments/conf/environment/obstruction2d_generalized.yaml
ROBOCODE_EDITOR_TEST_ARTIFACTS=$PWD/outputs/editor-tests/assets pytest tests/runtime/test_editor_containers.py
```

The 16 cases cover Motion2D / variable-count Obstruction2D × Docker / Apptainer ×
Claude / Codex adapters × policy-blackbox / policy-whitebox. Scripted CLIs edit
transition code twice, render the environment, and submit commits across recreated
containers. Obstruction2D sweeps counts 0–4. The second turn receives a changed
policy in a newly started service; tests check its behavior and revision metadata.
No models are called; fixture-reported costs only exercise accounting.

`tests/runtime/test_editor.py` checks connection refresh preserves edits and that
mismatched frozen inputs or conflicting source-access settings fail before launch.
