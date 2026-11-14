# Source-access exchanges

`experiments/run_exchange.py --policy-whitebox` gives the policy agent the current
accepted environment source. `--policy-access whitebox` independently gives the
editor the frozen policy code; the policy service runs in either editor mode.

A policy submission freezes its training source view. Evaluation and the policy
service use that same view for imports and native observation types, even when
running against a different environment revision. Metadata separates
`policy_source_environment_id` and `whitebox_source_id` from the evaluated
`environment_artifact_id`. Evaluation source is mounted only in its environment
service, so source access does not expose held-out simulator implementations.

During development, `robocode.runtime.whitebox.make_training_environment()` connects
to the training server with native state types. Local source instantiation covers
only the current revision. The research runner's `--mixture` option routes server
resets uniformly across accepted revisions and retains the native agent session
and sandbox between turns. Submitted programs must act on their received states.

The research runner uses `--editor-policy-access whitebox` for editor visibility.
It also accepts `--require-action`: structural validation is
followed by full-horizon zero-action episodes on two seeds at every configured
count. Any sampled success rejects the candidate for repair. This is a finite
screen, not a proof over the environment's task distribution.
