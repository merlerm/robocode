# Frozen environment exports

`robocode.runtime.artifacts` exports a committed environment repository into a
fresh directory. It records the Git commit/tree, resolved configuration, explicit
assets, copied dependency files and a resolved runtime image SHA256. A content
identity covers that manifest and every exported file, including executable bits.
Dirty, staged, untracked and ignored source files are excluded. Explicit assets
can include ignored simulator data, but cannot overwrite committed source.

```python
artifact = export_environment(
    kinder_repo, "HEAD", output_dir,
    package="kinder", config=resolved_config,
    runtime_image=image_sha256,
    dependencies={"uv.lock": lockfile},
)
```

`load_environment_artifact(path, expected_id=...)` checks identity and file
contents. In a fresh worker, prepend `artifact.import_root` to `PYTHONPATH`, disable
bytecode writes, and mount the export read-only. After environment imports,
`artifact.verify_imports()` checks loaded modules and namespace-package paths
against that export. Recheck after lazy imports. These checks catch accidentally
installed/cached copies; they are not a sandbox for malicious Python.

Exports support ordinary files/directories; Git submodule parents and symlinks
are rejected. Export the editable package repository itself. Keep output outside
that repository. Failures propagate and may leave an incomplete output directory
for inspection. A missing final manifest means it is not an accepted artifact.

This is the first slice of checkpoint 1C. It does not change the experiment runner,
launch containers, enforce the declared image/dependencies, validate task semantics
or implement submissions. Launchers must enforce the runtime identity and mount
boundaries in the next slice. No parameter-only editing interface is imposed:
committed environment programs, including reactive disturbance code, use the same
export contract.

## Verification, 2026-11-10

19 tests cover committed-only exports, reproducible identities, E0 -> E1 -> E0
execution in fresh interpreters, input changes, tampering, source/asset boundaries,
and incorrect or mixed import origins, including namespace packages. All 36
existing environment-service tests also pass. Scoped Black/isort/docformatter,
mypy and pylint checks pass; pylint emits the repository's existing legacy
configuration warnings.

A real isolated Apptainer probe exported KinDER commit
`5b2dbac482142b879b48449b3fa2af63f641aed3`, mounted the export read-only, instantiated
Motion2D from its manifest, and passed reset/step/render. All 14 loaded KinDER
modules resolved inside `/artifact/source/src`; the original KinDER source path
was absent. Manifest/file verification passed again after execution. No model calls.

Artifact: `sha256:e8d3bc5f77c12c3960ddfb9aacc8569c8861aa7dde1f22f881d904542dd491d7`.
Local evidence: `outputs/artifact-verification-2026-11-10/` contains the manifest,
probe/runner, command, log and imported-module paths. The large exported tree and
test scratch are removed after verification; `verify.py` recreates the probe from
the pinned checkout and retained local SIF. This tests export consumption, not the
future containerized environment service or evaluation pipeline.

## Real revision round trip

[Manual E0/E1/E0 verification](round_trip/README.md) now passes through the
service API in fresh Apptainer containers: same-policy mean steps are
39.3 -> 108.8 -> 39.3. Restored E0 traces and saved renders match byte-for-byte.
Normal-runner integration and separate evaluation isolation remain pending.
