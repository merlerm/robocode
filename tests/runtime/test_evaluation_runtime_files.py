"""The blackbox evaluator's source subset must import nothing outside itself."""

import ast
from pathlib import Path

from robocode.runtime.container_evaluation import _RUNTIME_FILES

_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "robocode"


def _module_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Imports that run when the module loads, including under try/if/with."""
    found: list[ast.Import | ast.ImportFrom] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            pending.extend(node.orelse)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody"):
                pending.extend(getattr(node, field, []))
            for handler in getattr(node, "handlers", []):
                pending.extend(handler.body)
    return found


def test_evaluator_runtime_files_are_import_closed() -> None:
    """Every robocode import of a shipped file resolves to another shipped file."""
    shipped = set(_RUNTIME_FILES)
    missing = []
    for name in _RUNTIME_FILES:
        tree = ast.parse((_PACKAGE / name).read_text())
        for node in _module_level_imports(tree):
            if isinstance(node, ast.Import):
                targets = [[alias.name] for alias in node.names]
            else:
                assert node.level == 0, f"{name}: relative import"
                base = node.module or ""
                # "from a import b" needs a.py, or a/b.py when b is a module.
                targets = [[base, f"{base}.{alias.name}"] for alias in node.names]
            for candidates in targets:
                if not candidates[0].startswith("robocode"):
                    continue
                files = {
                    c.removeprefix("robocode").lstrip(".").replace(".", "/") + ".py"
                    for c in candidates
                }
                if not files & shipped:
                    missing.append(f"{name} imports {candidates[0]}")
    assert not missing, missing
