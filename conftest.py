"""Run the same tests against source paths or one fresh installed product."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parent
INSTALLED = os.environ.get("ZLC_TEST_INSTALLED") == "1"

if INSTALLED:
    # Pytest loads this file from the checkout, but product imports must come
    # only from the fresh environment's site-packages.
    for item in tuple(sys.path):
        try:
            if Path(item or ".").resolve() == REPO_ROOT:
                sys.path.remove(item)
        except OSError:
            continue
else:
    sys.path.insert(0, str(REPO_ROOT))

# Deliberately the first product import in the verification process.
import zou_lab_control  # noqa: E402

if INSTALLED and REPO_ROOT in Path(zou_lab_control.__file__).resolve().parents:
    raise RuntimeError("installed evidence imported zou_lab_control from the checkout")
if INSTALLED:
    # Windows spawn must be able to unpickle pytest's importlib-mode module
    # name (packages.<layer>.tests...). Append, never prepend: site-packages
    # remains the product authority and the checkout is visible only as a test
    # module namespace.
    sys.path.append(str(REPO_ROOT))

manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
source_roots = tuple(
    (REPO_ROOT / item).resolve()
    for item in manifest["tool"]["setuptools"]["packages"]["find"]["where"]
    if item != "."
)
if INSTALLED:
    os.environ["PYTHONPATH"] = ""
else:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), *(str(source) for source in source_roots))
    )
    for source in reversed(source_roots):
        sys.path.insert(0, str(source))

# A few suites intentionally share test doubles by bare module name.  Their
# test directories are not product code and are safe in both modes.
for source in source_roots:
    tests = source.parent / "tests"
    if tests.is_dir():
        sys.path.insert(0, str(tests))
