"""Every package name must resolve to the repo that owns it.

Three separate incidents in this project came from an import that succeeded
while the wrong code ran: a pre-split monolith installed under the same
top-level names, a package that was never installed resolving to an empty
namespace because the working directory sat beside it, and an editable install
pointing at a copy that had been deleted.  None of them raised; all of them
wasted a day.

This runs as an ordinary test so the condition is checked continuously, not
only when someone remembers to run the tool.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from zlc_workbench.tools.check_environment import OWNED, RETIRED, check


def test_every_package_resolves_to_its_own_repo_and_no_retired_name_survives() -> None:
    problems = check()
    assert problems == [], "\n".join(problems)


def test_the_check_covers_every_package_in_the_workspace() -> None:
    """A guard that scans nothing passes for the wrong reason."""

    assert len(OWNED) >= 8
    assert "zlc_workbench" in OWNED
    assert set(RETIRED) == {"zlc_storage", "zlc_frontend", "zlc_neutral_atom"}


def test_the_tool_is_runnable_from_outside_the_workspace() -> None:
    """The namespace-package trap only shows up from another directory.

    Running it from the workspace root would let a directory that merely sits
    beside an uninstalled package stand in for a real install -- which is
    exactly how zlc_ui appeared to be installed when it was not.
    """

    with tempfile.TemporaryDirectory() as elsewhere:
        root = Path(__file__).resolve().parents[3]
        environment = dict(
            os.environ,
            PYTHONPATH=(
                str(root)
                + os.pathsep
                + os.environ.get("PYTHONPATH", "")
            ),
        )
        script = (
            "import zou_lab_control_v2\n"
            "from zlc_workbench.tools import check_environment as tested_module\n"
            "print(tested_module.__file__)\n"
            "from zou_lab_control_v2 import __main__ as product_entry\n"
            "raise SystemExit(product_entry.main(['check']))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=elsewhere,
            capture_output=True,
            text=True,
            env=environment,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "resolve to their own repo" in completed.stdout
