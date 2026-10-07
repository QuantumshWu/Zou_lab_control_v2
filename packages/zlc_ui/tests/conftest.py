from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]
if os.environ.get("ZLC_TEST_INSTALLED") != "1" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Once, before any test module is imported: the in-process Qt tests run
# offscreen on Agg (every child gets both again in ``_run_qt``).  Said by one
# test module at import, a file run on its own -- the normal targeted run --
# built its QApplication on the real desktop: windows on the operator's
# screen, placement asserted against the real monitor.  ``setdefault`` leaves
# an explicitly chosen platform alone.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


#: Every child starts here.  Without it the subprocess resolves the layers
#: through whatever the editable install points at -- on this machine,
#: sibling checkouts of the same package names -- so the suite silently
#: tested a DIFFERENT zlc_plot than the one beside it.  The product
#: bootstrap is what puts this checkout's layers on the path, and it is the
#: same one every launcher uses; the child then says which checkout it tested.
_BOOTSTRAP = (
    "import zou_lab_control\n"
    "import zlc_ui\n"
    "print('ROOT', zou_lab_control.__file__, 'UI', zlc_ui.__file__)\n"
)


def _run_qt(
    code: str,
    *,
    timeout: float = 60,
    extra_path: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run one offscreen Qt snippet in its own process and require it to pass."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            *(() if environment.get("ZLC_TEST_INSTALLED") == "1" else (str(REPO_ROOT), str(SRC))),
            *(str(path) for path in extra_path),
        )
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    # Fixed for the child, not inherited: a matplotlib backend chosen by
    # whatever the operator exported is how this harness produced access
    # violations at teardown that had nothing to do with the code under test.
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed


@pytest.fixture
def run_qt():
    return _run_qt
