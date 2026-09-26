"""The platform every workbench test runs on, whichever file is run."""

from __future__ import annotations

import os


# Once, before any test module is imported: the in-process Qt tests run
# offscreen and every child inherits one matplotlib backend.  Said by each
# test module at import, a file that did not say it -- test_task_console_app
# -- run on its own built its QApplication on the real desktop.
# ``setdefault`` leaves an explicitly chosen platform alone.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
