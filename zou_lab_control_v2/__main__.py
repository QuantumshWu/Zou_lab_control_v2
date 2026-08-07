"""Run one of this product's windows: ``python -m zou_lab_control_v2 <app>``.

The launchers go through here rather than straight at ``zlc_workbench.apps``
for the same reason a notebook does: importing this package first is what makes
THIS checkout the code that runs.  A launcher that skipped it would open the
window from whatever happens to be installed, which is the failure with no
symptom -- the change you just made simply does not appear.
"""

from __future__ import annotations

import sys


APPS = ("pulse_editor", "task_console", "figure_viewer", "device_manager")
#: Not a window: it OPENS one, photographs it and exits.  It goes through this
#: entry for the same reason the windows do -- a screenshot taken of whatever
#: is installed elsewhere is a picture of code nobody is editing, which is the
#: most convincing way there is to conclude a fix did not work.
TOOLS = {"capture": "zlc_workbench.tools.capture_acceptance"}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        names = "|".join(APPS + tuple(TOOLS))
        print(f"usage: python -m zou_lab_control_v2 {{{names}}} [options]")
        return 0 if arguments else 2
    name, rest = arguments[0], arguments[1:]
    if name in TOOLS:
        module = __import__(TOOLS[name], fromlist=["main"])
        return int(module.main(rest))
    if name not in APPS:
        print(f"unknown window {name!r}; expected one of {', '.join(APPS + tuple(TOOLS))}")
        return 2
    module = __import__(f"zlc_workbench.apps.{name}", fromlist=["main"])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
