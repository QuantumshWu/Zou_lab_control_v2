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


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(f"usage: python -m zou_lab_control_v2 {{{'|'.join(APPS)}}} [options]")
        return 0 if arguments else 2
    name, rest = arguments[0], arguments[1:]
    if name not in APPS:
        print(f"unknown window {name!r}; expected one of {', '.join(APPS)}")
        return 2
    module = __import__(f"zlc_workbench.apps.{name}", fromlist=["main"])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
