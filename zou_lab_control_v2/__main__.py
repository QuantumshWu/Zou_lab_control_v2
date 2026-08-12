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
TOOLS = {
    "capture": "zlc_workbench.tools.capture_acceptance",
    #: The FPGA geometry/capacity CLI, which the build launchers run.  Through
    #: here for the same reason: "python -m zlc_pulse.fpga" with the layer's own
    #: folder on the path does not find the package at all (the source is under
    #: src/), so it only worked where a standalone repository happened to be
    #: pip-installed -- and then it derived the board geometry from THAT tree.
    "fpga": "zlc_pulse.fpga",
    #: The pulse server, for the same reason: run_server.bat starts it on
    #: the machine wired to the board, and it must be THIS tree's server.
    "pulse_server": "zlc_pulse.remote",
    #: What update.bat runs to prove the pull still imports.  It has to be
    #: asked through the entry or it proves it about the wrong tree.
    "check": "zlc_workbench.tools.check_environment",
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        names = "|".join(APPS + tuple(TOOLS))
        print(f"usage: python -m zou_lab_control_v2 {{{names}}} [options]")
        return 0 if arguments else 2
    name, rest = arguments[0], arguments[1:]
    if name in TOOLS:
        module = __import__(TOOLS[name], fromlist=["main", "_main"])
        entry = getattr(module, "main", None) or module._main
        # Some tools take the remaining arguments and some take none; asking
        # the function rather than assuming keeps one dispatch for both.
        import inspect

        takes_argv = bool(inspect.signature(entry).parameters)
        return int(entry(rest) if takes_argv else entry())
    if name not in APPS:
        print(f"unknown window {name!r}; expected one of {', '.join(APPS + tuple(TOOLS))}")
        return 2
    module = __import__(f"zlc_workbench.apps.{name}", fromlist=["main"])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
