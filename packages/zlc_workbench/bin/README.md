# Package launchers

The formal experiment entry is the repository-root `bin\experiment.bat`. It
opens Device Manager first and, after `Init devices`, opens TaskConsole and
Pulse UI in the same process over one shared session. The wrappers in this
package directory are developer/package entry points, not a second experiment
workflow.

Double-click the repository-root launcher. `Workspace.discover()` uses the
nearest directory that already contains `pulses/` or `apparatus.json`; when
none exists it uses the one configured default at `<checkout>\workspace` (or
`ZLC_WORKSPACE`). It never treats an arbitrary launch directory as a new
experiment.

| Launcher | Window | First run |
|---|---|---|
| `task_console.bat` | Developer wrapper for the same unified experiment app | `task_console.bat --template virtual` |
| `pulse_editor.bat` | Edit a pulse, preview it, fire it | `pulse_editor.bat --connect virtual` |
| `figure_viewer.bat` | Open a saved figure and read what it was | `figure_viewer.bat --path data\2026_08_05\run.npz` |

Every launcher takes `--check`, which builds everything and exits without
opening a window. That is the fastest way to find out whether a machine is set
up, and it is what the tests run.

## Connecting the pulse editor to a board

The board is driven by a server on the machine it is wired to. Start the
monorepo launcher, which bootstraps the same checkout before importing
`zlc_pulse`:

    bin\run_server.bat

It prints the exact endpoint to type into the editor's Connection panel — use
`127.0.0.1:18861` on the same computer, or the LAN address it shows under
`SERVER ADDRESS` from another. Then either:

* type the endpoint in the window, pick **Remote server**, press **Connect**; or
* start already connected: `pulse_editor.bat --connect remote:127.0.0.1:18861`

`--connect virtual` needs no server and no board: it is zlc_pulse's own host
driving a memory-backed register file, so the command sequence is the real one
with only the wire substituted. `Offline (edit only)` connects to nothing —
Run, Stop and Sync are disabled and say so rather than looking available.

## Looking at a window

There is one way to capture a window for visual acceptance, and it is not a
screenshot script:

    python -c "import zou_lab_control_v2; from zlc_workbench.tools.capture_acceptance import main; raise SystemExit(main())" --view pulse --connect virtual

It opens the same `create_window()` a human uses through
`zlc_ui.acceptance.capture_window`, which checks the window is exactly the
shared screen-fit size and writes a physical-DPR crop. It refuses the offscreen
backend on purpose: an offscreen canvas cannot reproduce the monitor the
picture is meant to inspect, and a wrong launcher fails loudly instead of
producing a plausible image at a size somebody typed.

## Environment

| Variable | Meaning |
|---|---|
| `ZLC_PY_CMD` | Interpreter to use (default: `python`) |
| `ZLC_NO_PAUSE` | Set to anything to skip the pause on failure (used by scripts) |

The launchers first import the repository bootstrap and print the modules they
resolved. They do not rely on a global/editable `zlc_*` installation, which
could silently test a sibling checkout instead of this product tree.
