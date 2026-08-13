"""Open the pulse editor.

    python -m zlc_workbench.apps.pulse_editor
    python -m zlc_workbench.apps.pulse_editor --pulse untitled
    python -m zlc_workbench.apps.pulse_editor --connect remote:127.0.0.1:18861

The window opens whether or not there is a pulse to show; with none it says how
to get one.  Product pulses are ``zlc.pulse.v1`` JSON documents in an
experiment's ``pulses/`` directory, read through the same loader the session
uses so the window cannot show a pulse assembled differently from the one that
will fire.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a pulse in the editor.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory holding pulses/ (default: found at or above this one)",
    )
    parser.add_argument(
        "--pulse",
        default=None,
        help="pulse to open at startup (default: open empty)",
    )
    parser.add_argument(
        "--connect",
        default=None,
        metavar="MODE[:ENDPOINT]",
        help=(
            "connect at startup: 'virtual', or 'remote:HOST:PORT' for a server "
            "started by zlc_pulse/fpga/run_server.bat (default: stay offline)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build everything and exit without opening a window",
    )
    return parser


def dial(mode: str, endpoint: str):
    """Open the sequencer the operator asked the window for.

    ``remote`` is the server started by ``zlc_pulse/fpga/run_server.bat`` on the
    machine wired to the board.  ``virtual`` is zlc_pulse's own host driving a
    memory-backed register file -- the real command sequence with the wire
    substituted, which is what makes it worth rehearsing on.
    """

    from zlc_pulse import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        MemoryRegisterTransport,
        PulseStreamer,
        connect,
        load_streamer_config,
        pulse_target_from_xdc,
    )
    from ..pulse_editor import CONNECTION_REMOTE, CONNECTION_VIRTUAL

    if mode == CONNECTION_REMOTE:
        text = str(endpoint or default_endpoint())
        host, _, port = text.partition(":")
        streamer = connect(host or DEFAULT_HOST, int(port or DEFAULT_PORT))
        # connect() only constructs a client; open() is the first call that
        # actually reaches the server.  Without it a window with nothing
        # listening at that address reports "connected" and only fails later,
        # when the operator presses a button and blames the pulse.
        streamer.open()
        return streamer
    if mode == CONNECTION_VIRTUAL:
        config = load_streamer_config()
        if config["source"] is None:
            raise RuntimeError(
                "no streamer config was found, so the board geometry is unknown"
            )
        geometry = config["params"]
        streamer = PulseStreamer(
            MemoryRegisterTransport(geom=geometry, auto_done=True),
            geometry,
            config["clock_hz"],
            target=pulse_target_from_xdc(config_path=config["source"]),
        )
        streamer.open()
        return streamer
    raise ValueError(f"unknown connection mode {mode!r}")


def default_endpoint() -> str:
    """Where a board is usually reached, as one line for a window to offer.

    From zlc_pulse, which is the package that decides where a pulse server
    listens.  Nothing else in this project writes that address down.
    """

    from zlc_pulse import DEFAULT_HOST, DEFAULT_PORT

    return f"{DEFAULT_HOST}:{DEFAULT_PORT}"


def build(
    view: object,
    state: object = None,
    *,
    path: str = "",
    pulses_directory: str = "",
    sequencer: object | None = None,
    device_use: object | None = None,
    allow_dial: bool = True,
    connection_label: str | None = None,
) -> object:
    """Wire one editor window, with or without a pulse in it."""

    import zlc_plot as plot

    from ..pulse_editor import PulseEditorPresenter

    def make_preview(timeline, *, size: str = "2x2"):
        # The host, and only the host.  It is what a save writes through, what
        # the next edit updates rather than replaces, and -- since it can be
        # asked for its own widget and its own size -- the whole of what the
        # window needs.  This used to hand back a QWidget it had constructed
        # here, which is a composition root assembling a UI.
        host = plot.RasterPlotHost.from_plot(
            timeline, plot.PulseTimelinePlot(), size=str(size)
        )
        # No selector is placed here.  A selection is something the operator
        # drags onto the picture; putting one there at build time gave every
        # preview a full-width band it had not asked for and could not remove,
        # because the Selectors switch says whether one may be DRAWN, not
        # whether one exists.
        # The preview page lays content out at its natural size rather than
        # stretching it, so nothing may be mounted before a front exists: a
        # raster host has no size until it has painted one.
        host.wait_for_front(5.0)
        return host

    def update_preview(host, timeline, *, size: str = "2x2"):
        """Give the standing preview new data, and say how big it now is.

        The canvas widget was sized once, when it was first mounted, so a
        pulse that grew -- 2 rows becoming 22 the moment Show off rows is
        switched on -- was drawn in full into a widget still shaped for the
        old one, and the operator saw the top three channels and blank space.
        The host knows its new size; this hands it back.
        """

        planned = host.set_size(str(size)).result(timeout=5.0)
        host.update_data(timeline).result(timeout=5.0)
        plan = getattr(planned, "value", None)
        return tuple(getattr(plan, "logical_size", ()) or ()) or None

    return PulseEditorPresenter(
        view,
        state,
        make_preview=make_preview,
        update_preview=update_preview,
        sequencer=sequencer,
        device_use=device_use,
        dial=dial if allow_dial else None,
        pulses_directory=pulses_directory,
        path=path,
        default_endpoint=default_endpoint() if sequencer is None else "",
        connection_label=(
            "Experiment session" if connection_label is None else str(connection_label)
        ),
    )


def load_state(workspace: Path, name: str):
    """The complete editor state stored in one named JSON pulse document.

    Read through the same loader the session uses, so the window cannot show a
    pulse assembled differently from the one that would be fired.
    """

    from ..session import Workspace, read_pulse

    space = Workspace(workspace)
    path = space.pulse(name)
    if not path.is_file():
        raise FileNotFoundError(f"no pulse named {name!r} in {space.pulses}")
    return read_pulse(path), str(path)


def resolve(workspace=None, pulse=None):
    """The workspace and the pulse a caller asked for, or a plain refusal.

    Shared by the window entry and the command line so both answer the same
    way; a window that silently opens empty where the CLI would have refused
    is two different products.
    """

    from ..session import Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()
    if not pulse:
        return space, None, ""
    state, path = load_state(space.root, pulse)
    return space, state, path


def create_window(
    *,
    workspace=None,
    pulse: str | None = None,
    connect: str | None = None,
    window_ratio: float | None = None,
):
    """Open the editor the way a human does, and return its window.

    The non-blocking public entry.  It is what the launchers run, what a
    notebook cell calls, and what zlc_ui's acceptance capture opens -- one
    entry means the window under inspection is the window that ships.
    """

    from zlc_ui import open_pulse_editor

    space, state, path = resolve(workspace, pulse)
    # One call, one handle.  This layer never names a widget class: what comes
    # back has signals to hear and methods to call, and nothing to assemble.
    window = open_pulse_editor(
        title="PulseGUI@Zou lab", window_ratio=window_ratio
    )
    window.presenter = build(
        window,
        state,
        path=path,
        pulses_directory=str(space.pulses) if space is not None else "",
    )
    # The window owns the presenter for as long as it is open; closing it is
    # what releases the plot worker and any board it dialled.
    window.closed.connect(window.presenter.close)
    if connect:
        mode, _, endpoint = str(connect).partition(":")
        window.presenter.connect_to(mode, endpoint)
    return window


def create_bound_window(
    *,
    workspace: object,
    sequence: object,
    sequencer: object,
    device_use: object,
    path: str = "",
    window_ratio: float | None = None,
):
    """Open PulseGUI over a sequencer borrowed from an ExperimentSession.

    This entry has no dial path.  Connection controls therefore cannot replace
    the experiment's sequencer with a second client, and closing the editor
    retires only its presenter/preview -- the session remains the device owner.
    """

    from zlc_ui import open_pulse_editor

    from ..pulse_state import PulseEditorState
    from ..session import Workspace

    space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    window = open_pulse_editor(
        title="PulseGUI@Zou lab", window_ratio=window_ratio
    )
    try:
        window.presenter = build(
            window,
            PulseEditorState(sequence=sequence),
            path=str(path),
            pulses_directory=str(space.pulses),
            sequencer=sequencer,
            device_use=device_use,
            allow_dial=False,
            connection_label="Experiment session",
        )
    except BaseException:
        window.close()
        raise
    from ..board import attach_qt

    refresh_timer = attach_qt(window.presenter.refresh_run_state, interval_ms=100)
    window._device_control_refresh_timer = refresh_timer

    def close_bound_control() -> bool:
        try:
            window.presenter.close()
        except BaseException as error:
            window.show_warning(str(error))
            return False
        refresh_timer.stop()
        return True

    window.set_close_guard(close_bound_control)
    return window


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app, open_pulse_editor

    application = ensure_qt_app([])
    try:
        space, state, path = resolve(arguments.workspace, arguments.pulse)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"workspace: {space.root if space is not None else '(none found)'}")

    if arguments.check:
        # The same composition, through the same entry: a smoke test, not
        # acceptance.  It used to build the body class directly, which is the
        # one shape this layer must not know.
        view = open_pulse_editor(window_ratio=0.4)
        presenter = build(
            view,
            state,
            path=path,
            pulses_directory=str(space.pulses) if space is not None else "",
        )
        try:
            if arguments.connect:
                mode, _, endpoint = str(arguments.connect).partition(":")
                if not presenter.connect_to(mode, endpoint):
                    print(
                        f"error: could not connect via {arguments.connect}",
                        file=sys.stderr,
                    )
                    return 3
                print(f"connected: {mode} {endpoint}".rstrip())
            if state is None or state.sequence is None:
                print("editor ready: no pulse open")
            else:
                sequence = state.sequence
                print(
                    f"pulse ready: {sequence.name!r}, "
                    f"{len(sequence.periods)} period(s), "
                    f"{len(sequence.target.ports)} port(s)"
                )
            return 0
        finally:
            presenter.close()

    window = create_window(
        workspace=arguments.workspace,
        pulse=arguments.pulse,
        connect=arguments.connect,
    )
    del window
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
