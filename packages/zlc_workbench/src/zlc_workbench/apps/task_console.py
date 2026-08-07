"""Start the task console.

    python -m zlc_workbench.apps.task_console --workspace D:/experiment

This is the composition root at its most literal: it builds the session, the
views, the presenter and the display beat, connects them, and gets out of the
way.  Every decision it appears to make is really a default being passed
through -- which apparatus, which pulse, which signal to show first.

The window drives the same ExperimentSession a notebook drives.  If a button
ever needs something the notebook cannot do, that capability is missing from the
session, not from the window.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the neutral-atom task console.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "directory holding pulses/, data/ and apparatus.json "
            "(default: found at or above this one)"
        ),
    )
    parser.add_argument(
        "--template",
        default=None,
        help="start from a named apparatus template instead of apparatus.json (e.g. virtual)",
    )
    parser.add_argument(
        "--pulse",
        default="calibration",
        help="pulse to load at startup (default: calibration)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build everything and exit without opening a window (a startup smoke test)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=200,
        help="display beat in milliseconds (default: 200)",
    )
    return parser


def open_experiment(workspace=None, template=None, pulse="calibration"):
    """The session behind a console window, and the first thing it will show.

    Kept apart from the window so the same assembly serves the smoke check, a
    notebook, and the window entry -- three ways in, one experiment.
    """

    from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode

    from ..session import ExperimentSession, Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()
    if space is None:
        raise FileNotFoundError(
            "no experiment directory here.  A console needs one -- it is where "
            "pulses/, data/ and apparatus.json live.  None of "
            f"{', '.join(Workspace.MARKERS)} was found at or above {Path.cwd()}; "
            "pass --workspace, or start the launcher from your experiment folder."
        )

    session = ExperimentSession.open(space.root, template=template)
    loaded = session.load_pulse(pulse)
    node = CameraMeasurementNode(
        camera=session.camera,
        signal_plane=session.signal_plane,
        producer="camera",
        repeat=1,
        frames_per_cycle=int(loaded["camera_windows"]),
    )
    session.nodes = [node]
    # Start the camera monitoring so the first panel has something to show.
    monitor = node.monitor(buffer_frames=4)
    session.fire(shots=1)
    monitor.poll()
    return space, session, node, monitor, loaded


def build_console(session, node, *, interval_ms, release_bootstrap=None, window_ratio=None):
    """One console presenter over one session, with the view it drives."""

    import zlc_plot as plot

    from zlc_ui import open_task_console

    from ..console import ConsolePresenter

    # One call, one handle: this layer never names a widget class.
    view = open_task_console(title="TaskConsole@Zou lab", window_ratio=window_ratio)

    def _kind_of(name: str):
        return next((item for item in plot.PlotKind if item.value == str(name)), None)

    def _spec_for(snapshot, kind: str = ""):
        """The spec this data admits, as the chosen kind or as its own shape."""

        return plot.fitting_spec(snapshot.block.schema, _kind_of(kind))

    def _panel_kinds():
        """Which kinds a panel may be, named the way the plotting package names them."""

        return plot.panel_kinds()

    def _make_host(initial, signal, kind=""):
        # The operator's kind when they chose one, the signal's own schema when
        # they did not.  Every panel used to be built as an image against two
        # named spatial axes, so Add Panel on anything that is not a camera
        # frame -- an ROI total, a fit centre, a scan curve -- could not draw.
        spec = _spec_for(initial, kind)
        if spec is None:
            raise ValueError(f"{signal!r} cannot be drawn as {kind or 'anything'}")
        return plot.RasterPlotHost.from_plot(
            initial, replace(spec, labels=replace(spec.labels, title=str(signal)))
        )

    presenter = ConsolePresenter(
        session,
        view,
        make_host=_make_host,
        panel_kinds=_panel_kinds,
        spec_for=_spec_for,
        # The window asks which signal; the presenter projects what exists and
        # the window renders the question.  Neither knows what any of it means.
        choose_signal=view.choose_signal,
        open_saved=lambda start: _open_saved_figure(view, start),
        edit_panel=lambda host, title: _edit_panel(view, host, title),
        release_bootstrap=release_bootstrap,
        # A logic node is what publishes a signal.  The question of which type
        # to add and the settings form are Qt; what a node IS, and what running
        # one means, are not, and stay out of this file.
        choose_logic=lambda rows: _choose_logic(view, rows),
        edit_logic=lambda descriptor, values: _edit_logic(view, descriptor, values),
        default_interval_ms=interval_ms,
    )

    # One panel is opened so the window is not empty on arrival; every other
    # panel comes from Add Panel, which offers whatever the plane is carrying
    # at the moment it is asked.
    signal = node.signal_key("frames")
    value = session.signal_plane.freeze().value(signal)
    if value is not None:
        presenter.add_panel(signal, value.snapshot, title="camera")
    return view, presenter



def _edit_panel(parent: object, host: object, title: str) -> object:
    """Open one panel's own plot controls.

    What a plot can be told to show, and what one semantic edit produces,
    both belong to zlc_plot -- which also owns the panel and now the window
    that holds it.  This only says which plot and what to call it.
    """

    from zlc_plot import edit_plot_display

    # The function crosses inward and the window parents it: this layer holds
    # neither the dialog nor the widget it would sit on.
    return parent.run_host_dialog(edit_plot_display, host, title=f"{title} display")


def _choose_logic(parent: object, rows) -> str | None:
    """Ask which kind of node to add.

    The same dialog the signal chooser uses, over the node catalog: a node type
    is picked the same way a signal is, and inventing a second picker would give
    an operator two ways to answer one kind of question.
    """

    return parent.choose_signal(
        tuple(
            (
                name,
                f"{name}  ({kind})",
                # What this bench can actually do with it, from the presenter
                # that tried.  "available" used to be written here for every
                # row by the one place with no way to check.
                "live" if not blocked else blocked,
                kind,
                f"publishes {publishes}",
            )
            for name, kind, publishes, blocked in rows
        ),
    )


def _edit_logic(parent: object, descriptor: object, values: object):
    """Show one node's declared settings, and return what was set.

    The form comes from the node's own authoring schema -- the same projection
    the apparatus editor uses -- so a node that gains a setting gains a box for
    it with no window change at all.  The dialog itself is zlc_ui's: a modal
    form over a spec is a control, not something a composition root assembles.
    """

    from ..authoring_form import display_value, project_schema, project_values

    schema = descriptor.authoring_schema
    if not schema.fields:
        return dict(values)
    edited = parent.edit_values(
        project_schema(schema),
        {name: display_value(value) for name, value in dict(values).items()},
        title=f"{descriptor.api_name} settings",
    )
    if edited is None:
        return None
    try:
        return project_values(schema, edited)
    except Exception as error:
        # A refusal is not a cancellation.  Returning the same None as Cancel
        # made a rejected setting silently leave the node as it was, with
        # nothing on screen to say why.
        parent.show_warning(f"{descriptor.api_name} settings", str(error))
        return None


def create_window(
    *,
    workspace=None,
    template=None,
    pulse="calibration",
    interval_ms=200,
    window_ratio=None,
):
    """Open the console the way a human does, and return its window.

    The non-blocking public entry: what the launcher runs, what a notebook
    calls, and what zlc_ui's acceptance capture opens.  Closing it releases the
    display beat, the presenter, the camera monitor and the session in that
    order -- a worker left running keeps the process alive with no window.
    """

    from ..board import attach_qt

    _space, session, node, monitor, _loaded = open_experiment(
        workspace, template, pulse
    )
    # The window is opened by build_console, through zlc_ui's one entry: this
    # layer composes and wires, and no longer knows what a window is made of.
    window, presenter = build_console(
        session,
        node,
        interval_ms=interval_ms,
        release_bootstrap=monitor.close,
        window_ratio=window_ratio,
    )
    # The presenter's beat, not the board's: the board is one step of it.
    timer = attach_qt(presenter.beat, interval_ms=interval_ms)

    def _release():
        """Let go of everything, whatever failed first.

        It used to be four bare statements: a presenter that raised on the way
        down stranded the camera monitor and the session behind it -- an armed
        camera and an open device set, with no window left to release them.
        """

        failures: list[BaseException] = []
        for step in (timer.stop, presenter.close, monitor.close, session.close):
            try:
                step()
            except BaseException as error:  # noqa: BLE001 - every step still runs
                failures.append(error)
        if failures:
            raise failures[0]

    released: list[bool] = []

    def _guard() -> bool:
        """Let go BEFORE the window goes, and keep it if letting go failed.

        This window owns a render worker, an armed camera and whatever logic
        nodes are running.  Releasing them on ``closed`` -- after the close is
        committed -- means a failure part-way leaves an armed camera and an
        open device set with no window left to reach them.  A close guard is
        the migrated mechanism for exactly that and had no caller: the X now
        does nothing until the owners are confirmed down, and a failure leaves
        the window up so the operator can try again.
        """

        if released:
            return True
        try:
            _release()
        except BaseException:
            return False
        released.append(True)
        return True

    window.presenter = presenter
    window.session = session
    window.set_close_guard(_guard)
    return window


def _open_saved_figure(parent: object, start: str) -> object | None:
    """Open one saved figure in its own window, over today's data folder.

    The console does not become a viewer; it asks for one.  What a saved figure
    is, and how to read it, belongs to the viewer -- which needs no session and
    happily opens a file from another bench or another year.
    """

    from .figure_viewer import create_window as create_viewer_window

    path = parent.ask_open_path(
        "Open saved figure", start, "Saved figures (*.npz);;All files (*)"
    )
    if not path:
        return None
    # The viewer's own public entry, not a second assembly of it: one window
    # definition means the console cannot open a viewer that differs from the
    # one the viewer's launcher opens.
    return create_viewer_window(path=path).presenter


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app

    application = ensure_qt_app([])

    if arguments.check:
        # The same assembly, without a window: a smoke test, not acceptance.
        try:
            space, session, node, monitor, pulse = open_experiment(
                arguments.workspace, arguments.template, arguments.pulse
            )
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"workspace: {space.root}", flush=True)
        for key, failure in session.failures.items():
            print(f"warning: device {key!r} did not open: {failure}", file=sys.stderr)
        # The release covers building the console too.  A failure while the
        # window is being assembled is the whole point of a smoke check, and it
        # is exactly when a session left open holds a camera nobody can reach.
        presenter = None
        try:
            _view, presenter = build_console(
                session,
                node,
                interval_ms=arguments.interval_ms,
                release_bootstrap=monitor.close,
            )
            for _beat in range(3):
                presenter.beat()
            print(
                f"console ready: {len(presenter.panels)} panel(s), "
                f"{len(presenter.offered_signals())} more signal(s) offerable, "
                f"pulse {pulse['name']!r}, {pulse['camera_windows']} camera window(s)"
            )
            return 0
        finally:
            if presenter is not None:
                presenter.close()
            monitor.close()
            session.close()

    try:
        window = create_window(
            workspace=arguments.workspace,
            template=arguments.template,
            pulse=arguments.pulse,
            interval_ms=arguments.interval_ms,
        )
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"workspace: {window.session.workspace.root}")
    for key, failure in window.session.failures.items():
        print(f"warning: device {key!r} did not open: {failure}", file=sys.stderr)
    del window
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
