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


def open_experiment(workspace=None, template=None):
    """Open the shared Experiment session behind an initially empty console.

    Kept apart from the window so the same assembly serves the smoke check, a
    notebook, and the window entry -- three ways in, one experiment.
    """

    from ..session import ExperimentSession, Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()

    session = ExperimentSession.open(space.root, template=template)
    return space, session


def build_console(session, *, interval_ms, window_ratio=None):
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
        """Plot kinds that belong on TaskConsole's live signal board."""

        return tuple(
            row
            for row in plot.panel_kinds()
            if row[0] != plot.PlotKind.PULSE_TIMELINE.value
        )

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
        open_saved=lambda start: _open_saved_figure(view, start),
        default_interval_ms=interval_ms,
    )
    return view, presenter


class ExperimentGuiFlow:
    """One DeviceManager-owned session shared by TaskConsole and PulseGUI."""

    def __init__(
        self,
        *,
        workspace=None,
        template=None,
        interval_ms=200,
        window_ratio=None,
    ) -> None:
        from ..session import Workspace

        self.space = Workspace(workspace) if workspace is not None else Workspace.discover()
        self.template = template
        self.catalog = None
        self.interval_ms = int(interval_ms)
        self.window_ratio = window_ratio
        self.devices = None
        self.session = None
        self.console = None
        self.console_presenter = None
        self.pulse = None
        self.timer = None
        self._closing_console = False
        self._closing_all = False

    def open(self) -> "ExperimentGuiFlow":
        from zlc_atom.install import (
            discover_device_catalog,
            installation_config_from_template,
        )

        from .device_manager import create_window as create_device_window

        if self.devices is not None:
            return self
        catalog = discover_device_catalog()
        initial_config = (
            None
            if self.template is None
            else installation_config_from_template(catalog, str(self.template))
        )
        self.catalog = catalog
        self.devices = create_device_window(
            workspace=self.space.root,
            window_ratio=self.window_ratio,
            catalog=catalog,
            initial_config=initial_config,
            initialize_session=self._initialize_session,
            on_initialized=self._open_work_windows,
            shutdown_session=self._shutdown_session,
        )
        return self

    def _initialize_session(self, config: object):
        from ..session import ExperimentSession

        if self.catalog is None:
            raise RuntimeError("experiment device catalog is not initialized")
        return ExperimentSession.from_config(self.space, config, catalog=self.catalog)

    def _open_work_windows(self, session: object) -> None:
        from ..board import attach_qt
        from .pulse_editor import create_bound_window

        if self.session is not None:
            raise RuntimeError("this experiment flow already has an active session")
        console = pulse = presenter = timer = None
        try:
            console, presenter = build_console(
                session,
                interval_ms=self.interval_ms,
                window_ratio=self.window_ratio,
            )
            pulse = create_bound_window(
                workspace=self.space,
                sequence=None,
                sequencer=session.sequencer,
                device_use=session.device_use,
                path="",
                window_ratio=self.window_ratio,
            )

            def beat() -> None:
                presenter.beat()
                pulse.presenter.refresh_run_state()

            timer = attach_qt(beat, interval_ms=self.interval_ms)
            console.presenter = presenter
            console.session = session
            console.set_close_guard(self._console_close_guard)
        except BaseException:
            if timer is not None:
                timer.stop()
            if pulse is not None:
                pulse.presenter.close()
                pulse.close()
            if presenter is not None:
                presenter.close()
            if console is not None:
                console.set_close_guard(lambda: True)
                console.close()
            raise
        self.session = session
        self.console = console
        self.console_presenter = presenter
        self.pulse = pulse
        self.timer = timer
        self.devices.hide()

    def _retire_pulse(self) -> None:
        pulse = self.pulse
        if pulse is None:
            return
        pulse.presenter.close()
        pulse.close()
        self.pulse = None

    def _retire_console(self, *, close_window: bool) -> None:
        timer = self.timer
        if timer is not None:
            timer.stop()
        presenter = self.console_presenter
        if presenter is not None:
            try:
                presenter.close()
            except BaseException:
                if timer is not None:
                    timer.start()
                raise
        self.timer = None
        self.console_presenter = None
        console, self.console = self.console, None
        if close_window and console is not None:
            console.set_close_guard(lambda: True)
            console.close()

    def _shutdown_session(self, session: object) -> None:
        if self.session is not None and session is not self.session:
            raise RuntimeError("DeviceManager tried to retire another experiment session")
        self._retire_pulse()
        self._retire_console(close_window=not self._closing_console)
        session.close()
        self.session = None
        if self.devices is not None and not self._closing_all:
            self.devices.restore()

    def _console_close_guard(self) -> bool:
        if self._closing_console:
            return False
        self._closing_console = True
        self._closing_all = True
        try:
            if self.devices is not None:
                if not self.devices.presenter.shutdown_active():
                    return False
                self.devices.presenter.close()
                self.devices.close()
            return True
        except BaseException:
            return False
        finally:
            self._closing_console = False

    def close(self) -> None:
        """Retire Pulse, Console, the shared session, then DeviceManager."""

        self._closing_all = True
        if self.console is not None:
            console = self.console
            console.close()
            if self.console is not None:
                raise RuntimeError("TaskConsole refused to close its active experiment")
        elif self.devices is not None:
            if not self.devices.presenter.shutdown_active():
                raise RuntimeError("DeviceManager refused to retire its active session")
            self.devices.presenter.close()
            self.devices.close()
        self.devices = None


def create_experiment_flow(
    *,
    workspace=None,
    template=None,
    interval_ms=200,
    window_ratio=None,
) -> ExperimentGuiFlow:
    """Open the v1-shaped experiment entry without entering Qt's event loop."""

    return ExperimentGuiFlow(
        workspace=workspace,
        template=template,
        interval_ms=interval_ms,
        window_ratio=window_ratio,
    ).open()


def create_console_window(
    *,
    workspace=None,
    template=None,
    interval_ms=200,
    window_ratio=None,
):
    """Open only TaskConsole for notebook and acceptance-capture callers.

    This compatibility helper owns the session it creates.  The experiment
    launcher does not use it: ``main`` uses :func:`create_experiment_flow` so
    DeviceManager creates one session shared by TaskConsole and PulseEditor.
    """

    from ..board import attach_qt

    _space, session = open_experiment(workspace, template)
    # The window is opened by build_console, through zlc_ui's one entry: this
    # layer composes and wires, and no longer knows what a window is made of.
    window, presenter = build_console(
        session,
        interval_ms=interval_ms,
        window_ratio=window_ratio,
    )
    # The presenter's beat, not the board's: the board is one step of it.
    timer = attach_qt(presenter.beat, interval_ms=interval_ms)

    def _release():
        """Release owners in order, never closing devices under a live node."""

        timer.stop()
        try:
            presenter.close()
        except BaseException:
            # A timed-out node still needs the normal beat to poll its worker.
            # The close guard keeps the window, so keep that window responsive.
            timer.start()
            raise
        session.close()

    released: list[bool] = []

    def _guard() -> bool:
        """Let go BEFORE the window goes, and keep it if letting go failed.

        This window owns a render worker and whatever logic nodes are running.
        Releasing them on ``closed`` -- after the close is committed -- means
        a failure part-way leaves an open device set with no window left to
        reach it.  A close guard is
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


def create_window(
    *,
    workspace=None,
    template=None,
    interval_ms=200,
    window_ratio=None,
):
    """Compatibility entry for callers that explicitly request one console."""

    return create_console_window(
        workspace=workspace,
        template=template,
        interval_ms=interval_ms,
        window_ratio=window_ratio,
    )


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
            space, session = open_experiment(arguments.workspace, arguments.template)
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
                interval_ms=arguments.interval_ms,
            )
            for _beat in range(3):
                presenter.beat()
            print(
                f"console ready: {len(presenter.panels)} panel(s), "
                f"{len(presenter.offered_signals())} more signal(s) offerable"
            )
            return 0
        finally:
            if presenter is not None:
                presenter.close()
            session.close()

    try:
        flow = create_experiment_flow(
            workspace=arguments.workspace,
            template=arguments.template,
            interval_ms=arguments.interval_ms,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"workspace: {flow.space.root}")
    try:
        return int(application.exec_())
    finally:
        flow.close()


if __name__ == "__main__":
    raise SystemExit(main())
