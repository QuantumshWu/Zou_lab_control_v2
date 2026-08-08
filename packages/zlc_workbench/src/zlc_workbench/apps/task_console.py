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
    """Open the shared Experiment session behind an initially empty console.

    Kept apart from the window so the same assembly serves the smoke check, a
    notebook, and the window entry -- three ways in, one experiment.
    """

    from ..session import ExperimentSession, Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()

    session = ExperimentSession.open(space.root, template=template)
    loaded = session.load_pulse(pulse)
    return space, session, loaded


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
        # A logic node is what publishes a signal.  The question of which type
        # to add and the settings form are Qt; what a node IS, and what running
        # one means, are not, and stay out of this file.
        choose_logic=lambda rows: _choose_logic(view, rows),
        default_interval_ms=interval_ms,
    )
    return view, presenter

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


def _template_config(name: str | None):
    """Project one named installation template into DeviceManager's plain draft."""

    if name is None:
        return None
    from zlc_atom.install import INSTALLATION_TEMPLATES
    from zlc_atom.install.configuration import DeviceInstanceConfig, InstallationConfig

    try:
        specs = INSTALLATION_TEMPLATES[str(name)]
    except KeyError as error:
        raise KeyError(f"unknown installation template {name!r}") from error
    return InstallationConfig(
        tuple(
            DeviceInstanceConfig(
                instance_id=spec.key,
                role=spec.key,
                type_id=spec.type_id,
                parameters=dict(spec.config),
            )
            for spec in specs
        )
    )


class ExperimentGuiFlow:
    """One DeviceManager-owned session shared by TaskConsole and PulseGUI."""

    def __init__(
        self,
        *,
        workspace=None,
        template=None,
        pulse="calibration",
        interval_ms=200,
        window_ratio=None,
    ) -> None:
        from ..session import Workspace

        self.space = Workspace(workspace) if workspace is not None else Workspace.discover()
        self.template = template
        self.pulse_name = str(pulse)
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
        from .device_manager import create_window as create_device_window

        if self.devices is not None:
            return self
        self.devices = create_device_window(
            workspace=self.space.root,
            window_ratio=self.window_ratio,
            initial_config=_template_config(self.template),
            initialize_session=self._initialize_session,
            on_initialized=self._open_work_windows,
            shutdown_session=self._shutdown_session,
        )
        return self

    def _initialize_session(self, config: object):
        from ..session import ExperimentSession

        session = ExperimentSession.from_config(self.space, config)
        try:
            session.load_pulse(self.pulse_name)
        except BaseException:
            session.close()
            raise
        return session

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
            timer = attach_qt(presenter.beat, interval_ms=self.interval_ms)
            pulse = create_bound_window(
                workspace=self.space,
                sequence=session.pulse_sequence,
                sequencer=session.sequencer,
                path=str(session.pulse_path or ""),
                window_ratio=self.window_ratio,
            )
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
        pulse, self.pulse = self.pulse, None
        if pulse is None:
            return
        pulse.presenter.close()
        pulse.close()

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
    pulse="calibration",
    interval_ms=200,
    window_ratio=None,
) -> ExperimentGuiFlow:
    """Open the v1-shaped experiment entry without entering Qt's event loop."""

    return ExperimentGuiFlow(
        workspace=workspace,
        template=template,
        pulse=pulse,
        interval_ms=interval_ms,
        window_ratio=window_ratio,
    ).open()


def create_console_window(
    *,
    workspace=None,
    template=None,
    pulse="calibration",
    interval_ms=200,
    window_ratio=None,
):
    """Open only TaskConsole for notebook and acceptance-capture callers.

    This compatibility helper owns the session it creates.  The experiment
    launcher does not use it: ``main`` uses :func:`create_experiment_flow` so
    DeviceManager creates one session shared by TaskConsole and PulseEditor.
    """

    from ..board import attach_qt

    _space, session, _loaded = open_experiment(workspace, template, pulse)
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
    pulse="calibration",
    interval_ms=200,
    window_ratio=None,
):
    """Compatibility entry for callers that explicitly request one console."""

    return create_console_window(
        workspace=workspace,
        template=template,
        pulse=pulse,
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
            space, session, pulse = open_experiment(
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
                interval_ms=arguments.interval_ms,
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
            session.close()

    try:
        flow = create_experiment_flow(
            workspace=arguments.workspace,
            template=arguments.template,
            pulse=arguments.pulse,
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
