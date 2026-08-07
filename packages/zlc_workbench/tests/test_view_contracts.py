"""The stand-in views in these tests must not be able to lie.

Every presenter test here drives a hand-written double instead of a Qt widget,
which is what keeps them headless and fast.  It also means a double can declare
a method the real view does not have, or the same method with different
arguments, and every test still passes while the window built from the same
presenter raises ``TypeError`` on startup.  That has now happened four times,
each time costing a full debugging round on a symptom -- once as a process that
hung for five minutes with the traceback trapped behind a live worker thread.

So it is checked rather than remembered: for every double, each public member it
declares must exist on the real thing, and callables must have the SAME
signature.  One direction only, deliberately.  A real view may offer more than
any presenter uses -- that is not drift.  Drift is a double that answers a call
the real view would refuse.
"""

from __future__ import annotations

import dataclasses
import inspect
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt5 import QtCore

from zlc_ui.console.panel_card_view import PanelCardView
from zlc_ui.console.logic_row_view import LogicRowView
from zlc_ui.console.handle import TaskConsoleHandle
from zlc_ui.device_manager.handle import DeviceManagerHandle
from zlc_ui.figure_viewer.handle import FigureViewerHandle
from zlc_ui.pulse.handle import PulseEditorHandle
from zlc_ui.pulse.preview_view import PulsePreviewView
from zlc_ui.pulse.scan_view import PulseScanView
from zlc_ui.pulse.schedule_view import PulseScheduleView
from zlc_ui.pulse.target_view import PulseTargetView
from zlc_atom.devices.sequencer import device_types as atom_sequencer_types
from zlc_atom.devices.sequencer import protocol as atom_sequencer
from zlc_atom.devices.sequencer.virtual import VirtualPulseStreamer
from zlc_pulse import load_streamer_config
from zlc_pulse import device, remote
from zlc_pulse.device import PulseStreamer
from zlc_pulse.transport import MemoryRegisterTransport

_CONFIG = load_streamer_config()
_GEOMETRY = _CONFIG["params"]
_CLOCK_HZ = _CONFIG["clock_hz"]

import test_console_logic as logic_tests
import test_console_presenter as console_tests
import test_device_manager as manager_tests
import test_pulse_editor as pulse_tests
import test_viewer as viewer_tests


PAIRS = [
    (console_tests._CardView, PanelCardView),
    (console_tests._ConsoleView, TaskConsoleHandle),
    (logic_tests._LogicRowView, LogicRowView),
    (viewer_tests._ViewerView, FigureViewerHandle),
    (manager_tests._ManagerView, DeviceManagerHandle),
    (pulse_tests._ScheduleView, PulseScheduleView),
    (pulse_tests._PreviewView, PulsePreviewView),
    (pulse_tests._TargetView, PulseTargetView),
    (pulse_tests._ScanView, PulseScanView),
    # The editor is reached through its HANDLE now, so that is what the double
    # has to mirror: the port is what a presenter can call, and the widget tree
    # behind it is not reachable from here at all.
    (pulse_tests._EditorView, PulseEditorHandle),
    # Not a view, and here for the same reason: a board that answers questions
    # the real board would refuse lets a presenter be tested against a machine
    # that does not exist.
    (pulse_tests._Sequencer, PulseStreamer),
]

#: Members a double owns rather than mirrors: Qt's own vocabulary, and the
#: recording a test does to see what it was told.  ``cards``/``logic_rows``
#: are that recording exposed as a property -- class-level, so the walk finds
#: them, but they answer "what did the window end up holding", which the real
#: handle deliberately does not offer outward.
NOT_CONTRACT = {"setParent", "show", "close", "deleteLater", "cards", "logic_rows"}


def _double_members(double: type) -> dict[str, object]:
    """What the double declares as a view: its methods and its signals.

    Attributes assigned in ``__init__`` are the recordings a test reads back, so
    only class-level members are contract.  A signal is class-level in the real
    view and instance-level in the double, so signals are found by name against
    the real view instead.
    """

    return {
        name: value
        for name, value in vars(double).items()
        if not name.startswith("_") and name not in NOT_CONTRACT
        and (callable(value) or isinstance(value, property))
    }


def _signal_names(double: type) -> tuple[str, ...]:
    """Signal names the double raises, taken from what its instances hold."""

    source = inspect.getsource(double)
    return tuple(
        line.split("=")[0].strip().removeprefix("self.")
        for line in source.splitlines()
        if "= _Signal()" in line
    )


@pytest.mark.parametrize("double,real", PAIRS, ids=lambda item: item.__name__)
def test_a_stand_in_view_declares_only_what_the_real_view_has(double, real) -> None:
    missing = [name for name in _double_members(double) if not hasattr(real, name)]
    assert not missing, (
        f"{double.__name__} answers {missing}, which {real.__name__} does not "
        "have.  Either the real view is missing it or the double invented it."
    )


@pytest.mark.parametrize("double,real", PAIRS, ids=lambda item: item.__name__)
def test_a_stand_in_view_takes_the_arguments_the_real_view_takes(double, real) -> None:
    wrong = []
    for name, member in _double_members(double).items():
        target = inspect.getattr_static(real, name, None)
        if target is None:
            continue
        # A property and a method are not the same member.  This guard was
        # written for exactly that mistake -- a double declaring include_off_rows
        # as a method against a real view that declares it as a property -- and
        # then skipped both, because a property object is not callable.
        if isinstance(member, property) != isinstance(target, property):
            kind = "a property" if isinstance(member, property) else "a method"
            wrong.append(f"{name} is {kind} here and the other kind there")
            continue
        here_fn = member.fget if isinstance(member, property) else member
        there_fn = target.fget if isinstance(target, property) else target
        try:
            here = inspect.signature(here_fn)
            there = inspect.signature(there_fn)
        except (TypeError, ValueError):  # pragma: no cover - C-level members
            continue
        if _shape(here) != _shape(there):
            wrong.append(f"{name}{here} != {name}{there}")
    assert not wrong, (
        f"{double.__name__} and {real.__name__} disagree about: "
        + "; ".join(wrong)
        + ".  A presenter tested against the left one crashes against the right."
    )


def _shape(signature: inspect.Signature) -> tuple[tuple[str, str], ...]:
    """A signature's callable shape: names and kinds, without annotations.

    Types are not compared -- a double is allowed to be untyped -- but names and
    how they may be passed are exactly what a caller depends on.
    """

    return tuple(
        (name, str(parameter.kind))
        for name, parameter in signature.parameters.items()
        if name != "self"
    )


@pytest.mark.parametrize("double,real", PAIRS, ids=lambda item: item.__name__)
def test_a_stand_in_view_raises_only_signals_the_real_view_raises(double, real) -> None:
    invented = [
        name
        for name in _signal_names(double)
        if not isinstance(getattr(real, name, None), QtCore.pyqtSignal)
    ]
    assert not invented, (
        f"{double.__name__} raises {invented}; {real.__name__} has no such "
        "signal, so nothing in the real window can ever raise it."
    )


# --------------------------------------------------------------- device mirrors

#: zlc_atom writes out what it needs of a pulse board instead of importing
#: zlc_pulse, so a camera or a sequencer can be driven with no zlc_pulse present.
#: This is the one place both are visible, so this is where the copy is checked.
MIRRORS = [
    (atom_sequencer.DoneReport, device.DoneReport),
    (atom_sequencer.SafeReadback, device.SafeReadback),
]


@pytest.mark.parametrize("mirror,real", MIRRORS, ids=lambda item: item.__name__)
def test_a_mirrored_record_has_the_fields_the_real_one_has(mirror, real) -> None:
    """A copy that names fields the original does not is a copy nobody can use.

    SafeReadback had drifted to three fields the board has never had.  Nothing
    read them, so nothing failed -- until the first piece of code that did.
    """

    here = {field.name: field.type for field in dataclasses.fields(mirror)}
    there = {field.name: field.type for field in dataclasses.fields(real)}
    invented = sorted(set(here) - set(there))
    assert not invented, (
        f"{mirror.__name__} names {invented}, which {real.__name__} does not "
        "have.  Either the board grew them or the copy imagined them."
    )
    differing = sorted(
        name for name in here if str(here[name]) != str(there[name])
    )
    assert not differing, (
        f"{mirror.__name__} and {real.__name__} disagree about the type of "
        f"{differing}"
    )


def test_the_board_protocol_asks_only_for_methods_the_board_has() -> None:
    """A requirement naming a method nobody implements admits nothing."""

    missing = [
        name
        for name in atom_sequencer.PulseStreamer.__protocol_attrs__
        if not hasattr(device.PulseStreamer, name)
    ]
    assert not missing, (
        f"zlc_atom asks a pulse board for {missing}, which zlc_pulse's board "
        "does not offer"
    )


def test_a_virtual_board_answers_the_questions_a_real_one_answers() -> None:
    """A twin that reports its state under different keys reports nothing.

    The caller asks whether it is firing, gets nothing back, and concludes an
    idle board -- the one answer that was never true.  Snapshot is the level
    state everything else is derived from, so its vocabulary is the contract.
    """

    real = set(
        device.PulseStreamer(
            MemoryRegisterTransport(geom=_GEOMETRY, auto_done=True),
            _GEOMETRY,
            _CLOCK_HZ,
        ).snapshot()
    )
    twin = set(VirtualPulseStreamer().snapshot())

    assert not real - twin, (
        f"the virtual sequencer cannot answer {sorted(real - twin)}, which a "
        "real board reports"
    )


def test_the_apparatus_form_offers_the_port_a_server_actually_listens_on() -> None:
    """An apparatus form cannot import zlc_pulse, so it writes the port down.

    Then the two can disagree, and the way that shows up is an operator filling
    in a saved apparatus, pressing connect, and being told nothing is there.
    """

    fields = {
        field.name: field.default
        for field in atom_sequencer_types.HARDWARE_SEQUENCER_SCHEMA.fields
    }

    assert fields["port"] == remote.DEFAULT_PORT
    assert fields["host"] == remote.DEFAULT_HOST


# ------------------------------------------------------- panel geometry mirror

def test_a_card_is_sized_for_the_figure_that_goes_in_it() -> None:
    """The nine presets and their pixel geometry are written in two packages.

    zlc_plot draws the figure and zlc_ui sizes the card that holds it, and
    neither may import the other -- zlc_ui carries no domain dependency at all.
    So the constants are a forced mirror, and if they drift the operator gets a
    card with the wrong-sized picture in it, at every preset.
    """

    from zlc_plot import DEFAULTS
    from zlc_plot.kinds import PlotKind
    from zlc_plot.layout import resolve_surface
    from zlc_ui.board import PANEL_SIZES, panel_display_size

    for name in PANEL_SIZES:
        surface = resolve_surface(
            name, PlotKind.CURVE, layout=DEFAULTS.layout, style=DEFAULTS.style
        )
        assert tuple(surface.logical_size) == panel_display_size(name), name

    assert PANEL_SIZES == tuple(
        preset.name for preset in DEFAULTS.layout.presets
    ), "the two packages offer different panel sizes"


def test_the_two_content_name_rules_agree() -> None:
    """zlc_data and zlc_pulse each state the digest width, and cannot import
    each other -- zlc_pulse depends on numpy and pyserial so a board can be
    driven with no data layer present.  This is where both are visible."""

    from zlc_data.validation import DIGEST_BITS as DATA_BITS
    from zlc_pulse.canonical import DIGEST_BITS as PULSE_BITS

    assert DATA_BITS == PULSE_BITS


def test_both_done_reports_agree_on_what_a_bad_shot_is() -> None:
    """A shot that errored or underran must not look like a clean one.

    The rule is stated in zlc_pulse and mirrored in zlc_atom, which cannot
    import it.  If the two disagree, the same board answer is a fault on one
    path and a good shot on the other.
    """

    from zlc_pulse.wire import STATUS_ERROR, STATUS_UNDERFLOW

    assert atom_sequencer.STATUS_ERROR == STATUS_ERROR
    assert atom_sequencer.STATUS_UNDERFLOW == STATUS_UNDERFLOW

    for status, underflow in ((0, False), (STATUS_ERROR, False), (0, True)):
        real = device.DoneReport(status, None, underflow, 0.0)
        mirror = atom_sequencer.DoneReport(status, None, underflow, 0.0)
        assert real.fault == mirror.fault, (status, underflow)
    assert device.DoneReport(0, None, False, 0.0).fault == ""
    assert device.DoneReport(STATUS_ERROR, None, False, 0.0).fault
