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
from zlc_atom.devices.sequencer.virtual import VirtualPulseStreamer
from zlc_pulse import load_streamer_config, pulse_target_from_xdc
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
    explicit = tuple(
        line.split("=")[0].strip().removeprefix("self.")
        for line in source.splitlines()
        if "= _Signal()" in line
    )
    declared = tuple(getattr(double, "_INTENTS", ())) + tuple(
        getattr(double, "_SIGNALS", ())
    )
    return tuple(dict.fromkeys(explicit + declared))


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
            target=pulse_target_from_xdc(config_path=_CONFIG["source"]),
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


def test_the_card_does_not_own_a_second_copy_of_the_figure_geometry() -> None:
    """A card is a frame around a picture; the picture's size is the drawer's.

    zlc_ui used to restate the margins, the cell unit and the display scale,
    and the test here stood over the two copies checking they still agreed --
    a guard whose existence admits there are two of something.  Correcting the
    panel margins against the instrument these figures descend from made the
    copy drift, and this is what noticed.  Now the size is SUPPLIED by the
    composition layer, which is the only one that may import both, so the two
    cannot disagree.

    What is left to check is that nobody quietly copies it back.
    """

    from pathlib import Path

    import zlc_ui.board.panel_geometry as geometry
    from zlc_plot import DEFAULTS
    from zlc_plot.kinds import PlotKind
    from zlc_plot.layout import resolve_surface
    from zlc_ui.board import PANEL_SIZES, panel_display_size

    for name in PANEL_SIZES:
        plan = resolve_surface(
            name, PlotKind.CURVE, layout=DEFAULTS.layout, style=DEFAULTS.style
        )
        assert panel_display_size(name) == tuple(plan.logical_size), name

    assert PANEL_SIZES == tuple(
        preset.name for preset in DEFAULTS.layout.presets
    ), "the two packages offer different panel sizes"

    # No margins, no cell unit, no scale: the numbers live in one package.
    source = Path(geometry.__file__).read_text(encoding="utf-8")
    for copied in ("_PANEL_MARGINS_PX", "_PANEL_UNIT_PX", "_PANEL_DISPLAY_SCALE"):
        assert copied not in source, f"{copied} is a second copy of zlc_plot's geometry"


def test_with_nobody_drawing_a_card_is_sized_as_the_empty_frame_it_is() -> None:
    """zlc_ui must stay usable alone -- its gallery and demos are real users.

    So an unsupplied board falls back to its OWN cell, which copies nothing:
    the fallback is proportional to the preset's grid and carries no margin
    and no scale.  That is the difference between a default and a duplicate.
    """

    import zlc_ui.board.panel_geometry as geometry

    supplied = geometry._measure
    geometry._measure = None
    try:
        cell_width, cell_height = geometry.PLACEHOLDER_CELL_PX
        assert geometry.panel_display_size("2x2") == (2 * cell_width, 2 * cell_height)
        assert geometry.panel_display_size("8x4") == (4 * cell_width, 8 * cell_height)
    finally:
        geometry._measure = supplied


def test_the_two_content_name_rules_agree() -> None:
    """zlc_data and zlc_pulse each state the digest width, and cannot import
    each other -- zlc_pulse depends on numpy and pyserial so a board can be
    driven with no data layer present.  This is where both are visible."""

    from zlc_data.validation import DIGEST_BITS as DATA_BITS
    from zlc_pulse.canonical import DIGEST_BITS as PULSE_BITS

    assert DATA_BITS == PULSE_BITS
