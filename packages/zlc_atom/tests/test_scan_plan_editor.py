"""The scan plan editor, authored as rows rather than as JSON.

The manual row is the one that carries no port -- everything else about it
is an ordinary axis -- so the only way to know it authors the right
document is to build the widget, press the button, and read what it
emitted.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from zlc_atom.nodes.scan import ScanPlan, manual_axis_name
from zlc_atom.nodes.scan.editor import scan_plan_editor_factory
from zlc_atom.nodes.scan.plan import ScanPort
from zlc_ui import ensure_qt_app


BIAS = ScanPort("pulse:param:da_bias_x", "da_bias_x", "V", -1.0, 1.0, -0.5, 0.5)


def _editor(manual_axes: bool = True):
    ensure_qt_app()
    editor = scan_plan_editor_factory(
        device_ports=False, hardware_slots=True, manual_axes=manual_axes
    )
    # The projection is what a node hands its editor; only the ports and the
    # authored plan matter here, so the rest of it stays out of the way.
    editor._ports = (BIAS,)
    editor.add_button.setEnabled(True)
    return editor


def _plan(editor) -> ScanPlan:
    return ScanPlan.from_tree(json.loads(editor._plan_text))


def test_a_manual_row_authors_a_name_and_the_values_a_hand_will_set() -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("power")
        row.start_spin.setValue(1.0)
        row.stop_spin.setValue(4.0)
        row.points_spin.setValue(4)

        plan = _plan(editor)
        assert [axis.port for axis in plan.axes] == [
            "manual:power",
            "pulse:param:da_bias_x",
        ], "a manual axis is authored OUTSIDE the axes a machine advances"
        manual = plan.axes[0]
        assert manual_axis_name(manual.port) == "power"
        # The same from/to/points every other row authors: a coordinate has
        # to exist before the data it describes, whoever turns the knob.
        assert manual.values == (1.0, 2.0, 3.0, 4.0)
    finally:
        editor.deleteLater()


def test_a_fresh_manual_row_is_already_runnable() -> None:
    """An unnamed axis is not a plan, so the button never authors one."""

    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        editor._add_manual_axis()
        names = [
            manual_axis_name(axis.port)
            for axis in _plan(editor).axes
            if axis.port.startswith("manual:")
        ]
        assert names == ["manual 1", "manual 2"], (
            "a default name, and never the same one twice"
        )
    finally:
        editor.deleteLater()


def test_a_manual_row_cannot_be_authored_inside_the_machine_axes() -> None:
    """Display order IS nesting order, so the row goes where it runs."""

    editor = _editor()
    try:
        editor._add_axis()
        editor._add_axis()
        editor._add_manual_axis()
        editor._add_manual_axis()
        assert [row.manual for row in editor._rows] == [
            True,
            True,
            False,
            False,
        ]
        positions = [
            editor.rows_layout.indexOf(row) for row in editor._rows
        ]
        assert positions == sorted(positions), (
            "the laid-out order must match the authored order"
        )
    finally:
        editor.deleteLater()


def test_a_node_that_cannot_stop_for_a_hand_never_offers_the_button() -> None:
    editor = _editor(manual_axes=False)
    try:
        assert editor.add_manual_button.isHidden()
        assert not editor.add_manual_button.isEnabled()
    finally:
        editor.deleteLater()


def test_the_summary_says_how_many_stops_the_operator_is_signing_up_for() -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("power")
        row.points_spin.setValue(3)
        editor._refresh_summary()
        assert "3 of those points are reached by hand" in editor.summary.text()
    finally:
        editor.deleteLater()


@pytest.mark.parametrize("points", (1, 7))
def test_a_manual_row_round_trips_through_the_document(points: int) -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("angle")
        row.points_spin.setValue(points)
        text = editor._plan_text

        reopened = _editor()
        try:
            reopened._rebuild_rows(text)
            restored = next(row for row in reopened._rows if row.manual)
            assert restored.name_edit.text() == "angle"
            assert int(restored.points_spin.value()) == points
        finally:
            reopened.deleteLater()
    finally:
        editor.deleteLater()
