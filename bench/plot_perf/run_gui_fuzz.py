"""Seeded real-Qt interaction runs; outputs live only in bench/results.

Setup reuses ConsoleBench's virtual experiment. Tested actions travel through
widgets, never through presenter setters. A supervisor owns each child so a
broken event loop cannot strand an invisible test or an open test window.

Run from this checkout: python -m bench.plot_perf.run_gui_fuzz --chain --seed 307 --actions 120
Replay its recorded actions with --replay <run>/replay.json; --stop-on-report
<substring> narrows a reported failure. Results stay in ignored bench/results.
An observation_completed run is NOT a blanket pass: inspect needs_review,
findings, reported errors, skipped actions and the observer canary. Expected
input refusals (for example a missing axis role) remain visible in the evidence.
"""
from __future__ import annotations

import zou_lab_control

import argparse
import faulthandler
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .run_console import ConsoleBench
from .common import Pointer, axis_center, axis_by_role
from .guards import ProductBeat
from .gui_checks import (check_panel, install_observers, panel_checkpoint,
                         check_scientific_chain, check_overlay_pixels)


class UnavailableAction(RuntimeError):
    """A planned widget is no longer a place an operator can click."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def visible(widget, app):
    """Scroll the actual ancestor viewports, then use the widget's live box."""
    from PyQt5 import QtWidgets
    window = widget.window()
    if window.isVisible() and not window.isActiveWindow():
        window.raise_()
        window.activateWindow()
        app.processEvents()
    parent = widget.parentWidget()
    while parent is not None:
        if (isinstance(parent, QtWidgets.QScrollArea)
                and parent.widget() is not None
                and (parent.widget() is widget or parent.widget().isAncestorOf(widget))):
            parent.ensureWidgetVisible(widget, 8, 8)
        parent = parent.parentWidget()
    app.processEvents()
    return widget.isVisible() and widget.isEnabled()


def click(widget, app):
    from PyQt5 import QtCore, QtTest, QtWidgets
    from PyQt5 import sip
    if not visible(widget, app):
        parents = []
        current = widget
        while current is not None:
            parents.append((type(current).__name__, current.isVisible(), current.isEnabled(),
                            (current.x(), current.y(), current.width(), current.height())))
            current = current.parentWidget()
        print("UNUSABLE_TARGET", parents, flush=True)
        raise UnavailableAction(f"target is not usable: {type(widget).__name__}")
    point = widget.rect().center()
    if (isinstance(widget, QtWidgets.QAbstractButton) and sip.ispycreated(widget)
            and not widget.hitButton(point)):
        # A stretched form cell is not the switch/check-box's painted track.
        # Use Qt's own size hint and hit test, not a guessed per-widget width.
        point.setX(min(widget.width(), widget.sizeHint().width()) // 2)
        if not widget.hitButton(point):
            raise UnavailableAction("button has no hittable content at its size hint")
    root = widget.window()
    hit = root.childAt(root.mapFromGlobal(widget.mapToGlobal(point)))
    if hit is not widget and not widget.isAncestorOf(hit):
        raise UnavailableAction(f"target is covered: {type(widget).__name__} by {type(hit).__name__}")
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, pos=point)


def choose(combo, wanted, app):
    """Select a real popup row, including a leaf in the signal tree."""
    from PyQt5 import QtCore, QtTest
    from enum import Enum
    click(combo, app)
    app.processEvents()
    view = combo._popup_view
    if view is None or not view.isVisible():
        raise RuntimeError("choice popup did not open")
    model = view.model()

    def find(parent=QtCore.QModelIndex()):
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            value = index.data(QtCore.Qt.UserRole)
            if (value == wanted or isinstance(value, Enum) and str(value) == wanted
                    or value is None and index.data(QtCore.Qt.DisplayRole) == wanted):
                return index
            result = find(index)
            if result is not None:
                return result
        return None

    index = find()
    if index is None:
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Escape)
        raise UnavailableAction(f"choice not present in opened popup: {wanted!r}")
    parents = []
    ancestor = index.parent()
    while ancestor.isValid():
        parents.append(ancestor)
        ancestor = ancestor.parent()
    for ancestor in reversed(parents):
        view.scrollTo(ancestor)
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton,
                               pos=view.visualRect(ancestor).center())
        app.processEvents()
    view.scrollTo(index)
    app.processEvents()
    QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton,
                           pos=view.visualRect(index).center())


def inventory(bench):
    cards = {}
    for identity, card in bench.view._cards.items():
        fields = []
        form = card._settings_form
        if form is not None:
            for field in form.spec.fields:
                widget = form.widget_for(field.key)
                try:
                    value = form.read_value(field.key)
                except ValueError as error:
                    value = {"invalid_input": str(error)}
                fields.append(dict(key=field.key, label=field.label, kind=field.kind,
                    widget=type(widget).__name__, enabled=widget.isEnabled(),
                    visible=widget.isVisible(), value=value,
                    choices=[dict(value=item.value, label=item.label) for item in field.choices]))
        cards[identity] = dict(fields=fields, title=card._title_label.text())
    return cards


def checkpoint(bench, previous, *, stable):
    states, findings = {}, []
    for identity, panel in bench.presenter.panels.items():
        card = bench.view._cards.get(identity)
        editor = bench.view._panel_editors.get(identity)
        states[identity] = panel_checkpoint(panel, card, editor=editor)
        findings.extend(check_panel(panel, card, editor=editor,
                                    stable=stable, before=previous.get(identity)))
    return states, findings


def tab(bench, index):
    from PyQt5 import QtCore, QtTest
    bar = bench.view._view.tabs.tabBar()
    QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=bar.tabRect(index).center())
    bench.app.processEvents()


def settings(bench, panel_id):
    tab(bench, 0)
    hide_settings(bench, except_panel=panel_id)
    card = bench.view._cards[panel_id]
    if card._settings_form is None or not card._settings_form.isVisible():
        click(card.settings_button, bench.app)
        bench.app.processEvents()
    return card, card._settings_form


def hide_settings(bench, *, except_panel=None):
    # Several page-owned overlays may coexist. Close the actually exposed
    # close button first; never send through one popup into another.
    while True:
        opened = [card for key, card in bench.view._cards.items()
                  if key != except_panel and card._settings_popup is not None
                  and card._settings_popup.isVisible()]
        if not opened:
            return
        for card in reversed(opened):
            button = card._settings_close_button
            root = button.window()
            hit = root.childAt(root.mapFromGlobal(button.mapToGlobal(button.rect().center())))
            if hit is button or button.isAncestorOf(hit):
                click(button, bench.app)
                bench.app.processEvents()
                break
        else:
            raise UnavailableAction("no settings close button is exposed; drag or layout observation required")


def enter_text(widget, text, app):
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    if hasattr(widget, "edit") and isinstance(widget.edit, QtWidgets.QLineEdit):
        widget = widget.edit
    edit = widget.lineEdit() if isinstance(widget, QtWidgets.QAbstractSpinBox) else widget
    click(edit, app)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Backspace)
    # Resolve the current keyboard receiver each time: reconciliation may
    # replace a control. Sending to a deleted cached QWidget is a harness bug.
    for character in str(text):
        target = app.focusWidget()
        if target is None:
            raise RuntimeError("text edit lost keyboard focus")
        if character in ("\n", "\t"):
            QtTest.QTest.keyClick(target, QtCore.Qt.Key_Return if character == "\n" else QtCore.Qt.Key_Tab)
        else:
            event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, 0, QtCore.Qt.NoModifier, character)
            app.sendEvent(target, event)
        app.processEvents()
    target = app.focusWidget()
    if target is not None and not isinstance(edit, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
        QtTest.QTest.keyClick(target, QtCore.Qt.Key_Return)


def field_edit(bench, action):
    from PyQt5 import QtWidgets
    _card, form = settings(bench, action["panel"])
    key, value = action["field"], action["value"]
    if key not in form.keys:
        raise UnavailableAction(f"field is absent from current form: {key}")
    widget = form.widget_for(key)
    if hasattr(widget, "_popup_view"):
        choose(widget, value, bench.app)
    elif isinstance(widget, QtWidgets.QAbstractButton):
        if widget.isChecked() != value:
            click(widget, bench.app)
    else:
        enter_text(widget, value, bench.app)


def source_choices(bench):
    return sorted({name for card in bench.view._cards.values()
                   for name in card._signal_runtime.choice_names("signal")})


def data_axes(widget):
    return [axis for axis in widget.presented_front.interaction.axes
            if axis.role in ("main", "curve", "image", "histogram", "history", "rolling", "facet_cell")]


def roi_drag(bench, action, beat):
    from PyQt5 import QtCore, QtGui
    tab(bench, 0)
    hide_settings(bench)
    panel = bench.presenter.panels[action["panel"]]
    widget = bench.surface(panel)
    if widget is None or widget.presented_front is None:
        return "no image surface"
    if not widget.interaction_enabled:
        click(bench.view._view.selectors_switch, bench.app)
    if panel.state.kind == "facet_grid" and widget.presented_front.interaction.facet_focus_index is None:
        axes = data_axes(widget)
        pointer = Pointer(widget, bench.app, QtCore, QtGui)
        pointer.post = True
        pointer.dclick(*axis_center(axes[0]))
        if not beat.run_until(lambda: widget.presented_front.interaction.facet_focus_index is not None, 4.0):
            raise AssertionError("double click did not focus the image cell")
    axes = data_axes(widget)
    axis = axes[0]
    mx, my = sum(axis.x_limits)/2, sum(axis.y_limits)/2
    width = abs(axis.x_limits[1]-axis.x_limits[0])*action["width"]
    height = abs(axis.y_limits[1]-axis.y_limits[0])*action["height"]
    begin = axis.display_to_normalized(mx-width/2, my-height/2)
    end = axis.display_to_normalized(mx+width/2, my+height/2)
    pointer = Pointer(widget, bench.app, QtCore, QtGui)
    pointer.post = True
    pointer.press(*begin)
    beat.run(.02)
    for part in range(1,6):
        pointer.move(begin[0]+(end[0]-begin[0])*part/5,
                     begin[1]+(end[1]-begin[1])*part/5)
        beat.run(.005)
    pointer.release(*end)


def pick_action(bench, rng, step):
    view = bench.view._view
    panels = tuple(bench.presenter.panels.values())
    if not panels or (len(panels) < 4 and rng.random() < 0.12):
        return dict(kind="add", plot=rng.choice(("curve", "histogram", "image", "facet_grid", "rolling")),
                    source=rng.choice(source_choices(bench) or [bench.signal]))
    panel = rng.choice(panels)
    options = ["settings", "selectors", "pause", "tab", "gesture", "gesture", "edit",
               "field", "field", "field", "scope", "source", "restart", "stop", "save"]
    if len(panels) > 1:
        options.append("remove")
    kind = rng.choice(options)
    action = dict(kind=kind, panel=panel.panel_id)
    if kind == "field":
        form = bench.view._cards[panel.panel_id]._settings_form
        if form is None:
            return dict(kind="settings", panel=panel.panel_id)
        allowed = []
        for field in form.spec.fields:
            widget = form.widget_for(field.key)
            if not widget.isEnabled():
                continue
            if field.key in {"cell_kind", "size", "interval_ms", "semantic__reduction",
                "display__relim_mode", "display__x_relim_mode", "display__show_grid",
                "display__show_colorbar", "display__uncertainty", "display__colormap",
                "display__x_scale", "display__y_scale", "display__density", "display__cumulative",
                "display__log_x", "display__log_y", "display__threshold_classifier",
                "fit__model"} or field.key.startswith("semantic__fate:"):
                allowed.append(field)
            elif field.key in {"display__window", "display__bin_count", "fit__expression"}:
                allowed.append(field)
        if not allowed:
            return dict(kind="settings", panel=panel.panel_id)
        field = rng.choice(allowed)
        if field.choices:
            values = [item.value for item in field.choices]
            if field.key == "size": values = [x for x in values if x in ("2x2", "4x4")]
            if field.key == "fit__model": values = [x for x in values if str(x) in ("_ParameterChoice.NONE", "gaussian_offset", "radial_gaussian_center", "anisotropic_gaussian_center", "histogram_gaussian", "bimodal_gaussian")]
            if not values:
                return dict(kind="settings", panel=panel.panel_id)
            value = rng.choice(values)
        elif field.kind == "bool":
            value = not form.read_value(field.key)
        elif field.key == "display__window":
            value = rng.choice((1, 2, 5, 10))
        elif field.key == "display__bin_count":
            value = rng.choice((2, 16, 64, 128, 256))
        else:
            value = rng.choice(("", "A=guess(20)", "not an expression", "B=0"))
        action.update(field=field.key, value=value)
    elif kind == "gesture":
        action.update(gesture=rng.choice(("double", "area", "click", "wheel", "pan", "escape", "cancel", "clim")),
                      x=rng.uniform(.2,.65), y=rng.uniform(.2,.65),
                      dx=rng.uniform(-.2,.3), dy=rng.uniform(-.2,.3),
                      steps=rng.choice((-2,-1,1,2)))
    elif kind == "tab":
        action["index"] = rng.randrange(view.tabs.count())
    elif kind == "scope":
        form = bench.view._cards[panel.panel_id]._settings_form
        fields = [] if form is None else [field for field in form.spec.fields
            if field.cycle_choices is not None and len(field.cycle_choices)
            and form.widget_for(field.key).isEnabled()]
        if not fields:
            return dict(kind="settings", panel=panel.panel_id)
        action.update(field=rng.choice(fields).key, steps=rng.choice((-3, -1, 1, 3)))
    elif kind == "source":
        sources = [name for name in bench.view._cards[panel.panel_id]._signal_runtime.choice_names("signal")
                   if name != panel.state.signal]
        if not sources:
            return dict(kind="settings", panel=panel.panel_id)
        action["source"] = rng.choice(sources)
    elif kind == "save":
        action["name"] = f"figure-{step}"
    return action


def perform_action(bench, action, beat, output):
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    view = bench.view._view
    kind = action["kind"]
    def signal_name(spelling):
        if not spelling.startswith("$"):
            return spelling
        alias, leaf = spelling[1:].split("/", 1)
        return f"@logic/{bench.nodes[alias]}/{leaf}"

    if kind in ("review_script", "manual_script"):
        from zlc_ui.console.point_review_view import PointReviewView
        from zlc_ui.fluent import FluentCardDialog, FluentSectionLabel
        manual = kind == "manual_script"
        manual_node = bench.nodes.get(action.get("node"), action.get("node"))
        answered_request_id = None
        pending = list(action["actions"])
        deadline = time.monotonic() + float(action.get("timeout", 60))
        timer = QtCore.QTimer(bench.view._view)
        timer.setSingleShot(True)
        bench.dialog_timers.append(timer)
        def advance():
            nonlocal answered_request_id
            review = None if manual else next((w for w in bench.app.allWidgets()
                           if isinstance(w, PointReviewView) and w.isVisible()), None)
            manual_dialog = None
            try:
                if manual:
                    binding = bench.presenter.logic.get(manual_node)
                    host = None if binding is None else binding.host
                    request = None if host is None else host.operator_request
                    dialog = bench.app.activeModalWidget()
                    if (request is not None and request.kind == "manual-axis"
                            and isinstance(dialog, FluentCardDialog) and dialog.isVisible()
                            and any(label.text() == request.title
                                    for label in dialog.findChildren(FluentSectionLabel))):
                        manual_dialog = dialog
                if time.monotonic() >= deadline:
                    raise AssertionError(f"{kind} did not finish before its deadline")
                if manual:
                    if manual_dialog is None or request.request_id == answered_request_id:
                        timer.start(50)
                        return
                    item = pending[0]
                    for key in ("axis", "value", "point", "points"):
                        if key in item and request.payload.get(key) != item[key]:
                            raise AssertionError(f"manual request {key} differs: {request.payload.get(key)!r}")
                    label = str(item["button"])
                    if label not in ("Continue", "Stop"):
                        raise ValueError("manual button must be Continue or Stop")
                    buttons = [button for button in manual_dialog.findChildren(QtWidgets.QAbstractButton)
                               if button.text() == label and button.isVisible() and button.isEnabled()]
                    if len(buttons) != 1:
                        raise AssertionError("manual dialog has no unique usable answer button")
                    answered_request_id = request.request_id
                    click(buttons[0], bench.app)
                    bench.dialog_results.append(dict(action=item, status="completed", node=manual_node,
                        request_id=request.request_id,
                        payload={key: request.payload.get(key) for key in ("axis", "value", "point", "points")}))
                    pending.pop(0)
                    if pending:
                        timer.start(int(1000 * float(item.get("delay", .1))))
                    return
                if review is None:
                    timer.start(50)
                    return
                item = pending[0]
                if item["kind"] in ("review_point", "review_rectangle"):
                    from zlc_plot.point_review import ImagePointReviewSurface
                    surface = review.findChild(ImagePointReviewSurface)
                    if surface is None or surface._plot.presented_front is None:
                        timer.start(50)
                        return
                perform_artifact_action(bench, item, beat, output,
                    click=click, choose=choose, enter_text=enter_text)
                bench.dialog_results.append(dict(action=item, status="completed"))
                pending.pop(0)
                if pending:
                    timer.start(int(1000 * float(item.get("delay", .1))))
            except Exception:
                bench.dialog_results.append(dict(status="failed", error=traceback.format_exc()))
                if manual_dialog is not None and bench.app.activeModalWidget() is manual_dialog:
                    # Reject only the matched live manual modal, never a
                    # child QWidget or an unrelated warning/review window.
                    QtTest.QTest.keyClick(manual_dialog, QtCore.Qt.Key_Escape)
                elif review is not None:
                    review.window().close()
        timer.timeout.connect(advance)
        timer.start(50)
        return

    if kind == "signal_mark":
        publication = bench.session.signal_plane.latest_publication(signal_name(action["source"]))
        if publication is None:
            raise AssertionError("cannot mark a signal which has not published")
        bench.marks[action["name"]] = publication.event_ref
        return
    if kind == "dataset_check":
        import numpy as np
        from zlc_data import REPEAT, SCAN_POINT, SITE
        from zlc_runtime import DatasetCoverage

        source = signal_name(action["source"])
        plane = bench.session.signal_plane
        publication = plane.latest_publication(source)
        if action.get("absent"):
            facts = {"source": source, "absent": publication is None,
                     "retained": plane.retains(source), "live": plane.is_generation_live(source)}
            bench.dataset_checks.append(facts)
            assert facts["absent"] and not facts["retained"] and not facts["live"], facts
            return
        if publication is None:
            raise AssertionError("Dataset check requires an actual publication")
        snapshot = plane.current_dataset(source, publication)
        value = publication.value(source)
        schema = snapshot.block.schema

        def ref_key(ref):
            return {"stream": ref.stream_id.value, "generation": ref.generation.value,
                    "sequence": ref.sequence}

        facts = {"source": source, "event": ref_key(publication.event_ref),
                 "shape": list(snapshot.block.values.shape), "domains": [],
                 "direct_parents": [ref_key(ref) for ref in publication.direct_parent_refs]}
        bench.dataset_checks.append(facts)
        for domain in (schema.repeat_domain, schema.point_domain, schema.cell_domain):
            axes = []
            for axis in domain.axes:
                item = {"id": axis.axis_id.value, "name": axis.name, "size": axis.size,
                        "unit": axis.unit, "role": axis.role.value,
                        "coordinates_first": [axis.coordinate_at(i) for i in range(min(axis.size, 12))]}
                # Dense image pixel domains stay implicit; never expand their
                # coordinate grid for a diagnostic summary.
                if domain.axis_codes is not None:
                    codes = domain.codes(axis.axis_id)
                    item.update(codes_count=len(codes), codes_first=np.asarray(codes[:12]).tolist(),
                                codes_last=np.asarray(codes[-12:]).tolist())
                axes.append(item)
            facts["domains"].append(axes)
        coverage = value.coverage
        facts["coverage"] = (None if coverage is None else {
            "kind": type(coverage).__name__, "written": coverage.written_cells,
            "total": coverage.total_cells, "complete": coverage.complete})
        budget = int(action.get("max_values", 100_000))
        valid = None
        if snapshot.block.values.size <= budget:
            valid = np.asarray(snapshot.expanded_validity())
            facts.update(valid=int(np.count_nonzero(valid)), elements=int(valid.size))
        else:
            facts["validity_unchecked"] = "Dataset exceeds small-result budget"
        if "shape" in action and facts["shape"] != action["shape"]:
            raise AssertionError(f"unexpected canonical Dataset: {facts}")
        if "same_generation_as" in action:
            marked = bench.marks[action["same_generation_as"]]
            assert (publication.event_ref.stream_id, publication.event_ref.generation) == (
                marked.stream_id, marked.generation), "Dataset changed generation unexpectedly"
        if "expected_coverage" in action:
            assert isinstance(coverage, DatasetCoverage), "Expected finite Dataset coverage"
            assert [coverage.written_cells, coverage.total_cells] == action["expected_coverage"], facts
        if "root_generation_as" in action:
            # Generic finite-data audit: use the recorded placements and
            # causal parents, never infer a multi-axis Scan execution plan.
            assert isinstance(coverage, DatasetCoverage) and valid is not None
            assert schema == value.canonical_schema, "Expected canonical finite Dataset"
            expected_root = bench.marks[action["root_generation_as"]]
            parent_source = signal_name(action["parent_source"])
            with plane._lock:
                state = plane._state_for_signal_locked(source)
                assert state is not None and state.generation == publication.event_ref.generation
                retained = state.commit_chunks.get(source, ())
                assert len(retained) <= int(action.get("max_events", 2_000)), "Finite ledger exceeds audit budget"
                chunks = tuple(entry for entry in retained if entry[0] <= publication.event_ref.sequence)
            assert chunks and tuple(parent.event_ref for parent in chunks[-1][3]) == publication.direct_parent_refs
            written = np.zeros((schema.repeat_domain.size, schema.point_domain.size), dtype=bool)
            for _sequence, event, origin, parents in chunks:
                assert len(parents) == 1, "Expected one exact source publication per event"
                parent = parents[0]
                actual_source = parent.value(parent_source)
                assert actual_source is not None, "Exact parent lacks the declared source"
                roots = plane.publication_roots(parent)
                assert roots and {(root.stream_id, root.generation) for root in roots} == {
                    (expected_root.stream_id, expected_root.generation)}, "Finite prefix spliced source generations"
                assert np.array_equal(event.values, actual_source.values, equal_nan=True)
                event_valid = event.snapshot.expanded_validity()
                assert np.array_equal(event_valid, actual_source.snapshot.expanded_validity())
                r, p = origin
                rows, points = event.shape[:2]
                address = (slice(r, r + rows), slice(p, p + points))
                assert not written[address].any(), "Finite event placements overlap"
                written[address] = True
                assert np.array_equal(snapshot.block.values[address], event.values, equal_nan=True)
                assert np.array_equal(valid[address], event_valid)
            assert int(np.count_nonzero(written)) == coverage.written_cells
            assert not valid[~written].any(), "Unwritten finite Dataset cells became valid"
            facts.update(lineage_status="checked_full_prefix", audited_events=len(chunks),
                         root_generation={"stream": expected_root.stream_id.value,
                                          "generation": expected_root.generation.value},
                         unwritten_cells=int(np.count_nonzero(~written)))
        scan = action.get("scan")
        if scan is None:
            return
        # This is the user-requested one-axis Survival scan oracle, not a
        # second Scan writer. Read its actual fixed placement/parent ledger.
        assert isinstance(coverage, DatasetCoverage), "Scan lacks finite coverage"
        assert valid is not None, "Scan validity could not be checked within budget"
        assert schema == value.canonical_schema, "Scan display is not canonical"
        assert snapshot.block.values.dtype == np.dtype("?"), "Survival scan is not boolean"
        repeats, points, cells = schema.repeat_domain, schema.point_domain, schema.cell_domain
        sweeps, shots = int(scan["scan_repeats"]), int(scan["run_repeats"])
        coordinates = tuple(scan["coordinates"])
        assert len(repeats.axes) == 2 and all(axis.role == REPEAT for axis in repeats.axes)
        assert tuple(axis.axis_id.value for axis in repeats.axes) == ("scan.repeat", "pulse.run")
        assert tuple(axis.size for axis in repeats.axes) == (sweeps, shots)
        assert np.array_equal(repeats.codes(repeats.axes[0].axis_id), np.repeat(np.arange(sweeps), shots))
        assert np.array_equal(repeats.codes(repeats.axes[1].axis_id), np.tile(np.arange(shots), sweeps))
        assert len(points.axes) == 2 and points.axes[1].role == SCAN_POINT
        scan_axis = points.axes[1]
        assert scan_axis.axis_id.value == scan["axis"] and scan_axis.unit == scan["unit"]
        assert tuple(scan_axis.coordinate_at(i) for i in range(scan_axis.size)) == coordinates
        assert len(cells.axes) == 1 and cells.axes[0].role == SITE
        assert coverage.complete == bool(scan["complete"])
        if not scan["complete"]:
            assert 0 < coverage.written_cells < coverage.total_cells, "Scan did not stop at a partial prefix"
        expected_source = signal_name(scan["source"])
        record = publication.run_record
        assert (record["scan_repeats"], record["run_repeats"]) == (sweeps, shots)
        assert record["source_signal"] == expected_source
        assert len(record["plan"]["axes"]) == 1
        assert tuple(record["plan"]["axes"][0]["values"]) == coordinates
        # Public live replay refuses sealed generations. The diagnostic alone
        # reads the EXISTING finite ledger under its owner lock, retaining a
        # bounded local tuple only until this action returns. No tap/observer,
        # extra history lease, queue or cross-action payload retention.
        with plane._lock:
            state = plane._state_for_signal_locked(source)
            assert state is not None and state.generation == publication.event_ref.generation
            retained = state.commit_chunks.get(source, ())
            assert len(retained) <= int(action.get("max_events", 2_000)), "Scan ledger exceeds audit budget"
            chunks = tuple(entry for entry in retained if entry[0] <= publication.event_ref.sequence)
        assert chunks, "Scan has no retained exact commit ledger"
        assert tuple(parent.event_ref for parent in chunks[-1][3]) == publication.direct_parent_refs
        root_generations = set()
        expected_root = (bench.marks[scan["root_generation_as"]]
                         if "root_generation_as" in scan else None)
        source_schema = None
        for index, (_sequence, event, origin, parents) in enumerate(chunks):
            assert len(parents) == 1, "Scan event needs its one actual source publication"
            parent = parents[0]
            actual_source = parent.value(expected_source)
            assert actual_source is not None, "Scan exact parent does not contain the declared source"
            if source_schema is None:
                source_schema = actual_source.schema
            assert actual_source.schema == source_schema, "Scan spliced different source schemas"
            # Traverse real causal parents, not child/source revision numbers.
            try:
                roots = plane.publication_roots(parent)
            except (LookupError, RuntimeError) as error:
                facts["lineage_status"] = "unchecked"
                raise AssertionError("exact Scan parent payload is unavailable") from error
            assert len(roots) == 1, "This Camera chain has no unique exact root"
            for root in roots:
                root_generations.add((root.stream_id.value, root.generation.value))
                if expected_root is not None:
                    assert (root.stream_id, root.generation) == (
                        expected_root.stream_id, expected_root.generation), "Scan spliced another Camera generation"
            source_points = source_schema.point_domain.size
            sweep, rest = divmod(index, len(coordinates) * shots)
            row, shot = divmod(rest, shots)
            expected_origin = (sweep * shots + shot, row * source_points)
            assert origin == expected_origin == event.cell_origin, "Scan commit placement differs from played order"
            assert event.coverage.written_cells == (index + 1) * source_points
            assert event.coverage.total_cells == sweeps * shots * len(coordinates) * source_points
            assert np.array_equal(event.values, actual_source.values), "Scan event changed source booleans"
            event_valid = event.snapshot.expanded_validity()
            assert np.array_equal(event_valid, actual_source.snapshot.expanded_validity()), "Scan event changed source denominator"
            r, p = origin
            target = (slice(r, r + 1), slice(p, p + source_points), slice(None))
            assert np.array_equal(snapshot.block.values[target], event.values), "Canonical placement differs from its exact event"
            assert np.array_equal(valid[target], event_valid), "Canonical validity differs from its exact event"
        assert len(root_generations) == 1, "Scan prefix contains multiple source generations"
        assert source_schema.repeat_domain.size == 1
        assert points.axes[0] == source_schema.point_domain.axes[0] and cells == source_schema.cell_domain
        source_points = source_schema.point_domain.size
        assert repeats.size == sweeps * shots and points.size == len(coordinates) * source_points
        assert np.array_equal(points.codes(points.axes[0].axis_id),
                              np.tile(source_schema.point_domain.codes(points.axes[0].axis_id), len(coordinates)))
        assert np.array_equal(points.codes(scan_axis.axis_id), np.repeat(np.arange(len(coordinates)), source_points))
        assert coverage.written_cells == len(chunks) * source_points
        assert coverage.total_cells == sweeps * shots * len(coordinates) * source_points
        r = np.arange(repeats.size)[:, None]
        p = np.arange(points.size)[None, :]
        order = (r // shots) * (len(coordinates) * shots) + (p // source_points) * shots + r % shots
        future = order >= len(chunks)
        assert not valid[future].any(), "Unwritten Scan cells became valid"
        facts.update(scan_status="checked", lineage_status="checked_full_prefix",
                     audited_events=len(chunks), future_cells=int(np.count_nonzero(future)),
                     root_generations=[{"stream": stream, "generation": generation}
                                       for stream, generation in sorted(root_generations)])
        return
    if kind == "signal_wait":
        source = signal_name(action["source"])
        before = bench.marks.get(action.get("after"))
        def ready():
            publication = bench.session.signal_plane.latest_publication(source)
            if publication is None:
                return False
            if before is None:
                return True
            if action.get("new_generation"):
                return publication.event_ref.generation != before.generation
            return publication.event_ref != before
        if not beat.run_until(ready, float(action.get("timeout", 10))):
            raise AssertionError(f"signal did not make the requested progress: {source}")
        if action.get("same_generation") and before is not None:
            assert bench.session.signal_plane.latest_publication(source).event_ref.generation == before.generation
        return
    if kind == "science_check":
        signals = {key: signal_name(value) for key, value in action["signals"].items()}
        answer = check_scientific_chain(bench.session.signal_plane, **signals)
        bench.science_checks.append(answer)
        if answer["status"] == "failed":
            raise AssertionError(f"scientific publication mismatch: {answer['findings']}")
        if answer["status"] == "unchecked" and not action.get("allow_unchecked", False):
            raise AssertionError(f"scientific check could not be completed: {answer['sections']}")
        return
    if kind == "overlay_check":
        panel = bench.presenter.panels[action["panel"]]
        before = bench.overlay_checks[-1] if bench.overlay_checks else None
        for _ in range(int(action.get("samples", 1))):
            answer = check_overlay_pixels(panel, bench.view._cards[panel.panel_id], before=before)
            bench.overlay_checks.append(answer)
            if answer["status"] == "failed":
                raise AssertionError(f"overlay pixels disagree: {answer['findings']}")
            before = answer
            beat.run(float(action.get("interval", .1)))
        return
    if kind.startswith(("device_", "pulse_")):
        return perform_device_action(bench, action, beat, output,
            click=click, choose=choose, enter_text=enter_text)
    if kind.startswith(("review_", "slm_", "viewer_")):
        return perform_artifact_action(bench, action, beat, output,
            click=click, choose=choose, enter_text=enter_text)
    if kind == "logic_add":
        hide_settings(bench)
        choose(view.kind_combo, ("logic", action["api"]), bench.app)
        before = set(bench.presenter.logic)
        click(view.add_panel_button, bench.app)
        if not beat.run_until(lambda: set(bench.presenter.logic) != before, 3):
            raise AssertionError("Add Logic did not create a node")
        created = set(bench.presenter.logic) - before
        if len(created) != 1:
            raise AssertionError("Add Logic must create exactly one node")
        identity = created.pop()
        bench.nodes[action.get("as", action["api"])] = identity
        if action["api"] == "camera_measurement":
            bench.node = identity
            bench.signal = f"@logic/{identity}/frames"
        return
    if kind.startswith("logic_"):
        identity = bench.nodes.get(action["node"], action["node"])
        if kind == "logic_wait":
            from collections.abc import Mapping
            wanted = action.get("phase", "done")
            def ready():
                host = bench.presenter.logic[identity].host
                if host is None:
                    return False
                observation = host.observation
                if observation.phase == "failed" and wanted != "failed":
                    raise AssertionError(f"{identity}: {observation.error}")
                return observation.phase == wanted
            if not beat.run_until(ready, float(action.get("timeout", 60))):
                raise AssertionError(f"{identity} did not reach {wanted}")
            result = bench.presenter.logic[identity].host.final_result
            def result_field(key):
                return result[key] if isinstance(result, Mapping) else getattr(result, key)
            for alias, attribute in action.get("remember", {}).items():
                bench.artifacts[alias] = str(result_field(attribute))
            for key, expected in action.get("expect_result", {}).items():
                if result_field(key) != expected:
                    raise AssertionError(f"{identity} result {key}={result_field(key)!r}, expected {expected!r}")
            bench.logic_checks.append(dict(node=identity, phase=wanted,
                error=bench.presenter.logic[identity].host.observation.error,
                result={key: str(result_field(key)) for key in
                        (*action.get("remember", {}).values(), *action.get("expect_result", {}))}))
            return
        if kind == "logic_stop" and view._task_takeover:
            active = bench.presenter._active_task()
            if active is None or active.node_id != identity:
                raise AssertionError("Stop task would target a different running Task")
            if action.get("expect_running") and not active.host.running:
                raise AssertionError("Task already finished before the intended Stop")
            click(view.status_strip.action_button, bench.app)
            return
        editor = bench.view._logic_editors.get(identity)
        if editor is None:
            tab(bench, 1)
            click(bench.view._rows[identity].edit_button, bench.app)
            editor = bench.view._logic_editors[identity]
        # A running Task disables the row's Edit action; its already-open tab
        # still owns Stop. Select that real tab instead of reopening the editor.
        tab(bench, view.tabs.indexOf(editor))
        if kind in ("logic_start", "logic_stop", "logic_remove"):
            if action.get("expect_running") and not bench.presenter.logic[identity].host.running:
                raise AssertionError("the intended in-flight action was too late: node is already terminal")
            report_start = len(bench.reports)
            click(getattr(editor, kind.removeprefix("logic_") + "_button"), bench.app)
            if "expect_report" in action:
                wanted = str(action["expect_report"])
                if not beat.run_until(lambda: any(wanted in text for _severity, text
                                      in bench.reports[report_start:]), 3):
                    raise AssertionError(f"expected refusal {wanted!r}, got {bench.reports[report_start:]}")
            if action.get("expect_no_host") and bench.presenter.logic[identity].host is not None:
                raise AssertionError("refused Start unexpectedly created a run host")
        elif kind == "logic_source":
            source = action["source"]
            if source.startswith("$"):
                alias, output_name = source[1:].split("/", 1)
                source = f"@logic/{bench.nodes[alias]}/{output_name}"
            choose(editor.source_combo, source, bench.app)
        elif kind == "logic_device":
            choose(editor._device_combos[action["role"]], action["device"], bench.app)
        elif kind in ("logic_field", "logic_artifact"):
            form = editor.form if kind == "logic_field" else editor.artifact_form
            widget = form.widget_for(action["field"])
            value = action["value"]
            if isinstance(value, str) and value.startswith("$workspace/"):
                value = str(bench._tmp / value.removeprefix("$workspace/"))
            elif isinstance(value, str) and value.startswith("$artifact/"):
                value = bench.artifacts[value.removeprefix("$artifact/")]
            if hasattr(widget, "_popup_view"):
                choose(widget, value, bench.app)
            elif isinstance(widget, QtWidgets.QAbstractButton):
                if widget.isChecked() != value:
                    click(widget, bench.app)
            else:
                enter_text(widget, value, bench.app)
        else:
            raise ValueError(f"unknown logic action: {kind}")
        return
    if kind.startswith("scan_"):
        from zlc_atom.nodes.scan.editor import ScanPlanEditor
        identity = bench.nodes.get(action["node"], action["node"])
        tab(bench, 1)
        click(bench.view._rows[identity].edit_button, bench.app)
        editor = bench.view._logic_editors[identity]
        scan = next(value for value in editor._contributions.values()
                    if isinstance(value, ScanPlanEditor))
        if kind == "scan_add_axis":
            click(scan.add_manual_button if action.get("manual") else scan.add_button, bench.app)
        elif kind == "scan_axis":
            index = int(action.get("index", 0))
            if "port" in action:
                choose(scan._rows[index].port_combo, action["port"], bench.app)
            if "name" in action:
                enter_text(scan._rows[index].name_edit, action["name"], bench.app)
            if "unit" in action:
                choose(scan._rows[index].unit_picker, action["unit"], bench.app)
            for key, attribute in (("start", "start_spin"), ("stop", "stop_spin"), ("points", "points_spin")):
                if key in action:
                    enter_text(getattr(scan._rows[index], attribute), action[key], bench.app)
        elif kind == "scan_remove_axis":
            click(scan._rows[int(action.get("index", 0))].remove_button, bench.app)
        else:
            raise ValueError(f"unknown scan action: {kind}")
        return
    if kind == "wait":
        beat.run(float(action.get("seconds", 0.2)))
        return
    if kind == "roi":
        return roi_drag(bench, action, beat)
    if kind == "choose_plot":
        choose(view.kind_combo, ("plot", action["plot"]), bench.app)
        return
    if kind == "add":
        hide_settings(bench)
        choose(view.kind_combo, ("plot", action["plot"]), bench.app)
        before = set(bench.presenter.panels)
        click(view.add_panel_button, bench.app)
        beat.run_until(lambda: set(bench.presenter.panels) != before, 2.0)
        new = set(bench.presenter.panels)-before
        if len(new) != 1: raise AssertionError("Add Panel did not add exactly one card")
        identity = next(iter(new))
        _card, form = settings(bench, identity)
        choose(form.widget_for("signal"), action.get("source", bench.signal), bench.app)
        return
    if kind == "field":
        field_edit(bench, action)
    elif kind == "selectors":
        click(view.selectors_switch, bench.app)
    elif kind == "pause":
        click(view.pause_button, bench.app)
    elif kind == "tab":
        tab(bench, action["index"])
    elif kind == "settings":
        tab(bench, 0)
        card = bench.view._cards[action["panel"]]
        shown = card._settings_popup is not None and card._settings_popup.isVisible()
        click(card.settings_button, bench.app)
        bench.app.processEvents()
        if card._settings_popup is None or card._settings_popup.isVisible() == shown:
            raise AssertionError("the Setting button did not toggle its in-page frame")
    elif kind == "close_settings":
        hide_settings(bench)
    elif kind == "source":
        _card, form = settings(bench, action["panel"])
        choose(form.widget_for("signal"), action["source"], bench.app)
    elif kind == "overlay":
        _card, form = settings(bench, action["panel"])
        choose(form.widget_for("overlay_signal"), signal_name(action["source"]), bench.app)
    elif kind == "scope":
        _card, form = settings(bench, action["panel"])
        if action["field"] not in form.keys:
            return "scope field replaced by preceding edit"
        combo = form.widget_for(action["field"])
        click(combo, bench.app)
        popup = combo._popup_view
        index = popup.model().index(combo._cycle_row, 0)
        if not index.isValid():
            return "scope action changed before the click"
        popup.scrollTo(index)
        QtTest.QTest.mouseClick(popup.viewport(), QtCore.Qt.LeftButton, pos=popup.visualRect(index).center())
        bench.app.processEvents()
        pointer = Pointer(combo, bench.app, QtCore, QtGui)
        pointer.post = True
        pointer.wheel(.5, .5, action["steps"])
    elif kind in ("stop", "restart"):
        tab(bench, 1)
        row = bench.view._rows[bench.node]
        button = row.stop_button if kind == "stop" else row.start_button
        click(button, bench.app)
    elif kind == "remove":
        tab(bench, 0)
        hide_settings(bench)
        card = bench.view._cards[action["panel"]]
        click(card.close_button, bench.app)
        click(card.close_button, bench.app)
    elif kind == "edit":
        card, _form = settings(bench, action["panel"])
        click(card.edit_button, bench.app)
    elif kind == "save":
        card, _form = settings(bench, action["panel"])
        click(card.edit_button, bench.app)
        beat.run_until(lambda: action["panel"] in bench.view._panel_editors, 3.0)
        editor = bench.view._panel_editors.get(action["panel"])
        if editor is None:
            return "no accepted snapshot to edit"
        enter_text(editor.save_directory.edit, str(output), bench.app)
        if editor.save_auto_name.isChecked():
            click(editor.save_auto_name, bench.app)
        enter_text(editor.save_name, action["name"], bench.app)
        if not editor.save_button.isEnabled():
            return "save waiting for accepted editor surface"
        frozen = bench.presenter.panels[action["panel"]].frozen_data
        bench.fuzz_saves.append((output / f"{action['name']}.npz",
                                 frozen.snapshot.ref, tuple(frozen.description.selectors)))
        click(editor.save_button, bench.app)
    elif kind == "gesture":
        tab(bench, 0)
        hide_settings(bench)
        panel = bench.presenter.panels[action["panel"]]
        widget = bench.surface(panel)
        if widget is None or widget.presented_front is None:
            return "surface pending"
        if not visible(widget, bench.app): return "surface not visible"
        if not widget.interaction_enabled:
            click(view.selectors_switch, bench.app)
        axes = data_axes(widget)
        if not axes: return "no data axes"
        axis = axes[0]
        left, top, right, bottom = axis.bounds
        x, y = left+(right-left)*action["x"], top+(bottom-top)*action["y"]
        pointer = Pointer(widget, bench.app, QtCore, QtGui)
        pointer.post = True
        gesture = action["gesture"]
        if gesture == "clim":
            rail = axis_by_role(widget.presented_front, "distribution")
            limits = widget.presented_front.interaction.color_limits
            if rail is None or limits is None:
                return "surface has no color-limit rail"
            mid = sum(rail.x_limits) / 2
            previewed = False
            pointer.press(*rail.display_to_normalized(mid, limits.high))
            beat.run(.02)
            for part in range(1, 7):
                point = rail.display_to_normalized(mid, limits.high - limits.span * part / 50)
                pointer.move(*point)
                beat.run(.015)
                current = widget.presented_front.interaction.color_limits
                previewed |= current is not None and current != limits
            bench.fuzz_gestures.append(dict(panel=action["panel"], kind="clim",
                changed_while_held=previewed))
            pointer.release(*point)
            return
        if gesture == "double": pointer.dclick(x,y)
        elif gesture == "wheel": pointer.wheel(x,y,action["steps"])
        elif gesture == "escape": QtTest.QTest.keyClick(widget,QtCore.Qt.Key_Escape)
        else:
            button = QtCore.Qt.MiddleButton if gesture == "pan" else QtCore.Qt.LeftButton
            pointer.press(x,y,button)
            if gesture != "click":
                for offset in range(1,6):
                    pointer.move(x+(right-left)*action["dx"]*offset/5,
                                 y+(bottom-top)*action["dy"]*offset/5)
                    beat.run(.003)
                if gesture == "cancel":
                    tab(bench, 1)
                x += (right-left)*action["dx"]
                y += (bottom-top)*action["dy"]
            pointer.release(x,y,button)


def run_child(args, output):
    import zlc_workbench
    from PyQt5 import QtCore, QtTest
    print("ROOT", zou_lab_control.__file__, "WORKBENCH", zlc_workbench.__file__, flush=True)
    rng = random.Random(args.seed)
    errors = []
    journal = (output / "actions.jsonl").open("w", encoding="utf-8", buffering=1)
    probes = (output / "display-events.jsonl").open("w", encoding="utf-8", buffering=1)
    probe_cleanup = None
    current_step = -1
    def observe(event):
        probes.write(json.dumps(dict(step=current_step, **event), ensure_ascii=False) + "\n")
    previous_hook = sys.excepthook
    error_log = logging.Handler(level=logging.ERROR)
    def logged_error(record):
        # Never format arbitrary logging arguments: a raster operation can
        # contain an entire RGBA buffer. Keep the failure and traceback only.
        error = None if not record.exc_info else record.exc_info[1]
        stack = []
        trace = None if not record.exc_info else record.exc_info[2]
        while trace is not None and len(stack) < 32:
            code = trace.tb_frame.f_code
            stack.append((code.co_filename, trace.tb_lineno, code.co_name))
            trace = trace.tb_next
        errors.append({"logger": record.name,
                       "message": record.msg[:512] if isinstance(record.msg, str) else type(record.msg).__name__,
                       "exception": None if error is None else type(error).__name__,
                       "exception_message": next((arg[:512] for arg in getattr(error, "args", ()) if isinstance(arg, str)), ""),
                       "stack": stack})
    error_log.emit = logged_error
    logging.getLogger().addHandler(error_log)

    def exception_hook(kind, value, tb):
        message = "".join(traceback.format_exception(kind, value, tb))
        errors.append(message)
        journal.write(json.dumps(dict(event="unhandled_exception", message=message)) + "\n")
    sys.excepthook = exception_hook
    result = dict(seed=args.seed, requested_actions=args.actions, completed_actions=0,
                  frames_per_cycle=args.frames, errors=errors)
    try:
        with ConsoleBench() as bench:
            bench.feedback_scope_probe = bool(getattr(args, "feedback_scope_probe", False))
            bench.fuzz_saves = []
            bench.fuzz_gestures = []
            bench.nodes = {}
            bench.artifacts = {}
            bench.marks = {}
            bench.science_checks = []
            bench.overlay_checks = []
            bench.dataset_checks = []
            bench.logic_checks = []
            bench.dialog_timers = []
            bench.dialog_results = []
            if args.system:
                bench.open_devices()
                click(bench.flow.devices._view.lifecycle_button, bench.app)
                deadline = time.monotonic() + 30
                while bench.flow.console is None:
                    bench.app.processEvents()
                    if any(severity == "error" for severity, _text in bench.reports):
                        raise AssertionError(f"Device Init failed: {bench.reports[-1]}")
                    if time.monotonic() >= deadline:
                        raise AssertionError("Device Manager Init did not create TaskConsole")
                    QtTest.QTest.qWait(5)
                bench.adopt_initialized_flow()
            else:
                bench.start(camera="mot_camera", exposure=0.01,
                            frames_per_cycle=args.frames, clear_preview_panels=False)
            probe_cleanup = install_observers(bench, observe)
            bench._until(lambda: all(bench.surface(panel) is not None and bench.surface(panel).presented_front is not None
                         for panel in bench.presenter.panels.values()), "first real plot front", timeout=30)
            output.joinpath("workspace.txt").write_text(str(bench._tmp), encoding="utf-8")
            view = bench.view._view
            clock = []
            last = time.monotonic()
            timer = QtCore.QTimer()
            timer.setInterval(50)
            def heartbeat():
                nonlocal last
                now = time.monotonic()
                gap = (now-last)*1000
                clock.append(gap)
                if gap > 250:
                    journal.write(json.dumps(dict(event="slow_ui_heartbeat", step=current_step,
                        time_ns=time.perf_counter_ns(), milliseconds=gap)) + "\n")
                last = now
            timer.timeout.connect(heartbeat)
            timer.start()
            try:
                with ProductBeat(bench.app, bench.presenter, drive_timer=not args.system) as beat:
                    first = next(iter(bench.view._cards.values()), None)
                    if first is not None:
                        click(first.settings_button, bench.app)
                    beat.run(0.3)
                    if args.chain:
                        camera = next(iter(bench.presenter.panels))
                        setup = [dict(kind="roi", panel=camera, width=.07, height=.05)]
                        name = f"@logic/{camera}/roi_frame"
                        setup.extend(dict(kind="add", plot=kind, source=name)
                                     for kind in ("histogram", "facet_grid", "curve"))
                        for index, action in enumerate(setup):
                            journal.write(json.dumps(dict(event="setup.begin", index=index, action=action))+"\n")
                            perform_action(bench, action, beat, output)
                            if action["kind"] == "roi":
                                if not beat.run_until(lambda: bench.session.signal_plane.freeze().publication(name) is not None, 5):
                                    raise AssertionError("drawn ROI did not produce a signal")
                            else:
                                beat.run(.25)
                            journal.write(json.dumps(dict(event="setup.end", index=index))+"\n")
                        hide_settings(bench)
                        beat.run(.5)
                    last_state, initial_findings = checkpoint(bench, {}, stable=True)
                    result["initial_findings"] = initial_findings
                    write_json(output / "initial-state.json", last_state)
                    if args.inventory:
                        result["inventory"] = inventory(bench)
                        bench.view.save_screenshot(str(output / "inventory.png"))
                    else:
                        replay = json.loads(Path(args.replay).read_text(encoding="utf-8")) if args.replay else None
                        actions = []
                        result["findings"] = []
                        for step in range(args.actions if replay is None else len(replay)):
                            current_step = step
                            action = replay[step] if replay else pick_action(bench, rng, step)
                            # Apply the same plain action that the replay will
                            # read, including Qt's explicit '(none)' sentinel.
                            action = json.loads(json.dumps(action, default=str))
                            action.setdefault("delay", rng.choice((0.0, 0.01, 0.08, 0.25)))
                            actions.append(action)
                            write_json(output / "replay.json", actions)
                            journal.write(json.dumps(dict(event="begin", step=step, action=action)) + "\n")
                            faulthandler.dump_traceback_later(max(15, float(action.get("timeout", 0)) + 5))
                            started = time.perf_counter()
                            try:
                                skipped = perform_action(bench, action, beat, output)
                            except UnavailableAction as unavailable:
                                if replay is not None and not action.get("allow_unavailable", False):
                                    raise AssertionError(f"explicit UI action unavailable: {action}: {unavailable}") from unavailable
                                skipped = str(unavailable)
                            beat.run(action["delay"])
                            if any(item["status"] == "failed" for item in bench.dialog_results):
                                raise AssertionError(f"modal UI action failed: {bench.dialog_results[-1]}")
                            if errors:
                                raise AssertionError("unhandled application error was captured")
                            if args.stop_on_report:
                                matched = next((message for severity, message in bench.reports
                                                if args.stop_on_report in message), None)
                                if matched is not None:
                                    result["matched_report"] = matched
                                    raise AssertionError(matched)
                            if step % 5 == 4:
                                last_state, findings = checkpoint(bench, last_state, stable=False)
                                result["findings"].extend(dict(step=step, **item) for item in findings)
                                journal.write(json.dumps(dict(event="checkpoint", step=step, findings=findings)) + "\n")
                            journal.write(json.dumps(dict(event="end", step=step, skipped=skipped,
                                milliseconds=1000*(time.perf_counter()-started),
                                panel_conditions={key:panel.reported_condition for key,panel in bench.presenter.panels.items() if panel.reported_condition})) + "\n")
                            result["completed_actions"] = step+1
                            faulthandler.cancel_dump_traceback_later()
                        write_json(output / "replay.json", actions)
                        settled = beat.run_until(lambda: all(panel.configuration is None
                            and panel.editor_configuration is None
                            for panel in bench.presenter.panels.values())
                            and not bench.presenter._saving_panels, 10.0)
                        beat.run(0.2)
                        final_state, findings = checkpoint(bench, last_state, stable=settled)
                        write_json(output / "final-state.json", final_state)
                        result["findings"].extend(dict(step="final", **item) for item in findings)
                        result["settled"] = settled
                        bench.view.save_screenshot(str(output / "final.png"))
                        if args.stop_on_report:
                            matched = next((message for severity, message in bench.reports
                                            if args.stop_on_report in message), None)
                            if matched is not None:
                                result["matched_report"] = matched
                                raise AssertionError(matched)
                    result["reports"] = list(bench.reports)
                    result["report_events"] = list(bench.report_events)
                    result["gesture_checks"] = list(bench.fuzz_gestures)
                    result["inventory"] = inventory(bench)
                    result["nodes"] = dict(bench.nodes)
                    result["artifacts"] = dict(bench.artifacts)
                    result["science_checks"] = list(bench.science_checks)
                    result["overlay_checks"] = list(bench.overlay_checks)
                    result["dataset_checks"] = list(bench.dataset_checks)
                    result["event_loop_gap_ms"] = clock
                    result["status"] = "observation_completed"
            except Exception:
                result["status"] = "failed"
                result["failure"] = traceback.format_exc()
                result["reports"] = list(bench.reports)
                result["report_events"] = list(bench.report_events)
                bench.view.save_screenshot(str(output / "failure.png"))
                raise
            finally:
                timer.stop()
                for dialog_timer in bench.dialog_timers:
                    dialog_timer.stop()
                result["dialog_results"] = list(bench.dialog_results)
                result["logic_checks"] = list(bench.logic_checks)
                result["artifacts"] = dict(bench.artifacts)
                result["science_checks"] = list(bench.science_checks)
                result["overlay_checks"] = list(bench.overlay_checks)
                result["dataset_checks"] = list(bench.dataset_checks)
                if probe_cleanup is not None:
                    result["observer_canary"] = probe_cleanup()
                    probe_cleanup = None
        if errors:
            raise AssertionError("application error was captured, including during shutdown")
        # Read-back is evidence collection, not GUI work: keep decompression
        # outside the live application's lifetime and timing window.
        from zlc_data.figure_archive import read_archive, read_dataset
        from zlc_plot.figure_artifact import figure_plot_recipe
        result["saved_checks"] = []
        for path, expected_ref, expected_selectors in bench.fuzz_saves:
            check = dict(path=str(path), archive_exists=path.exists(),
                         preview_exists=path.with_suffix(".png").exists())
            if path.exists():
                info, arrays = read_archive(path)
                check["snapshot_matches_click"] = (
                    read_dataset(info, arrays, "data").ref == expected_ref)
                check["selectors_match_click"] = (
                    tuple(figure_plot_recipe(info, "data")["selectors"]) == expected_selectors)
                del arrays
                if not check["snapshot_matches_click"] or not check["selectors_match_click"]:
                    raise AssertionError(f"saved figure differs from the accepted click state: {path}")
            result["saved_checks"].append(check)
        result["needs_review"] = bool(
            any(severity == "error" for severity, _message in result.get("reports", ()))
            or result.get("findings")
            or result.get("observer_canary", {}).get("paint_data_mismatches")
        )
    except Exception:
        result["status"] = "failed"
        result["failure"] = traceback.format_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        sys.excepthook = previous_hook
        logging.getLogger().removeHandler(error_log)
        error_log.close()
        if probe_cleanup is not None:
            result["observer_canary"] = probe_cleanup()
        journal.close()
        probes.close()
        write_json(output / "result.json", result)


def file_dialog_action(button, path, app, *, click, enter_text, save=False, opener=None):
    """Use Qt's real file chooser; the native Windows shell is not this oracle."""
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

    attribute = QtCore.Qt.AA_DontUseNativeDialogs
    previous = app.testAttribute(attribute)
    errors, selected = [], []
    deadline = time.monotonic() + 5.0
    timer = QtCore.QTimer()

    def select_file():
        dialog = app.activeModalWidget()
        if not isinstance(dialog, QtWidgets.QFileDialog):
            if time.monotonic() < deadline:
                return
            timer.stop()
            errors.append(AssertionError("the Qt file chooser did not open"))
            if dialog is not None:
                QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
            return
        timer.stop()
        try:
            filename = dialog.findChild(QtWidgets.QLineEdit, "fileNameEdit")
            if filename is None:
                raise AssertionError("the Qt file chooser has no filename editor")
            click(filename, app)
            QtTest.QTest.keyClick(filename, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
            QtTest.QTest.keyClick(filename, QtCore.Qt.Key_Backspace)
            # enter_text commits Return; here the actual Open/Save button must
            # submit the dialog, so type without that final key.
            for character in str(path):
                app.sendEvent(filename, QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress, 0, QtCore.Qt.NoModifier, character))
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            submit = buttons.button(QtWidgets.QDialogButtonBox.Save if save
                                    else QtWidgets.QDialogButtonBox.Open)
            click(submit, app)
            selected.append(str(path))
        except Exception as error:
            errors.append(error)
            QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)

    timer.timeout.connect(select_file)
    try:
        app.setAttribute(attribute, True)
        timer.start(20)
        opened = opener() if opener is not None else click(button, app)
        if errors:
            raise errors[0]
        if not selected:
            raise AssertionError("the file chooser was not submitted")
        return opened
    finally:
        timer.stop()
        timer.deleteLater()
        app.setAttribute(attribute, previous)


def perform_device_action(bench, action, beat, output, *, click, choose, enter_text):
    """Exercise the existing Device Manager and device-owned Control widgets."""
    from PyQt5 import QtCore, QtTest, QtWidgets

    flow = getattr(bench, "flow", None)
    if flow is None or flow.devices is None:
        raise UnavailableAction("device actions require the initialized --system flow")
    kind = action["kind"]
    device = str(action.get("device", "sequencer"))
    device = getattr(bench, "fuzz_devices", {}).get(device, device)
    manager = flow.devices._view
    if kind == "device_checkpoint":
        cards = {key: {"type": card.type_combo.currentData(), "role": card.role_edit.text(),
                       "fields": [] if card.form is None else list(card.form.keys)}
                 for key, card in manager._cards.items()}
        leaves = {} if flow.session is None else flow.session.installation.devices
        facts = {"cards": cards, "loaded": {key: leaf.type_id for key, leaf in leaves.items()},
                 "lifecycle": manager.lifecycle_button.text(), "controls": list(flow.device_controls),
                 "simulation": dict(flow.devices.presenter.simulation),
                 "remote_enabled": {key: card.remote_button.isEnabled()
                                    for key, card in manager._loaded_cards.items()}}
        if "simulation" in action:
            assert facts["simulation"] == action["simulation"], facts
        if "type" in action:
            assert cards[device]["type"] == action["type"], facts
        if "role" in action:
            assert cards[device]["role"] == action["role"], facts
        if "fields" in action:
            assert set(action["fields"]) <= set(cards[device]["fields"]), facts
            if action.get("exact_fields"):
                assert set(action["fields"]) == set(cards[device]["fields"]), facts
        if "types" in action:
            assert set(action["types"]) == {card["type"] for card in cards.values()}, facts
        if "loaded" in action:
            assert (device in leaves) == bool(action["loaded"]), facts
        if "remember" in action:
            if not hasattr(bench, "fuzz_device_objects"):
                bench.fuzz_device_objects = {}
            bench.fuzz_device_objects[action["remember"]] = (
                flow.session, {key: leaf.device for key, leaf in leaves.items()})
        if "same_as" in action:
            old_session, old_devices = bench.fuzz_device_objects[action["same_as"]]
            assert flow.session is old_session, "device reconcile replaced the experiment session"
            keys = action.get("unchanged", tuple(old_devices))
            facts["same_device_objects"] = {
                key: key in leaves and leaves[key].device is old_devices[key] for key in keys}
            assert all(facts["same_device_objects"].values()), facts
            for alias in action.get("changed", ()):
                key = getattr(bench, "fuzz_devices", {}).get(alias, alias)
                assert key in leaves and leaves[key].device is not old_devices[key], facts
        control = flow.device_controls.get(device)
        view = getattr(control, "_view", None)
        if view is not None and hasattr(view, "_field_states"):
            expected_current = action.get("current", {})
            if expected_current and not beat.run_until(lambda: all(
                    view._field_states.get(field, {}).get("current") == expected
                    for field, expected in expected_current.items()), float(action.get("timeout", 5))):
                raise AssertionError(f"Control readback did not arrive: {expected_current}")
            facts["control"] = {"fields": dict(view._field_states),
                                "risk_enabled": view.risk_switch.isEnabled(),
                                "risk_accepted": view.risk_switch.isChecked(),
                                "owners": view.owner_label.text()}
            if "risk_enabled" in action:
                assert view.risk_switch.isEnabled() == action["risk_enabled"], facts
            if "risk_accepted" in action:
                assert view.risk_switch.isChecked() == action["risk_accepted"], facts
            for field, expected in action.get("editable", {}).items():
                assert view.form.widget_for(field).isEnabled() == expected, facts
                assert view._field_states[field]["editable"] == expected, facts
            for field, expected in action.get("current", {}).items():
                assert view._field_states[field]["current"] == expected, facts
        elif any(key in action for key in ("current", "risk_enabled", "risk_accepted", "editable")):
            raise AssertionError("the requested generic Control readback is absent")
        if "report_contains" in action:
            matches = [(severity, message) for severity, message in bench.reports
                       if action["report_contains"] in message]
            assert matches, f"expected refusal was not reported: {action['report_contains']}"
            facts["expected_report"] = matches
        if action.get("screenshot"):
            manager.window().grab().save(str(output / f"device-{action['name']}-manager.png"))
            if view is not None:
                view.window().grab().save(str(output / f"device-{action['name']}-control.png"))
        write_json(output / f"device-{action['name']}.json", facts)
        return
    if kind in ("device_save_as", "device_load"):
        path = Path(str(action["path"]))
        if not path.is_absolute():
            path = output / path
        file_dialog_action(manager.save_as_button if kind == "device_save_as" else manager.load_button,
                           path, bench.app, click=click, enter_text=enter_text,
                           save=kind == "device_save_as")
        return
    if kind in ("device_log_check", "device_log_close"):
        window = getattr(manager, "_device_log_windows", {}).get(device)
        if window is None or not window.isVisible():
            raise UnavailableAction(f"device Log is not open: {device}")
        if kind == "device_log_close":
            click(window.titleBar.closeBtn, bench.app)
            assert beat.run_until(lambda: device not in manager._device_log_windows, 3), "Log did not retire"
        else:
            write_json(output / f"device-log-{action['name']}.json",
                       {"text": window.loaded.toPlainText(), "polling": window.loaded._timer.isActive()})
            window.grab().save(str(output / f"device-log-{action['name']}.png"))
        return
    if kind == "device_add":
        domain = str(action["domain"])
        button = manager.domain_add_buttons.get(domain)
        if button is None:
            raise UnavailableAction(f"no Add device button for domain: {domain}")
        before = set(manager._cards)
        click(button, bench.app)
        if not beat.run_until(lambda: set(manager._cards) != before, 3):
            raise AssertionError("Add device did not create a draft card")
        created = set(manager._cards) - before
        if len(created) != 1:
            raise AssertionError("Add device must create exactly one draft card")
        device = created.pop()
        if "as" in action:
            if not hasattr(bench, "fuzz_devices"):
                bench.fuzz_devices = {}
            bench.fuzz_devices[str(action["as"])] = device
        if "type" in action:
            choose(manager._cards[device].type_combo, action["type"], bench.app)
        return
    if kind == "device_template":
        choose(manager.new_combo, action["value"], bench.app)
        return
    if kind in ("device_lifecycle", "device_init", "device_apply", "device_shutdown"):
        verb = action["verb"] if kind == "device_lifecycle" else kind[7:]
        label = {"init": "Init devices", "apply": "Apply device changes",
                 "shutdown": "Shutdown devices", "refresh": "Refresh device views"}[verb]
        if manager.lifecycle_button.text() != label:
            raise UnavailableAction(f"wanted {label!r}, current action is "
                                    f"{manager.lifecycle_button.text()!r}")
        if action.get("virtual_only") and verb in ("init", "apply"):
            assert all(item.type_id.endswith((".virtual", ".virtual_mot"))
                       for item in flow.devices.presenter.devices), "physical draft must not be initialized"
        click(manager.lifecycle_button, bench.app)
        if not beat.run_until(lambda: not manager._busy,
                              float(action.get("timeout", 15))):
            raise AssertionError(f"Device Manager did not finish {verb}")
        return
    if kind in ("device_type", "device_role", "device_remove", "device_parameter", "device_collapse"):
        card = manager._cards.get(device)
        if card is None:
            raise UnavailableAction(f"device has no draft card: {device}")
        if kind == "device_type":
            choose(card.type_combo, action["value"], bench.app)
            return
        if kind == "device_role":
            enter_text(card.role_edit, action["value"], bench.app)
            return
        if kind == "device_remove":
            click(card.remove_button, bench.app)
            return
        if kind == "device_collapse":
            click(card.collapse_button, bench.app)
            return
        if card._collapsed:
            click(card.collapse_button, bench.app)
        form = card.form
    elif kind in ("device_control", "device_close", "device_remote", "device_log"):
        card = manager._loaded_cards.get(device)
        if card is None:
            raise UnavailableAction(f"device is not loaded: {device}")
        click(getattr(card, kind[7:] + "_button"), bench.app)
        if kind == "device_control" and not beat.run_until(
                lambda: device in flow.device_controls, float(action.get("timeout", 5))):
            raise AssertionError(f"Control did not open: {device}")
        return
    elif kind in ("device_desired", "device_field_apply", "device_live",
                  "device_refresh", "device_risk"):
        control = flow.device_controls.get(device)
        if control is None or not hasattr(getattr(control, "_view", None), "_field_rows"):
            raise UnavailableAction(f"open the generic Control first: {device}")
        view = control._view
        if kind == "device_refresh":
            click(view.refresh_button, bench.app)
            return
        if kind == "device_risk":
            if view.risk_switch.isChecked() != bool(action["value"]):
                click(view.risk_switch, bench.app)
            return
        key = str(action["field"])
        if key not in view._field_rows:
            raise UnavailableAction(f"Control field is absent: {key}")
        if kind == "device_field_apply":
            click(view._field_rows[key][3], bench.app)
            return
        if kind == "device_live":
            switch = view._field_rows[key][2]
            if switch.isChecked() != bool(action["value"]):
                click(switch, bench.app)
            return
        form = view.form
    if kind in ("device_parameter", "device_desired"):
        key, value = str(action["field"]), action["value"]
        if form is None or key not in form.keys:
            raise UnavailableAction(f"device form field is absent: {key}")
        automatic = form._auto_switches.get(key)
        if automatic is not None:
            if automatic.isChecked() != (value is None):
                click(automatic, bench.app)
                bench.app.processEvents()
            if value is None:
                return
        widget = form.widget_for(key)
        if "unit" in action:
            picker = form.unit_picker_for(key)
            if picker is None:
                raise UnavailableAction(f"device field has no unit picker: {key}")
            choose(picker, action["unit"], bench.app)
        if hasattr(widget, "_popup_view"):
            choose(widget, value, bench.app)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            if widget.isChecked() != bool(value):
                click(widget, bench.app)
        else:
            enter_text(widget, "" if value is None else value, bench.app)
        return
    if kind.startswith("pulse_"):
        control = flow.device_controls.get(device)
        if control is None:
            raise UnavailableAction(f"open the sequencer Control first: {device}")
        view = getattr(control, "_view", None)
        if not hasattr(view, "schedule_view"):
            raise UnavailableAction(f"Control is not a Pulse Editor: {device}")
        default_page = ("Preview" if kind.startswith("pulse_preview_") else
                        "Scan" if kind.startswith("pulse_scan_") else
                        "Target" if kind.startswith("pulse_target_") else "Edit")
        label = str(action.get("tab", "Edit") if kind == "pulse_tab" else
                    action.get("page", default_page))
        index = next((i for i in range(view.tabs.count())
                      if view.tabs.tabText(i) == label), None)
        if index is None:
            raise UnavailableAction(f"Pulse tab is absent: {label}")
        bar = view.tabs.tabBar()
        if not visible(bar, bench.app):
            raise UnavailableAction("Pulse tab bar is not visible")
        QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton,
                               pos=bar.tabRect(index).center())
        bench.app.processEvents()
        if kind == "pulse_tab":
            return
        schedule = view.schedule_view
        presenter = control.presenter
        if kind == "pulse_load":
            spelling = str(action["path"])
            path = (bench._tmp / spelling.removeprefix("$workspace/")
                    if spelling.startswith("$workspace/") else Path(spelling))
            if not path.is_absolute():
                path = bench._tmp / "pulses" / path
            path = path.resolve()
            if not path.is_file():
                raise UnavailableAction(f"pulse file is absent: {path}")
            file_dialog_action(schedule.load_button, path, bench.app,
                               click=click, enter_text=enter_text)
            if Path(control.presenter.path).resolve() != path:
                raise AssertionError(f"Pulse Load did not accept {path.name}")
        elif kind == "pulse_sync":
            from zlc_ui.fluent.fluent import _FluentMessageDialog

            answers = []
            timer = QtCore.QTimer(view)
            def acknowledge_sync():
                dialog = bench.app.activeModalWidget()
                if not isinstance(dialog, _FluentMessageDialog):
                    return
                timer.stop()
                text = "\n".join(child.toPlainText() for child in dialog.findChildren(QtWidgets.QPlainTextEdit))
                answers.append(text)
                dialog.grab().save(str(output / "pulse-sync-dialog.png"))
                button = next(child for child in dialog.findChildren(QtWidgets.QAbstractButton) if child.text() == "OK")
                click(button, bench.app)
            timer.timeout.connect(acknowledge_sync)
            timer.start(20)
            try:
                click(schedule.sync_button, bench.app)
                assert beat.run_until(lambda: bool(answers), 10), "Sync did not report its result"
                write_json(output / "pulse-sync-dialog.json", {"messages": answers})
                assert "synced from the board" in answers[0], answers
            finally:
                timer.stop()
                timer.deleteLater()
        elif kind in ("pulse_run", "pulse_stop"):
            click(getattr(schedule, kind.removeprefix("pulse_") + "_button"), bench.app)
        elif kind == "pulse_edit_button":
            buttons = {"show_all": schedule.show_all_button, "hide_off": schedule.hide_off_button,
                       "add": schedule.add_button, "remove": schedule.remove_button,
                       "bracket": schedule.bracket_button}
            click(buttons[action["button"]], bench.app)
        elif kind == "pulse_period":
            period = schedule._cards[action["period"]]
            field, value = action["field"], action["value"]
            if field == "name":
                enter_text(period.name_edit, value, bench.app)
            elif field == "duration":
                if "unit" in action:
                    choose(period.unit_combo, action["unit"], bench.app)
                enter_text(period.duration_edit, value, bench.app)
            elif field == "ttl":
                checkbox = period.checks[action["port"]]
                if checkbox.isChecked() != value:
                    click(checkbox, bench.app)
            elif field == "dac_mode":
                choose(period.bus_mode_combos[action["port"]], value, bench.app)
            elif field == "dac_value":
                enter_text(period.bus_value_edits[action["port"]], value, bench.app)
            elif field == "binding":
                for _ in range(4):
                    if period.duration_dot._kind == value:
                        break
                    click(period.duration_dot, bench.app)
                    bench.app.processEvents()
                    period = schedule._cards[action["period"]]
                assert period.duration_dot._kind == value, "binding cycle did not reach the requested mode"
            else:
                raise ValueError(f"unknown period field: {field}")
        elif kind == "pulse_repeat":
            enter_text(schedule.channel_panel.run_repeats_spin, action["value"], bench.app)
        elif kind == "pulse_preview_field":
            preview = view.preview_view
            if action["field"] == "size":
                if not beat.run_until(lambda: not presenter._preview_busy
                        and preview.preview_size_combo.findText(str(action["value"])) >= 0,
                        float(action.get("timeout", 15))):
                    view.window().grab().save(str(output / "pulse-preview-not-ready.png"))
                    raise UnavailableAction(f"Preview size is not available: {action['value']}; "
                                            f"{preview.preview_status.text()}")
                choose(preview.preview_size_combo, action["value"], bench.app)
            else:
                switch = {"selectors": preview.preview_selectors_switch,
                          "include_off": preview.preview_include_off}[action["field"]]
                if switch.isChecked() != action["value"]:
                    click(switch, bench.app)
        elif kind == "pulse_preview_save":
            import shutil

            assert beat.run_until(lambda: not presenter._preview_busy, float(action.get("timeout", 15)))
            folder = Path(presenter.path).parent
            before_files = set(folder.glob("*.png"))
            click(view.preview_view.preview_save_figure_button, bench.app)
            assert beat.run_until(lambda: not presenter._preview_busy and bool(
                set(folder.glob("*.png")) - before_files), float(action.get("timeout", 15))), "Preview PNG was not saved"
            saved = next(iter(set(folder.glob("*.png")) - before_files))
            with saved.open("rb") as stream:
                assert stream.read(8) == b"\x89PNG\r\n\x1a\n", "Preview output is not PNG"
            shutil.copyfile(saved, output / "pulse-preview.png")
        elif kind == "pulse_scan_code":
            enter_text(view.scan_view.scan_code, action["value"], bench.app)
            actual = view.scan_view.scan_code.toPlainText()
            write_json(output / "pulse-scan-input.json", {"requested": action["value"], "actual": actual})
            assert actual == str(action["value"]), "Scan text input differs before Run"
        elif kind == "pulse_scan_repeats":
            enter_text(view.scan_view.scan_repeats_spin, action["value"], bench.app)
        elif kind == "pulse_scan_button":
            scan = view.scan_view
            buttons = {"run": scan.scan_run_button, "hold": scan.scan_hold_button,
                       "step_back": scan.scan_step_back_button, "step_forward": scan.scan_step_forward_button,
                       "column_template": scan.scan_column_template_button, "grid_template": scan.scan_grid_template_button}
            click(buttons[action["button"]], bench.app)
        elif kind == "pulse_target_name":
            row = next(row for row in view.target_view._rows if row.record.key == action["port"])
            enter_text(row.signal, action["value"], bench.app)
        elif kind == "pulse_target_apply":
            click(view.target_view.apply_button, bench.app)
        elif kind == "pulse_checkpoint":
            assert beat.run_until(lambda: not presenter._device_busy and not presenter._stop_busy
                and ("running" not in action or presenter.running == action["running"]),
                float(action.get("timeout", 10))), "Pulse command did not settle"
            facts = {"page": view.current_page, "running": presenter.running,
                     "answering": presenter._board_state.answering, "fault": presenter._board_state.fault,
                     "scan_rows": len(presenter._state.scan_rows), "held_point": presenter._held_point,
                     "summary": view.summary_label.text(), "scan_progress": view.scan_view.scan_progress_label.text(),
                     "preview_status": view.preview_view.preview_status.text(),
                     "periods": [{"id": p.period_id, "name": p.name, "duration": p.duration, "unit": p.unit}
                                 for p in presenter.sequence.periods]}
            for key in ("scan_rows", "held_point"):
                if key in action:
                    assert facts[key] == action[key], facts
            if action.get("stopped"):
                assert facts["answering"] and not facts["running"] and not facts["fault"], facts
                assert facts["summary"] == "Stopped", facts
            if action.get("target_locked"):
                target = view.target_view
                assert not target.add_digital_button.isEnabled() and not target.add_dac_button.isEnabled()
                assert all(not row.endpoints.isEnabled() and not row.width.isEnabled()
                           and not row.remove_button.isEnabled() for row in target._rows)
                facts["target_locked"] = True
                facts["target_labels"] = {row.record.key: row.signal.text() for row in target._rows}
            if action.get("screenshot"):
                view.window().grab().save(str(output / f"pulse-{action['name']}.png"))
            write_json(output / f"pulse-{action['name']}.json", facts)
        else:
            raise UnavailableAction(f"unknown Pulse action: {kind}")
        return
    raise UnavailableAction(f"unknown device action: {kind}")


def perform_artifact_action(bench, action, beat, output, *, click, choose, enter_text):
    """Drive review/SLM/Viewer widgets; never answer a node or set its draft.

    Review is a modal dialog: the caller must schedule these actions on a Qt
    timer before the owner enters its nested event loop. SLM actions use an
    already opened Control from the experiment flow's real device card.
    """
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

    def selected_path(*, figure=False, existing=False):
        text = str(action["path"])
        path = Path(bench.artifacts[text.removeprefix("$artifact/")]
                    if text.startswith("$artifact/") else text).expanduser()
        if not path.is_absolute():
            path = Path(output) / path
        if "figure" in action:
            if path.parent.name != "final":
                raise ValueError("figure selection requires a Task result in its final/ folder")
            path = (path.parent.parent / "figures" / str(action["figure"])).with_suffix(".npz")
        path = path.resolve()
        if figure and path.suffix.lower() != ".npz":
            raise ValueError("Viewer needs a Figure NPZ; use figure='site_map' for Calibration")
        if existing and not path.is_file():
            raise FileNotFoundError(path)
        return path

    kind = action["kind"]
    if kind.startswith("review_"):
        from zlc_ui.console.point_review_view import PointReviewView

        def current_review():
            return next((widget for widget in bench.app.allWidgets()
                         if isinstance(widget, PointReviewView) and widget.isVisible()), None)

        if kind == "review_wait":
            if not beat.run_until(lambda: current_review() is not None,
                                  float(action.get("timeout", 30.0))):
                raise AssertionError("Calibration review dialog did not open")
            return
        review = current_review()
        if review is None:
            raise UnavailableAction("no visible Calibration review dialog")
        if kind == "review_capture":
            from zlc_ui.acceptance import capture_window
            capture_window(lambda: review.window(),
                           output=Path(output) / action.get("name", "review.png"), close=False)
        elif kind == "review_search":
            enter_text(review.search, action.get("text", ""), bench.app)
        elif kind in ("review_point", "review_rectangle"):
            from dataclasses import replace
            from zlc_plot.point_review import ImagePointReviewSurface

            surface = review.findChild(ImagePointReviewSurface)
            if surface is None or surface._plot.presented_front is None:
                raise UnavailableAction("point-review image has no installed front")
            plot = surface._plot
            pointer = Pointer(plot, bench.app, QtCore, QtGui)
            pointer.post = True
            if kind == "review_point":
                overlay = surface._interaction._overlay
                identity = (str(action["site_id"]) if "site_id" in action
                            else overlay.point_ids[int(action.get("index", 0))])
                index = overlay.point_ids.index(identity)
                axis = axis_by_role(plot.presented_front, "image")
                # Use the plot's transform with canonical axes for an exact
                # site coordinate, not the rounded text in the checkbox.
                canonical = replace(axis, x_limits=axis.canonical_x_limits,
                                    y_limits=axis.canonical_y_limits)
                point = canonical.display_to_normalized(*overlay.coordinates[index])
                was_excluded = identity in review.excluded_ids
                pointer.press(*point)
                pointer.release(*point)
                if not beat.run_until(lambda: (identity in review.excluded_ids) != was_excluded, 3.0):
                    raise AssertionError(f"review click did not toggle {identity}")
            else:
                left, top, right, bottom = map(float, action["bounds"])
                pointer.press(left, top)
                beat.run(.02)
                for step in range(1, 5):
                    pointer.move(left + (right-left)*step/4, top + (bottom-top)*step/4)
                    beat.run(.01)
                pointer.release(right, bottom)
                beat.run(.02)
                if "expected_selected" in action and set(review.selected_ids) != set(action["expected_selected"]):
                    raise AssertionError("review rectangle selected different site identities")
        elif kind in ("review_site", "review_exclude", "review_restore"):
            identities = action.get("site_ids")
            if identities is None:
                identity = (action["site_id"] if "site_id" in action
                            else review._point_ids[int(action.get("index", 0))])
                identities = [identity]
            for identity in identities:
                checkbox = review._checks[str(identity)]
                retained = (bool(action["checked"]) if kind == "review_site"
                            else kind == "review_restore")
                if checkbox.isChecked() != retained:
                    click(checkbox, bench.app)
        else:
            buttons = {
                "review_exclude_selected": review.exclude_selected_button,
                "review_restore_selected": review.restore_selected_button,
                "review_reset": review.reset_button,
                "review_confirm": review.confirm_button,
                "review_cancel": review.stop_button,
            }
            if kind not in buttons:
                raise ValueError(f"unsupported artifact action {kind!r}")
            click(buttons[kind], bench.app)
        return

    if kind.startswith("slm_"):
        flow = bench.flow
        key = str(action.get("device", "slm"))
        control = flow.device_controls.get(key)
        if control is None:
            raise UnavailableAction(f"SLM Control {key!r} is not open")
        if kind in ("slm_tab", "slm_pupil_open", "slm_pupil_field", "slm_pupil_apply",
                    "slm_pupil_enabled", "slm_zernike_enabled", "slm_wavefront_field",
                    "slm_wavefront_reset", "slm_paint"):
            label = action.get("tab", "Wavefront" if kind.startswith("slm_wavefront_") else "Pattern")
            tabs = control._layer_tabs
            index = next(i for i in range(tabs.count()) if tabs.tabText(i) == label)
            if tabs.currentIndex() != index:
                QtTest.QTest.mouseClick(tabs.tabBar(), QtCore.Qt.LeftButton,
                                       pos=tabs.tabBar().tabRect(index).center())
                bench.app.processEvents()
            if kind == "slm_tab":
                return
        if kind == "slm_wait":
            if not beat.run_until(lambda: control.solver_idle and not control.command_active
                                  and not control._device_state_in_flight,
                                  float(action.get("timeout", 30.0))):
                raise AssertionError(f"SLM Control did not settle: {control.status_text}")
            if action.get("check"):
                import numpy as np

                command_revision, mapping_revision, commanded, receipt = control._device_state
                facts = dict(status=control.status_text, device_status=control._device_status.text(),
                             solver_idle=control.solver_idle, command_active=control.command_active,
                             send_enabled=control._send.isEnabled(), shape=list(control.shape),
                             phase_dtype=str(control._phase.dtype), target_sites=int(np.count_nonzero(control._target)),
                             phase_matches_command=bool(np.array_equal(control._phase, commanded)),
                             command_revision=command_revision, mapping_revision=mapping_revision,
                             command_outcome=receipt.get("outcome"))
                facts["pupil"] = control._pupil_settings()
                facts["operator"] = control._operator_settings()
                if "remember_command" in action:
                    if not hasattr(bench, "slm_command_marks"):
                        bench.slm_command_marks = {}
                    bench.slm_command_marks[action["remember_command"]] = (command_revision, mapping_revision)
                if "same_command_as" in action:
                    assert (command_revision, mapping_revision) == bench.slm_command_marks[action["same_command_as"]]
                for x, y, expected in action.get("target_points", ()):
                    assert float(control._target[int(y), int(x)]) == expected, (x, y, expected)
                if "expect_pupil" in action:
                    assert facts["pupil"] == action["expect_pupil"], facts
                if "expect_operator" in action:
                    assert facts["operator"] == action["expect_operator"], facts
                if action.get("check_context"):
                    from zlc_atom.devices.slm.solver import load_science_context

                    context = load_science_context(selected_path(existing=True))
                    facts["phase_matches_context"] = bool(np.array_equal(control._phase, context["phase"]))
                    facts["target_matches_context"] = bool(np.array_equal(control._target, context["target_intensity"]))
                    if not facts["phase_matches_context"] or not facts["target_matches_context"]:
                        raise AssertionError("SLM controls differ from the selected Science Context")
                if action.get("check_target"):
                    from zlc_atom.devices.slm.solver import load_target

                    target, objective = load_target(selected_path(existing=True))
                    facts["target_matches_file"] = bool(np.array_equal(control._target, target))
                    facts["objective_matches_file"] = control._objective_kind == objective
                    if not facts["target_matches_file"] or not facts["objective_matches_file"]:
                        raise AssertionError("SLM controls differ from the selected Target")
                checkpoint = str(action.get("check", "state"))
                write_json(Path(output) / f"slm-{checkpoint}.json", facts)
                control._window.grab().save(str(Path(output) / f"slm-{checkpoint}.png"))
                if "expect_status" in action and str(action["expect_status"]) not in facts["status"]:
                    raise AssertionError(f"SLM status differs from requested outcome: {facts}")
                if action.get("expect_command_match") and not facts["phase_matches_command"]:
                    raise AssertionError("SLM painted phase differs from its delivered command state")
        elif kind == "slm_preset":
            click(control._preset_button, bench.app)
            choose(control._preset_type, action.get("preset", "Grid"), bench.app)
            for name, value in action.get("values", {}).items():
                enter_text(control._preset_fields[str(name)], value, bench.app)
            if "text" in action:
                enter_text(control._preset_text, action["text"], bench.app)
            apply_button = next(button for button in control._preset_body.findChildren(QtWidgets.QAbstractButton)
                                if button.text() == "Apply")
            click(apply_button, bench.app)
        elif kind == "slm_clear":
            clear = next(button for button in control._body.findChildren(QtWidgets.QAbstractButton)
                         if button.text() == "Clear")
            click(clear, bench.app)
        elif kind == "slm_selectors":
            if control._selectors.isChecked() != bool(action["enabled"]):
                click(control._selectors, bench.app)
        elif kind == "slm_size":
            choose(control._plot_size, action["size"], bench.app)
        elif kind in ("slm_pupil_enabled", "slm_zernike_enabled"):
            switch = control._pupil_enabled if kind == "slm_pupil_enabled" else control._zernike_enabled
            if switch.isChecked() != bool(action["value"]):
                click(switch, bench.app)
        elif kind == "slm_pupil_open":
            if not control._pupil_body.isVisible():
                click(control._pupil_edit, bench.app)
        elif kind == "slm_pupil_field":
            fields = {"center_x": control._pupil_center_x, "center_y": control._pupil_center_y,
                      "diameter_x": control._pupil_diameter_x, "diameter_y": control._pupil_diameter_y}
            enter_text(fields[action["field"]], action["value"], bench.app)
        elif kind == "slm_pupil_apply":
            button = next(button for button in control._pupil_body.findChildren(QtWidgets.QAbstractButton)
                          if button.text() == "Apply pupil")
            click(button, bench.app)
        elif kind == "slm_wavefront_field":
            fields = {"carrier_x": control._carrier_x, "carrier_y": control._carrier_y, **control._zernike}
            enter_text(fields[action["field"]], action["value"], bench.app)
        elif kind == "slm_wavefront_reset":
            button = next(button for button in control._wavefront_parameter_scroll.findChildren(QtWidgets.QAbstractButton)
                          if button.text() == "Reset wavefront")
            click(button, bench.app)
        elif kind == "slm_paint":
            if control._selectors.isChecked():
                raise UnavailableAction("turn Selectors off before painting the Target")
            choose(control._mode, action["mode"], bench.app)
            if "intensity" in action:
                enter_text(control._intensity, action["intensity"], bench.app)
            widget = control._target_widget
            assert beat.run_until(lambda: widget.presented_front is not None, 5)
            axis = next(axis for axis in widget.presented_front.interaction.axes if axis.role == "image")
            assert axis.x_scale == axis.y_scale == "linear"
            left, top, right, bottom = axis.bounds
            x0, x1 = axis.canonical_x_limits
            y0, y1 = axis.canonical_y_limits
            points = [(left + (x-x0)/(x1-x0)*(right-left), top + (y-y1)/(y0-y1)*(bottom-top))
                      for x, y in action["points"]]
            if not visible(widget, bench.app):
                raise UnavailableAction("SLM Target is not visible")
            pointer = Pointer(widget, bench.app, QtCore, QtGui)
            pointer.post = True
            pointer.press(*points[0])
            beat.run(0.03)
            for point in points[1:]:
                pointer.move(*point)
                beat.run(0.03)
            pointer.release(*points[-1])
        elif kind == "slm_send":
            click(control._send, bench.app)
        elif kind == "slm_adopt":
            click(control._adopt, bench.app)
        elif kind in ("slm_save_target", "slm_load_target", "slm_import_target",
                      "slm_save_context", "slm_load_context"):
            labels = {
                "slm_save_target": "Save target", "slm_load_target": "Load target",
                "slm_import_target": "Import target",
                "slm_save_context": "Save science context",
                "slm_load_context": "Load science context",
            }
            button = next(button for button in control._body.findChildren(QtWidgets.QAbstractButton)
                          if button.text() == labels[kind])
            path = selected_path(existing=not kind.startswith("slm_save_"))
            file_dialog_action(button, path, bench.app, click=click,
                               enter_text=enter_text, save=kind.startswith("slm_save_"))
        else:
            raise ValueError(f"unsupported artifact action {kind!r}")
        return
    if kind.startswith("viewer_"):
        if kind == "viewer_open":
            if getattr(bench, "viewer", None) is not None:
                raise RuntimeError("close the existing benchmark Viewer before opening another")
            path = selected_path(figure=True, existing=True)
            # This is the application's injected window launcher. It uses
            # create_window and retains the Console's A/C leases; the real
            # chooser still chooses the file, never a patched return value.
            presenter = file_dialog_action(
                None, path, bench.app, click=click, enter_text=enter_text,
                opener=lambda: bench.presenter._open_saved(str(path.parent)))
            if presenter is None:
                raise AssertionError("FigureViewer launch returned no window")
            bench.viewer = presenter.view
            bench._capture_reports(presenter._panel_presenter)
            original_status = bench.viewer.set_status
            def viewer_status(text, *, error=False):
                severity = "error" if error else "info"
                bench.reports.append((severity, str(text)))
                bench.report_events.append(dict(time_ns=time.perf_counter_ns(),
                    severity=severity, message=str(text), application="viewer"))
                return original_status(text, error=error)
            bench.viewer.set_status = viewer_status
            return
        viewer = getattr(bench, "viewer", None)
        if viewer is None:
            raise UnavailableAction("no benchmark FigureViewer is open")
        view = viewer._view
        presenter = viewer.presenter
        if kind == "viewer_wait":
            report_cursor = bench.report_cursor()
            expected_path = selected_path(figure=True, existing=True) if "path" in action else None
            def ready():
                bench.require_no_errors(report_cursor, "FigureViewer operation")
                return (not presenter._busy and presenter.path is not None
                        and not presenter._panel_presenter._saving_panels
                        and (expected_path is None or presenter.path.resolve() == expected_path)
                        and bool(view._cards)
                        and all(card.surface is not None and card.surface.presented_front is not None
                                for card in view._cards.values()))
            if not beat.run_until(ready, float(action.get("timeout", 30.0))):
                raise AssertionError("FigureViewer did not settle with an installed figure")
        elif kind == "viewer_load":
            file_dialog_action(view.info_pane.path_edit.browse,
                               selected_path(figure=True, existing=True), bench.app,
                               click=click, enter_text=enter_text)
        elif kind == "viewer_refresh":
            click(view.info_pane.path_edit.refresh, bench.app)
        elif kind in ("viewer_info_tab", "viewer_tab"):
            tabs = view.info_pane.info_tabs if kind == "viewer_info_tab" else view.tabs
            index = next(index for index in range(tabs.count())
                         if tabs.tabText(index) == str(action["tab"]))
            bar = tabs.tabBar()
            QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=bar.tabRect(index).center())
            if tabs.currentIndex() != index:
                raise AssertionError("Viewer tab click did not select the requested page")
        elif kind.startswith("viewer_pulse_"):
            played = tuple(item for item in (presenter.description.pulses
                                            if presenter.description is not None else ())
                           if ("key" not in action or item.key == str(action["key"]))
                           and ("pulse" not in action or item.name == str(action["pulse"])))
            if not played:
                raise UnavailableAction("this archive has no matching played Pulse")
            if len(played) != 1:
                raise UnavailableAction("choose a unique played Pulse by key or pulse name")
            played = played[0]
            key = played.key
            if kind == "viewer_pulse_open":
                tabs = view.info_pane.info_tabs
                if tabs.tabText(tabs.currentIndex()) != "Devices":
                    raise UnavailableAction("open the Viewer Devices tab before its Pulse action")
                tree = tabs.currentWidget().tree
                rows = [(tree.topLevelItem(index), tree.itemWidget(tree.topLevelItem(index), 1))
                        for index in range(tree.topLevelItemCount())]
                matches = [(row, button) for row, button in rows
                           if isinstance(button, QtWidgets.QAbstractButton) and button.text() == played.name]
                if len(matches) != 1:
                    raise UnavailableAction("Devices does not show one unambiguous played Pulse button")
                row, button = matches[0]
                tree.scrollToItem(row, QtWidgets.QAbstractItemView.PositionAtCenter)
                tree.scrollTo(tree.indexFromItem(row, 1), QtWidgets.QAbstractItemView.EnsureVisible)
                bench.app.processEvents()
                if not visible(button, bench.app):
                    raise UnavailableAction("played Pulse action is not visible and enabled")
                # The record tree's value column may be wider than its
                # viewport. Click the visible part of the actual button,
                # not its offscreen centre or the presenter's action API.
                viewport = tree.viewport()
                exposed = button.rect().intersected(QtCore.QRect(
                    button.mapFromGlobal(viewport.mapToGlobal(QtCore.QPoint())), viewport.size()))
                if exposed.isEmpty():
                    raise UnavailableAction("played Pulse action is outside the tree viewport")
                point = exposed.center()
                window = button.window()
                hit = window.childAt(window.mapFromGlobal(button.mapToGlobal(point)))
                if hit is not button and not button.isAncestorOf(hit):
                    raise UnavailableAction("played Pulse action is covered")
                QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton, pos=point)
            else:
                page = view._pulse_tabs.get(key)
                if page is None or view.tabs.currentWidget() is not page:
                    raise UnavailableAction("select the opened played Pulse tab first")
                if kind == "viewer_pulse_wait":
                    def pulse_ready():
                        return (not presenter._busy and page._content_widget is not None
                                and page._content_widget.presented_front is not None
                                and page.preview_size_combo.count() > 0
                                and ("size" not in action or page.preview_size == str(action["size"]))
                                and ("include_off" not in action or page.include_off_rows == bool(action["include_off"]))
                                and ("selectors" not in action or page.preview_selectors_switch.isChecked() == bool(action["selectors"])))
                    if not beat.run_until(pulse_ready, float(action.get("timeout", 15))):
                        raise AssertionError(f"played Pulse did not settle: {page.preview_status.text()}")
                elif kind in ("viewer_pulse_selectors", "viewer_pulse_include_off"):
                    switch = (page.preview_selectors_switch if kind == "viewer_pulse_selectors"
                              else page.preview_include_off)
                    if switch.isChecked() != bool(action["value"]):
                        click(switch, bench.app)
                    if switch.isChecked() != bool(action["value"]):
                        raise AssertionError("played Pulse switch click was not accepted")
                elif kind == "viewer_pulse_size":
                    if presenter._busy:
                        raise UnavailableAction("wait for the current Pulse redraw before changing size")
                    choose(page.preview_size_combo, str(action["value"]), bench.app)
                elif kind == "viewer_pulse_save":
                    import shutil
                    from PIL import Image

                    if not beat.run_until(lambda: not presenter._busy, float(action.get("timeout", 20))):
                        raise AssertionError("played Pulse is still busy before Save Figure")
                    archive = Path(presenter.path)
                    folder = archive.parent
                    before_archive = archive.read_bytes()
                    before_npz = set(folder.glob("*.npz"))
                    before_png = set(folder.glob("*.png"))
                    click(page.preview_save_figure_button, bench.app)
                    if not beat.run_until(lambda: not presenter._busy and bool(
                            set(folder.glob("*.png")) - before_png), float(action.get("timeout", 20))):
                        raise AssertionError("played Pulse Save Figure did not publish a PNG")
                    saved = set(folder.glob("*.png")) - before_png
                    if len(saved) != 1:
                        raise AssertionError("played Pulse Save Figure published an ambiguous PNG set")
                    saved = saved.pop()
                    if not saved.name.startswith(f"{archive.stem}-pulse-"):
                        raise AssertionError("new PNG is not the Viewer played Pulse export")
                    with Image.open(saved) as image:
                        image_size, image_format = image.size, image.format
                        image.verify()
                    if image_format != "PNG":
                        raise AssertionError("played Pulse output is not a valid PNG")
                    unchanged = archive.read_bytes() == before_archive
                    no_new_npz = set(folder.glob("*.npz")) == before_npz
                    if not unchanged or not no_new_npz:
                        raise AssertionError("Pulse preview Save changed or created a Figure document")
                    name = str(action.get("name", "viewer-pulse-export"))
                    copied = Path(output) / f"{name}.png"
                    shutil.copyfile(saved, copied)
                    write_json(Path(output) / f"{name}.json", {
                        "archive": str(archive), "saved_png": str(saved), "copied_png": str(copied),
                        "pulse_key": key, "pulse_name": played.name,
                        "image_size": image_size, "image_format": image_format,
                        "size": page.preview_size, "include_off": page.include_off_rows,
                        "selectors": page.preview_selectors_switch.isChecked(),
                        "archive_bytes_unchanged": unchanged, "no_new_npz": no_new_npz,
                    })
                else:
                    raise ValueError(f"unsupported artifact action {kind!r}")
        elif kind == "viewer_panel_edit":
            key = str(action.get("panel", next(iter(view._cards))))
            card = view._cards[key]
            bar = view.tabs.tabBar()
            QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=bar.tabRect(0).center())
            bench.app.processEvents()
            if card._settings_popup is None or not card._settings_popup.isVisible():
                click(card.settings_button, bench.app)
            click(card.edit_button, bench.app)
        elif kind == "viewer_panel_field":
            key = str(action.get("panel", next(iter(view._cards))))
            editor = view._editors[key]
            name = str(action["field"])
            section = str(action.get("section", "display"))
            if "__" in name:
                section, name = name.split("__", 1)
            form = editor.panel_form if section == "panel" else editor.parameter_forms[section]
            field = next(field for field in form.spec.fields if field.key == name)
            widget = form.widget_for(field.key)
            if field.kind == "choice":
                choose(widget, action["value"], bench.app)
            elif field.kind == "bool":
                if widget.isChecked() != bool(action["value"]):
                    click(widget, bench.app)
            else:
                enter_text(widget, action["value"], bench.app)
        elif kind in ("viewer_panel_refresh", "viewer_panel_save"):
            key = str(action.get("panel", next(iter(view._cards))))
            editor = view._editors[key]
            if kind == "viewer_panel_refresh":
                click(editor.refresh_button, bench.app)
            else:
                path = selected_path()
                enter_text(editor.save_directory, path.parent, bench.app)
                if editor.save_auto_name.isChecked():
                    click(editor.save_auto_name, bench.app)
                enter_text(editor.save_name, path.stem, bench.app)
                if path.suffix.lower() in (".png", ".pdf", ".svg"):
                    choose(editor.save_format, path.suffix[1:].lower(), bench.app)
                frozen = presenter._panel_presenter.panels[key].frozen_data
                bench.fuzz_saves.append((path.with_suffix(".npz"), frozen.snapshot.ref,
                                         tuple(frozen.description.selectors)))
                click(editor.save_button, bench.app)
        elif kind == "viewer_save_image":
            click(view.save_image_button, bench.app)
        elif kind in ("viewer_new_data", "viewer_edit_data"):
            if kind == "viewer_edit_data" and "dataset" in action:
                choose(view.data_combo, action["dataset"], bench.app)
            click(view.new_data_button if kind == "viewer_new_data" else view.edit_data_button, bench.app)
        elif kind.startswith("viewer_data_"):
            editor = view.tabs.currentWidget()
            if editor not in view._data_editors.values():
                raise UnavailableAction("select an open Data editor tab first")
            if kind == "viewer_data_field":
                fields = {"name": editor.name_edit, "unit": editor.unit_edit,
                          "note": editor.note_edit, "dtype": editor.dtype_combo}
                widget = fields[action["field"]]
                if action["field"] == "dtype":
                    choose(widget, action["value"], bench.app)
                else:
                    enter_text(widget, action["value"], bench.app)
            elif kind == "viewer_data_cell":
                table = editor.axis_value_table if action.get("table") == "axis" else editor.value_table
                index = table.model().index(int(action.get("row", 0)), int(action.get("column", 0)))
                if not index.isValid():
                    raise ValueError("requested cell is outside the shown table")
                if not visible(table, bench.app):
                    raise UnavailableAction("Data table is not visible and enabled")
                table.scrollTo(index)
                bench.app.processEvents()
                QtTest.QTest.mouseDClick(table.viewport(), QtCore.Qt.LeftButton,
                                        pos=table.visualRect(index).center())
                bench.app.processEvents()
                edit = bench.app.focusWidget()
                if not isinstance(edit, QtWidgets.QLineEdit):
                    raise AssertionError("table double-click did not open its real cell editor")
                enter_text(edit, action["value"], bench.app)
            elif kind == "viewer_data_axis_select":
                choose(editor.axis_combo, action["axis"], bench.app)
            elif kind == "viewer_data_axis_field":
                fields = {"name": editor.axis_name_edit, "unit": editor.axis_unit_edit,
                          "length": editor.axis_size_spin, "domain": editor.domain_combo}
                if action["field"] == "domain":
                    choose(fields["domain"], action["value"], bench.app)
                else:
                    enter_text(fields[action["field"]], action["value"], bench.app)
            elif kind == "viewer_data_table_axis":
                choose(editor._axis_view_widgets[action["axis"]][2], action["mode"], bench.app)
            elif kind == "viewer_data_save":
                file_dialog_action(editor.save_button, selected_path(), bench.app,
                                   click=click, enter_text=enter_text, save=True)
            else:
                buttons = {"viewer_data_apply": editor.apply_button,
                           "viewer_data_discard": editor.discard_button,
                           "viewer_data_axis_add": editor.add_axis_button,
                           "viewer_data_axis_apply": editor.apply_axis_button,
                           "viewer_data_axis_delete": editor.remove_axis_button}
                if kind not in buttons:
                    raise ValueError(f"unsupported artifact action {kind!r}")
                click(buttons[kind], bench.app)
        elif kind == "viewer_close":
            # The window's close guard releases only this Viewer lease. Keep
            # its handle if close is refused so the outer finally can retry.
            viewer.close()
            if not beat.run_until(lambda: not viewer.is_visible(), float(action.get("timeout", 30.0))):
                raise AssertionError("FigureViewer has not finished its guarded close")
            bench.viewer = None
        else:
            raise ValueError(f"unsupported artifact action {kind!r}")
        if action.get("capture"):
            bench.app.processEvents()
            name = str(action["capture"])
            if not viewer._window.grab().save(str(Path(output) / f"{name}.png")):
                raise AssertionError("could not save actual FigureViewer screenshot")
            current = view.tabs.currentWidget()
            facts = {"path": str(presenter.path),
                     "info_tab": view.info_pane.info_tabs.tabText(view.info_pane.info_tabs.currentIndex()),
                     "main_tab": view.tabs.tabText(view.tabs.currentIndex()),
                     "busy": presenter._busy,
                     "info_tree_top_rows": {title: tab.tree.topLevelItemCount()
                                            for title, tab in view.info_pane._rows_tabs.items()}}
            if current in view._data_editors.values():
                facts.update(dirty=current.is_dirty(), message=current.message_label.text(),
                             first_cell=current.value_model.data(current.value_model.index(0, 0), QtCore.Qt.EditRole))
            if current in view._pulse_tabs.values():
                facts.update(pulse_size=current.preview_size, include_off=current.include_off_rows,
                             selectors=current.preview_selectors_switch.isChecked(),
                             pulse_status=current.preview_status.text(),
                             installed_front=(current._content_widget is not None
                                              and current._content_widget.presented_front is not None))
            write_json(Path(output) / f"{name}.json", facts)
        return
    raise ValueError(f"unsupported artifact action {kind!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actions", type=int, default=60)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--chain", action="store_true", help="Start with camera -> drawn ROI -> three ordinary downstream panels")
    parser.add_argument("--system", action="store_true", help="Use the actual Device Manager Init -> experiment GUI flow")
    parser.add_argument("--feedback-scope-probe", action="store_true")
    parser.add_argument("--replay")
    parser.add_argument("--stop-on-report", help="Stop a reproducer at this exact diagnostic substring")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("bench/results/gui-fuzz") / f"{int(time.time())}-seed-{args.seed}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.child:
        return run_child(args, output)
    command = [sys.executable, "-X", "faulthandler", "-u", "-m", "bench.plot_perf.run_gui_fuzz", "--child", "--seed", str(args.seed), "--actions", str(args.actions), "--frames", str(args.frames), "--output", str(output)]
    if args.inventory:
        command.append("--inventory")
    if args.chain:
        command.append("--chain")
    if args.system:
        command.append("--system")
    if args.feedback_scope_probe:
        command.append("--feedback-scope-probe")
    if args.replay:
        command += ["--replay", str(Path(args.replay).resolve())]
    if args.stop_on_report:
        command += ["--stop-on-report", args.stop_on_report]
    with (output / "console.log").open("w", encoding="utf-8") as log:
        import psutil
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owned = psutil.Process(child.pid)
        descendants = {}
        print("GUI_FUZZ", child.pid, output, flush=True)
        try:
            allowance = 90 + args.actions * 2
            if args.replay:
                allowance += sum(float(action.get("timeout", 0))
                                 for action in json.loads(Path(args.replay).read_text(encoding="utf-8")))
            deadline = time.monotonic() + allowance
            while child.poll() is None:
                try:
                    for process in owned.children(recursive=True):
                        descendants[(process.pid, process.create_time())] = process
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, allowance)
                try:
                    child.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
            status = child.returncode
        except subprocess.TimeoutExpired:
            import psutil
            owned = psutil.Process(child.pid)
            children = owned.children(recursive=True)
            write_json(output / "watchdog.json", dict(pid=child.pid, child_pids=[p.pid for p in children], reason="owned test exceeded its bounded duration"))
            for process in reversed(children):
                try: process.terminate()
                except psutil.NoSuchProcess: pass
            owned.terminate()
            status = child.wait(timeout=10)
        _gone, alive = psutil.wait_procs(list(descendants.values()), timeout=3)
        remaining = [process.pid for process in alive if process.is_running()]
        write_json(output / "shutdown.json", dict(
            test_pid=child.pid, exit_code=status,
            descendant_pids=sorted(process.pid for process in descendants.values()),
            remaining_after_close=remaining))
        if remaining:
            # Only this test's recorded process identities, never another
            # Python/GUI instance or a PID recycled after the process exited.
            for process in alive:
                try:
                    if process.is_running():
                        process.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=3)
            status = status or 1
        print("GUI_FUZZ_EXIT", status, output, flush=True)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
