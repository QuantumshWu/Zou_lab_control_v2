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
from .gui_checks import check_panel, install_observers, panel_checkpoint


class UnavailableAction(RuntimeError):
    """A planned widget is no longer a place an operator can click."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def visible(widget, app):
    """Scroll the actual ancestor viewports, then use the widget's live box."""
    from PyQt5 import QtWidgets
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
    if isinstance(widget, QtWidgets.QAbstractButton) and not widget.hitButton(point):
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
            if value == wanted or isinstance(value, Enum) and str(value) == wanted:
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
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, 0, QtCore.Qt.NoModifier, character)
        app.sendEvent(target, event)
        app.processEvents()
    target = app.focusWidget()
    if target is not None:
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
    from PyQt5 import QtCore, QtGui, QtTest
    view = bench.view._view
    kind = action["kind"]
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

    def exception_hook(kind, value, tb):
        message = "".join(traceback.format_exception(kind, value, tb))
        errors.append(message)
        journal.write(json.dumps(dict(event="unhandled_exception", message=message)) + "\n")
    sys.excepthook = exception_hook
    result = dict(seed=args.seed, requested_actions=args.actions, completed_actions=0,
                  frames_per_cycle=args.frames, errors=errors)
    try:
        with ConsoleBench() as bench:
            bench.fuzz_saves = []
            bench.fuzz_gestures = []
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
                with ProductBeat(bench.app, bench.presenter) as beat:
                    first = next(iter(bench.view._cards.values()))
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
                            faulthandler.dump_traceback_later(15)
                            started = time.perf_counter()
                            try:
                                skipped = perform_action(bench, action, beat, output)
                            except UnavailableAction as unavailable:
                                skipped = str(unavailable)
                            beat.run(action["delay"])
                            if errors:
                                raise AssertionError("unhandled Qt exception")
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
                if probe_cleanup is not None:
                    result["observer_canary"] = probe_cleanup()
                    probe_cleanup = None
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
        if probe_cleanup is not None:
            result["observer_canary"] = probe_cleanup()
        journal.close()
        probes.close()
        write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actions", type=int, default=60)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--chain", action="store_true", help="Start with camera -> drawn ROI -> three ordinary downstream panels")
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
    command = [sys.executable, "-u", "-m", "bench.plot_perf.run_gui_fuzz", "--child", "--seed", str(args.seed), "--actions", str(args.actions), "--frames", str(args.frames), "--output", str(output)]
    if args.inventory:
        command.append("--inventory")
    if args.chain:
        command.append("--chain")
    if args.replay:
        command += ["--replay", str(Path(args.replay).resolve())]
    if args.stop_on_report:
        command += ["--stop-on-report", args.stop_on_report]
    with (output / "console.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print("GUI_FUZZ", child.pid, output, flush=True)
        try:
            status = child.wait(timeout=90+args.actions*2)
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
        print("GUI_FUZZ_EXIT", status, output, flush=True)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
