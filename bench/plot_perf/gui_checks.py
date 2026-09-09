"""Read-only oracles for the real-Qt fuzz driver, not a second state machine.

Call on the Qt owner at a checkpoint; no event pumping, IPC waits or rendering.
Returned values contain only keys/shapes/scalars, never snapshots/front buffers.
``stable=True`` is the driver's claim that the tested action has settled.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import html
import re


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Selector documents are numeric/string contracts, never data arrays.
    raise TypeError(f"unexpected selector/checkpoint value: {type(value).__name__}")


def authored_selectors(panel):
    """Canonical authored selections; excludes host-local revision counters."""
    return {"region": _plain(panel.state.selector),
            "crosshair": _plain(panel.state.crosshair),
            "classifier_thresholds": _plain(panel.state.classifier_thresholds)}


def presented_selectors(card):
    """Canonical selector values carried by the pixels actually installed.

    Compare before/after navigation only at a matching data/kind context:
    live classifier thresholds can legitimately move with their fit.
    """
    surface = None if card is None else card.surface
    front = None if surface is None else surface.presented_front
    return None if front is None else _selector_values(front.interaction.selectors)


def _selector_values(states, *, include_revision=False):
    from zlc_plot.selectors import NumericRange, RectangleRange, CrosshairPoint

    result = []
    for state in states:
        value = state.value
        if isinstance(value, NumericRange):
            summary = [float(value.low), float(value.high)]
        elif isinstance(value, RectangleRange):
            summary = [[float(value.x.low), float(value.x.high)],
                       [float(value.y.low), float(value.y.high)]]
        elif isinstance(value, CrosshairPoint):
            summary = [float(value.x), float(value.y)]
        else:
            summary = float(value)
        item = {"kind": state.kind.value, "facet_index": state.facet_index, "value": summary}
        if include_revision:
            item["revision"] = state.revision
        result.append(item)
    return result


def editor_checkpoint(panel, editor=None):
    """Compare Edit's accepted record with its mounted widget, never a query.

    The driver supplies the actual PanelEditorView. Reading a remote host's
    latest front (or creating its widget) would not prove which pixels Edit
    has installed. Front selectors use display coordinates; the description
    uses canonical coordinates. Compare only kind/facet/revision metadata,
    not those differently measured values. Evidence retains at most eight
    entries in each explicitly labelled space and no data buffers.
    """
    frozen = panel.frozen_data
    description = None if frozen is None else frozen.description
    surface = None if editor is None else editor._surface
    front = None if surface is None else surface.presented_front
    result = {
        "open": panel.editor_open, "widget_observed": editor is not None,
        "configuration_pending": panel.editor_configuration is not None,
        "interaction_pending": surface is not None and (
            surface._pointer_button is not None or surface._gesture_front is not None),
        "snapshot_ref": None, "front": _observed_front(front),
        "accepted_host_id": None if panel.editor_host is None else panel.editor_host.host_id,
        "description": None, "selector_metadata_match": None,
    }
    if frozen is not None:
        ref = frozen.snapshot.ref
        result["snapshot_ref"] = {
            "block_id": str(ref.block_id.value), "schema_fingerprint": ref.schema_fingerprint,
            "generation": str(ref.stream_generation.value), "revision": int(ref.revision.value),
        }
    if description is not None:
        viewport = description.viewport
        result["description"] = {
            "kind": description.kind.value, "size": description.size,
            "focus": description.facet_focus,
            "viewport": None if viewport is None else [
                [viewport.x.low, viewport.x.high], [viewport.y.low, viewport.y.high]],
            "selector_count": len(description.selectors),
            "selector_coordinate_space": "canonical",
            "selectors": _selector_values(description.selectors[:8], include_revision=True),
        }
        if front is not None:
            result["selector_metadata_match"] = (
                [(item.kind, item.facet_index, item.revision) for item in description.selectors]
                == [(item.kind, item.facet_index, item.revision) for item in front.interaction.selectors])
    if front is not None:
        result["front"]["selector_coordinate_space"] = "display"
        result["front"]["selectors"] = _selector_values(
            front.interaction.selectors[:8], include_revision=True)
    return result


def _snapshot_shape(snapshot, source=None):
    """Derive title factors from Dataset axes and compact validity directly.

    Uses only this accepted publication's last written Repeat/Point position.
    Does not call production count/projection helpers or the title formatter.
    The small checkpoint oracle never expands validity over image pixels.
    """
    import numpy as np
    from zlc_data import Valid, Invalid
    from zlc_data.axis import SCALAR

    block, schema = snapshot.block, snapshot.block.schema
    domains = (schema.repeat_domain, schema.point_domain, schema.cell_domain)
    structure = [[(str(axis.name), int(axis.size)) for axis in domain.axes
                  if index != 2 or axis.role != SCALAR]
                 for index, domain in enumerate(domains)]
    validity = block.validity
    repeat, point = schema.repeat_domain, schema.point_domain
    mask = None if isinstance(validity, (Valid, Invalid)) else validity.mask
    work = len(repeat.axes) * (repeat.size + point.size + (0 if mask is None else mask.size))
    row_work = len(repeat.axes) * repeat.size
    unchecked = None
    positions = {}
    landed = []
    if isinstance(validity, Invalid):
        landed = [0 for _axis in repeat.axes]
    elif work > 250_000 or row_work > 2048:
        # This runs on the Qt owner. A large component mask is explicitly
        # unverified, never silently pooled or expanded into a pixel mask.
        landed = None
        unchecked = (f"compact Repeat-count work {work}, row visits bound {row_work}; "
                     "checkpoint budgets are 250000 values and 2048 rows")
    elif repeat.axes:
        for domain in (repeat, point):
            for axis in domain.axes:
                positions[axis.axis_id] = int(domain.codes(axis.axis_id)[-1])
        if source is not None:
            event = source.snapshot.block.schema
            declared = source.canonical_schema or event
            origin = source.cell_origin or (0, 0)
            for index, domain in enumerate((repeat, point)):
                source_domain = (declared.repeat_domain, declared.point_domain)[index]
                event_domain = (event.repeat_domain, event.point_domain)[index]
                last_row = int(origin[index]) + event_domain.size - 1
                by_id = {axis.axis_id: axis for axis in source_domain.axes}
                for axis in domain.axes:
                    if axis.axis_id in by_id:
                        source_axis = by_id[axis.axis_id]
                        coordinate = source_axis.coordinate_at(int(source_domain.codes(axis.axis_id)[last_row]))
                        position = axis.coordinate_position(coordinate)
                        if position is not None:
                            positions[axis.axis_id] = int(position)
        point_rows = np.arange(point.size)
        for axis in point.axes:
            point_rows = point_rows[point.codes(axis.axis_id)[point_rows] == positions[axis.axis_id]]
        codes = [repeat.codes(axis.axis_id) for axis in repeat.axes]
        for target, axis in enumerate(repeat.axes):
            rows = np.arange(repeat.size)
            for other, other_axis in enumerate(repeat.axes):
                if other != target:
                    rows = rows[codes[other][rows] == positions[other_axis.axis_id]]
            if not rows.size or not point_rows.size:
                landed.append(0)
                continue
            if mask is None:
                landed.append(len(set(map(int, codes[target][rows]))))
                continue
            # Other Repeat/Point coordinates are all fixed. Only duplicate
            # physical rows for the SAME complete coordinates may combine;
            # every simultaneously observed component retains its own count.
            observed = {}
            for row in rows:
                coordinate = int(codes[target][row])
                cells = np.any(mask[row, point_rows], axis=0).reshape(-1)
                if coordinate in observed:
                    observed[coordinate] |= cells
                else:
                    observed[coordinate] = cells
            counts = np.sum(list(observed.values()), axis=0)
            low, high = int(counts.min()), int(counts.max())
            landed.append(low if low == high else [low, high])
    sizes, names = [], []
    for index, group in enumerate(structure):
        if group:
            counts = landed if index == 0 else [size for _name, size in group]
            if counts is not None:
                sizes.append("(" + " × ".join(
                    f"{count[0]}–{count[1]}" if isinstance(count, list) else str(count)
                    for count in counts) + ")")
            names.append("(" + " × ".join(name for name, _size in group) + ")")
    return {"structure": _plain(structure), "landed": landed,
            "repeat_counts_status": "unchecked" if unchecked else "checked",
            "repeat_counts_unchecked": unchecked,
            "repeat_count_positions": {str(key.value): value for key, value in positions.items()},
            "domain_shapes": [list(domain.shape) for domain in domains],
            "values_shape": list(block.values.shape),
            "schema_physical_shape": list(schema.physical_shape),
            "title_sizes": None if unchecked else " × ".join(sizes), "title_names": " × ".join(names)}


def panel_checkpoint(panel, card=None, *, editor=None):
    """A small detached summary suitable for one fuzz/replay log entry."""
    accepted = panel.accepted_surface
    description = None if accepted is None else accepted.description
    carrier = None if accepted is None else accepted.plot_input
    snapshot = getattr(carrier, "snapshot", carrier)
    surface = None if card is None else card.surface
    front = None if surface is None else surface.presented_front
    state = panel.state
    lease = panel.history_lease
    parameter_surface = panel.parameter_surface
    result = {"panel_id": panel.panel_id, "signal": state.signal, "kind": state.kind,
              "cell_kind": state.cell_kind, "fit_model": state.fit.get("model"),
              "has_accepted_surface": accepted is not None, "has_front": front is not None,
              "configuration_pending": panel.configuration is not None,
              "surface_in_flight": panel.port is not None and panel.port.surface_busy,
              "interaction_pending": surface is not None and (
                  surface._pointer_button is not None or surface._gesture_front is not None),
              "conditions": [value for value in (panel.reported_condition, panel.vacancy, panel.unapplied_display) if value],
              "lease": None if lease is None else {"signal": lease.signal_name, "window": lease.window, "closed": lease.closed},
              "snapshot_ref": None, "front_identity": None, "shape": None,
              "authored_selectors": authored_selectors(panel), "presented_selectors": presented_selectors(card),
              "repair_keys": {section: [str(field["key"]) for field in parameter_surface.get(section, ())]
                              for section in ("semantic", "fit")},
              "expected_repair_keys": None, "history_requirement": "not-described",
              "accepted_target_matches": accepted is not None and accepted.target.signal == state.signal
                  and accepted.target.kind == state.kind and accepted.target.cell_kind == state.cell_kind}
    result["editor"] = editor_checkpoint(panel, editor)
    result["accepted_host_id"] = None if accepted is None or accepted.host is None else accepted.host.host_id
    if snapshot is not None and hasattr(snapshot, "block"):
        ref = snapshot.ref
        result["snapshot_ref"] = {"block_id": str(ref.block_id.value),
                                  "schema_fingerprint": ref.schema_fingerprint,
                                  "generation": str(ref.stream_generation.value), "revision": int(ref.revision.value)}
        source = None if accepted.publication is None else accepted.publication.value(accepted.target.signal)
        result["shape"] = _snapshot_shape(snapshot, source)
    if front is not None:
        identity = front.identity
        result["front_identity"] = {"host": identity.host_id, "sequence": identity.sequence,
                                    "generation": identity.data_generation, "revision": identity.data_revision,
                                    "kind": identity.kind, "preset": identity.preset,
                                    "display_revision": identity.display_revision,
                                    "layout_revision": identity.layout_revision}
    if description is not None:
        from zlc_plot.specs import history_window_requirement

        fit_keys = ["model"] if description.fit_models or description.fit.get("model") is not None else []
        if description.fit.get("model") is not None:
            fit_keys.append("expression")
        result["expected_repair_keys"] = {
            "semantic": [str(field.name) for field in description.semantics.fields if field.name != "kind"],
            "fit": fit_keys}
        result["accepted_kind"] = description.kind.value
        result["history_requirement"] = history_window_requirement(description.spec, description.display_state.values)
        if not result["accepted_target_matches"]:
            result["expected_repair_keys"] = None
    if card is not None:
        label = card._title_label
        title = html.unescape(re.sub(r"<[^>]*>", "", label.text()))
        form = card._settings_form
        result["card"] = {"structure": _plain(card._parameter_surface.get("data_structure", ())),
                          "landed": _plain(card._parameter_surface.get("data_valid", ())),
                          "title_text": title, "title_tooltip": label.toolTip(),
                          "title_may_be_elided": "…" in title,
                          "repair_keys": {section: [str(field["key"]) for field in card._parameter_surface.get(section, ())]
                                          for section in ("semantic", "fit")},
                          "visible_form_keys": ([field.key for field in form.spec.fields]
                                                if form is not None and form.isVisible() else None)}
    return result


def check_panel(panel, card=None, *, stable=False, before=None, editor=None):
    """Return structured findings, with pending states distinguished from bugs.

    ``before`` is an earlier panel_checkpoint dict, not a retained PanelBinding.
    Empty signal is an intentional disconnect; no initial front is not a bug.
    Background live work may be pending while the last accepted front remains
    perfectly checkable. The helper does not wait for or demand Plane latest.
    """
    current = panel_checkpoint(panel, card, editor=editor)
    findings = []
    unsettled = not stable or current["configuration_pending"] or current["interaction_pending"]

    def report(code, message, **evidence):
        findings.append({"code": code, "severity": "deferred" if unsettled else "error",
                         "panel_id": current["panel_id"], "message": message, "evidence": evidence})

    front, ref = current["front_identity"], current["snapshot_ref"]
    if front is not None and ref is not None:
        if (front["generation"], front["revision"]) != (ref["generation"], ref["revision"]):
            report("front_data_identity", "Presented pixels and accepted snapshot name different data",
                   front=front, snapshot_ref=ref)
        if front["host"] != current.get("accepted_host_id"):
            report("front_host_identity", "Presented front belongs to a different accepted host",
                   front=front, accepted_host=current.get("accepted_host_id"))
    # RasterIdentity exposes generation/revision, not block_id/fingerprint.
    # Keep the full snapshot ref in evidence without claiming those absent
    # fields were independently recovered from RGBA.
    edit = current["editor"]
    edit_front, edit_ref = edit["front"], edit["snapshot_ref"]
    if edit_front is not None and edit_ref is not None:
        edit_pending = unsettled or edit["configuration_pending"] or edit["interaction_pending"]
        mismatches = []
        if (edit_front["generation"], edit_front["revision"]) != (edit_ref["generation"], edit_ref["revision"]):
            mismatches.append("data_identity")
        if edit_front["host"] != edit["accepted_host_id"]:
            mismatches.append("host_identity")
        if edit["selector_metadata_match"] is False:
            mismatches.append("selector_metadata")
        description = edit["description"]
        if description is not None and edit_front["focus"] != description["focus"]:
            mismatches.append("focus")
        if mismatches:
            findings.append({
                "code": "editor_frozen_front", "severity": "deferred" if edit_pending else "error",
                "panel_id": current["panel_id"],
                "message": "Installed Edit pixels and the accepted frozen Save record disagree",
                "evidence": {"mismatched_fields": mismatches, **edit},
            })
    shape = current["shape"]
    card_state = current.get("card")
    if shape is not None:
        if shape["values_shape"] != shape["schema_physical_shape"]:
            report("snapshot_physical_shape", "Shown values disagree with their declared physical domains", shape=shape)
        if card_state is not None and front is not None:
            if (card_state["structure"] != shape["structure"]
                    or shape["landed"] is not None and card_state["landed"] != shape["landed"]):
                report("title_data_structure", "Card title structure does not describe its accepted snapshot",
                       expected=shape, card=card_state)
            if shape["title_sizes"]:
                expected = shape["title_sizes"] + "\n" + shape["title_names"]
                if expected not in card_state["title_tooltip"]:
                    report("title_rendered_structure", "Non-elided title tooltip omits the shown Dataset structure",
                           expected=expected, tooltip=card_state["title_tooltip"],
                           title_may_be_elided=card_state["title_may_be_elided"])

    same_target = before is not None and all(current[key] == before[key]
                                              for key in ("panel_id", "signal", "kind", "cell_kind"))
    expected = current["expected_repair_keys"]
    if expected is None and same_target and current["signal"]:
        # A temporary unavailable/failed source must retain the repair face
        # it already described. New schema/accepted kinds use their new list.
        expected = before["expected_repair_keys"] or before["repair_keys"]
    if expected is not None and current["signal"]:
        for section in ("semantic", "fit"):
            needed = set(expected[section])
            if section == "fit" and same_target and current["fit_model"] != before["fit_model"]:
                needed.discard("expression")
            owners = [("binding", current["repair_keys"])]
            if card_state is not None:
                owners.append(("card", card_state["repair_keys"]))
            for owner, actual in owners:
                missing_keys = sorted(needed - set(actual[section]))
                if missing_keys:
                    report("repair_fields_missing", "Previously/accepted declared repair controls disappeared",
                           owner=owner, section=section, missing=missing_keys)
            if card_state is not None and card_state["visible_form_keys"] is not None:
                missing_keys = sorted(f"{section}__{key}" for key in needed
                                      if f"{section}__{key}" not in card_state["visible_form_keys"])
                if missing_keys:
                    report("repair_widgets_missing", "Visible Setting form omits declared repair widgets", missing=missing_keys)

    if current["accepted_target_matches"] and current["history_requirement"] is None and current["lease"] is not None:
        if not current["lease"]["closed"]:
            report("panel_history_not_released", "This accepted panel no longer requests its own history lease",
                   lease=current["lease"], signal=current["signal"])
    # Histogram window=1 has no demand; Rolling window=1 is still a legal
    # lease, and MEAN trailing may require more. Never inspect/remove another
    # consumer's demand or require the signal's source-index axis to vanish.
    if (same_target and current["signal"] and before["has_front"] and not current["conditions"]
            and (not current["has_front"] or not current["has_accepted_surface"])):
        report("normal_surface_cleared", "A settled same-signal/kind panel lost its previously accepted surface",
               has_front=current["has_front"], has_accepted_surface=current["has_accepted_surface"],
               surface_in_flight=current["surface_in_flight"])
    return findings


def _observed_value(value, depth=0):
    """Bounded intent arguments; unknown objects are types, never repr(data)."""
    from itertools import islice

    if isinstance(value, str):
        return value[:256]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _observed_value(value.value, depth)
    if depth < 3 and isinstance(value, Mapping):
        return {key[:80] if isinstance(key, str) else f"<{type(key).__name__}>":
                _observed_value(item, depth + 1) for key, item in islice(value.items(), 12)}
    if depth < 3 and isinstance(value, (tuple, list)):
        return [_observed_value(item, depth + 1) for item in value[:12]]
    return {"type": type(value).__name__}


def _observed_front(front):
    from zlc_plot.front import RasterFront

    if front is None:
        return None
    if not isinstance(front, RasterFront):
        return {"type": type(front).__name__}
    identity = front.identity
    return {"host": identity.host_id, "sequence": identity.sequence,
            "generation": identity.data_generation, "revision": identity.data_revision,
            "kind": identity.kind, "preset": identity.preset,
            "display_revision": identity.display_revision, "layout_revision": identity.layout_revision,
            "size": list(front.logical_size), "dpr": front.device_pixel_ratio,
            "focus": front.interaction.facet_focus_index,
            "color_limits": (None if front.interaction.color_limits is None else
                             [front.interaction.color_limits.low, front.interaction.color_limits.high]),
            "axis_count": len(front.interaction.axes),
            "axes": [{"role": axis.role, "cell": axis.cell_index, "bounds": list(axis.bounds),
                      "x_limits": list(axis.x_limits), "y_limits": list(axis.y_limits),
                      "canonical_x_limits": list(axis.canonical_x_limits),
                      "canonical_y_limits": list(axis.canonical_y_limits)}
                     for axis in front.interaction.axes[:8]],
            "selector_count": len(front.interaction.selectors),
            "selectors": _selector_values(front.interaction.selectors[:8])}


def install_observers(bench, emit):
    """Observe real intents and QWidget install/virtual paint; return cleanup.

    Patch the actual Qt5PlotWidget CLASS, not an instance paintEvent that Qt's
    virtual dispatch can ignore. No rendering/event delivery is initiated here.
    After a real action/paint, require cleanup.counts['install'] and ['paint']
    to be nonzero: empty bindings are not evidence. cleanup() returns/emits
    this canary result and restores every method/subscription, even on failure.
    Intent records are signal-subscriber observations, not emit-entry hooks;
    existing product slots keep their original order ahead of these observers.
    Set bench.feedback_scope_probe=True BEFORE this call for optional B-side
    Feedback stages, remote description timing and GC spans over 20 ms. This
    does not install anything in A/C or change GC/Numba configuration.
    """
    from time import perf_counter_ns
    from zlc_plot.backends import Qt5PlotWidget

    counts = {"install": 0, "paint": 0, "events": 0, "observer_errors": 0,
              "paint_data_mismatches": 0}
    connections, originals = [], {}
    scope_cleanup = None

    def record(event, **facts):
        counts["events"] += 1
        try:
            emit({"event": event, "probe_sequence": counts["events"], "time_ns": perf_counter_ns(), **facts})
        except Exception:
            # A diagnostic sink must not change a product call's outcome.
            counts["observer_errors"] += 1

    def snapshot(widget):
        try:
            facts = {"widget": id(widget), "visible": widget.isVisible(), "closed": widget._closed,
                     "widget_size": [widget.width(), widget.height()], "front": _observed_front(widget._front)}
            owners = [("console", bench.presenter, bench.view._cards, bench.view._panel_editors)]
            viewer = getattr(bench, "viewer", None)
            if viewer is not None:
                owners.append(("viewer", viewer.presenter._panel_presenter,
                               viewer._view._cards, viewer._view._editors))
            for application, presenter, cards, editors in owners:
                for key, panel in presenter.panels.items():
                    card, editor = cards.get(key), editors.get(key)
                    if card is not None and card.surface is widget:
                        accepted = panel.accepted_surface
                        facts.update(application=application, panel=key, owner="live",
                                     pending=panel.configuration is not None)
                    elif editor is not None and editor._surface is widget:
                        accepted = panel.frozen_data
                        facts.update(application=application, panel=key, owner="editor",
                                     pending=panel.editor_configuration is not None)
                    else:
                        continue
                    if accepted is not None:
                        data = getattr(accepted.plot_input, "snapshot", accepted.plot_input)
                        facts["accepted_data"] = {"generation": data.ref.stream_generation.value,
                                                  "revision": data.ref.revision.value}
                    return facts
            return facts
        except Exception as error:
            counts["observer_errors"] += 1
            return {"widget": id(widget), "snapshot_error": type(error).__name__}

    def watch(name, label):
        original = getattr(Qt5PlotWidget, name)
        originals[name] = original

        def wrapped(widget, *args):
            counts[label] += 1
            facts = snapshot(widget)
            if label == "install":
                try:
                    facts["incoming"] = _observed_front(args[0])
                except Exception as error:
                    counts["observer_errors"] += 1
                    facts["incoming_summary_error"] = type(error).__name__
            record(f"front.{label}.before", **facts)
            try:
                answer = original(widget, *args)
            except BaseException as error:
                record(f"front.{label}.error", **snapshot(widget), exception=type(error).__name__,
                       message=next((arg[:256] for arg in error.args if isinstance(arg, str)), ""))
                raise
            facts = snapshot(widget)
            if label == "paint" and facts.get("visible") and facts.get("accepted_data") and facts.get("front"):
                if any(facts["accepted_data"][key] != facts["front"][key]
                       for key in ("generation", "revision")):
                    counts["paint_data_mismatches"] += 1
            record(f"front.{label}.after", **facts, answer=_observed_value(answer))
            return answer

        setattr(Qt5PlotWidget, name, wrapped)

    def listen(name):
        signal = getattr(bench.view, name)

        def observed(*args):
            try:
                arguments = _observed_value(args)
            except Exception as error:
                counts["observer_errors"] += 1
                arguments = {"summary_error": type(error).__name__}
            record("plot.error.observed" if name == "panel_plot_error" else "intent.observed",
                   signal=name, arguments=arguments)

        signal.connect(observed)
        connections.append((signal, observed))

    def cleanup():
        scope_summary = None if scope_cleanup is None else scope_cleanup()
        for name, original in originals.items():
            setattr(Qt5PlotWidget, name, original)
        originals.clear()
        for signal, callback in connections:
            try:
                signal.disconnect(callback)
            except (TypeError, RuntimeError):
                counts["observer_errors"] += 1
        connections.clear()
        result = {**counts, "passed": counts["install"] > 0 and counts["paint"] > 0
                  and counts["observer_errors"] == 0}
        if scope_summary is not None:
            result["scope_probe"] = scope_summary
        record("observer.canary", **result)
        result["observer_errors"] = counts["observer_errors"]
        result["passed"] = result["passed"] and counts["observer_errors"] == 0
        return result

    cleanup.counts = counts
    try:
        if getattr(bench, "feedback_scope_probe", False):
            scope_cleanup = _install_feedback_scope_probe(bench, record)
        watch("_install_front", "install")
        watch("paintEvent", "paint")
        for name in ("panel_state_changed", "add_panel_requested", "panel_remove_requested",
                     "panel_edit_requested", "panel_snapshot_refresh_requested", "panel_save_figure_requested",
                     "pause_toggled", "selectors_toggled", "panel_order_committed", "panel_plot_error",
                     "logic_start_requested", "logic_stop_requested", "logic_draft_changed"):
            listen(name)
    except BaseException:
        cleanup()
        raise
    return cleanup


def check_scientific_chain(
    plane, *, frames_signal, counts_signal, occupied_signal,
    frame_judged_signal=None, survival_signal=None,
    max_values=100_000,
):
    """Check current exact Camera -> Occupancy -> Frame Survival transactions.

    Each child anchors its OWN latest event and resolves its exact parents;
    independently advancing latest revisions are never joined by number.
    No taps, leases, history, calibration execution or camera-pixel scans.
    Call at a checkpoint, not on every paint. Returned evidence is detached
    and bounded; unavailable parents/oversized payloads are NOT passes.
    This checks published science consistency, not calibration accuracy or
    the displayed overlay (the existing front/paint checks own the latter).
    """
    import numpy as np
    from zlc_data import READOUT_EVENT, SITE, SPATIAL_X, SPATIAL_Y, Valid, Invalid

    if type(max_values) is not int or max_values < 1:
        raise ValueError("max_values must be a positive integer")
    sections, findings = {}, []

    def event_key(ref):
        return {"stream": ref.stream_id.value, "generation": ref.generation.value,
                "sequence": int(ref.sequence)}

    def require(section, condition, code):
        section["checks"] += 1
        if not condition:
            findings.append({"code": code, "severity": "error",
                             "section": section["name"], "event": section["event"]})
        return bool(condition)

    def unchecked(section, reason):
        section["unchecked"].append(reason)

    def latest(label, signal):
        section = {"name": label, "event": None, "checks": 0,
                   "unchecked": [], "signals": {}}
        sections[label] = section
        publication = plane.latest_publication(signal)
        if publication is None:
            unchecked(section, "no publication yet (absent, stopped or awaiting source)")
        else:
            section["event"] = event_key(publication.event_ref)
        return section, publication

    def bundle(section, publication, names):
        values = [publication.value(name) for name in names]
        if not require(section, all(value is not None for value in values), "missing_sibling"):
            return None
        # One bundle establishes sibling causality; shared Dataset identities
        # below are a separate within-publication contract, NOT a source join.
        refs = [value.snapshot.ref for value in values]
        require(section, all((ref.stream_generation, ref.revision) ==
                             (refs[0].stream_generation, refs[0].revision)
                             for ref in refs), "sibling_content_identity")
        require(section, all(value.coverage == values[0].coverage and
                             value.cell_origin == values[0].cell_origin
                             for value in values), "sibling_placement")
        return values

    def parent(section, publication, names):
        try:
            parents = plane.direct_parent_publications(publication)
        except (LookupError, RuntimeError) as error:
            unchecked(section, "exact parent unavailable: " + type(error).__name__)
            return None
        if not require(section, tuple(item.event_ref for item in parents) ==
                       publication.direct_parent_refs, "parent_event_refs"):
            return None
        matching = [item for item in parents if all(item.value(name) is not None for name in names)]
        if len(matching) != 1:
            unchecked(section, "required exact parent payload is absent or not unique")
            return None
        section["parent"] = event_key(matching[0].event_ref)
        return matching[0]

    def small(section, value, *, boolean=False):
        array = np.asarray(value.values)
        ref = value.snapshot.ref
        section["signals"][value.name] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "generation": ref.stream_generation.value, "revision": int(ref.revision.value),
            "block_id": ref.block_id.value,
        }
        if array.size > max_values:
            unchecked(section, "small-result budget exceeded: " + value.name)
            return None
        if not require(section, array.shape == value.schema.physical_shape and array.ndim == 3,
                       "result_physical_shape"):
            return None
        if not require(section, array.dtype.kind == "b" if boolean else array.dtype.kind in "iuf",
                       "result_dtype"):
            return None
        valid = np.asarray(value.snapshot.expanded_validity())
        section["signals"][value.name]["valid"] = int(np.count_nonzero(valid))
        if boolean:
            section["signals"][value.name]["true_valid"] = int(np.count_nonzero(array & valid))
        return array, valid

    def frame_site(section, value):
        schema = value.schema
        return require(section, len(schema.point_domain.axes) == 1 and
                       schema.point_domain.axes[0].role == READOUT_EVENT and
                       len(schema.cell_domain.axes) == 1 and
                       schema.cell_domain.axes[0].role == SITE, "frame_site_axes")

    def same_domains(left, right):
        return all(getattr(left.schema, name) == getattr(right.schema, name)
                   for name in ("repeat_domain", "point_domain", "cell_domain"))

    section, publication = latest("occupancy", counts_signal)
    if publication is not None:
        names = [counts_signal, occupied_signal]
        if frame_judged_signal is not None:
            names.append(frame_judged_signal)
        siblings = bundle(section, publication, names)
        source_event = parent(section, publication, (frames_signal,))
        if siblings is not None:
            counts, occupied = siblings[:2]
            frame_site(section, counts)
            geometry = require(section, same_domains(counts, occupied), "occupancy_sibling_geometry")
            c, o = small(section, counts), small(section, occupied, boolean=True)
            if geometry and c is not None and o is not None:
                cv, cm = c
                ov, om = o
                require(section, np.array_equal(cm, om), "occupancy_sibling_validity")
                require(section, np.isfinite(cv[cm]).all(), "valid_counts_not_finite")
                if cv.dtype.kind == "f":
                    require(section, np.isnan(cv[~cm]).all(), "invalid_counts_not_nan")
                require(section, not ov[~om].any(), "invalid_occupancy_not_false")
            if source_event is not None:
                frames = source_event.value(frames_signal)
                require(section, counts.schema.repeat_domain == frames.schema.repeat_domain and
                        counts.schema.point_domain == frames.schema.point_domain, "camera_frame_identity")
                require(section, tuple(axis.role for axis in frames.schema.cell_domain.axes) ==
                        (SPATIAL_Y, SPATIAL_X), "camera_spatial_axes")
                validity = frames.block.validity
                if frames.schema.repeat_domain.size * frames.schema.point_domain.size > max_values:
                    frame_valid = None
                    unchecked(section, "camera event row count exceeds budget")
                elif isinstance(validity, (Valid, Invalid)):
                    frame_valid = np.full(frames.shape[:2], isinstance(validity, Valid))
                elif validity.mask.size <= max_values:
                    mask = np.asarray(validity.mask)
                    frame_valid = mask.all(axis=tuple(range(2, mask.ndim)))
                else:
                    frame_valid = None
                    unchecked(section, "camera component-validity mask exceeds budget")
                if o is not None and frame_valid is not None and o[1].shape[:2] == frame_valid.shape:
                    require(section, not (o[1] & ~frame_valid[..., None]).any(), "invalid_camera_frame_judged")
                if frame_judged_signal is not None:
                    judged = siblings[2]
                    require(section, judged.schema == frames.schema, "judged_frame_schema")
                    if judged.block.validity is not validity:
                        unchecked(section, "judged validity not shared; full image comparison not attempted")
                    # Runtime restamping preserves the source buffer. Prove
                    # exact same view without scanning multi-megapixel frames.
                    left, right = np.asarray(judged.values), np.asarray(frames.values)
                    same_view = (left.shape == right.shape and left.strides == right.strides and
                                 left.dtype == right.dtype and
                                 left.__array_interface__["data"][0] == right.__array_interface__["data"][0])
                    section["judged_source_same_view"] = same_view
                    if not same_view:
                        unchecked(section, "judged pixels not shared; full image comparison not attempted")

    if survival_signal is not None:
        section, publication = latest("survival", survival_signal)
        if publication is not None:
            output = publication.value(survival_signal)
            require(section, output is not None, "missing_survival_output")
            source_event = parent(section, publication, (occupied_signal,))
            result = None if output is None else small(section, output, boolean=True)
            if source_event is not None and result is not None:
                source = source_event.value(occupied_signal)
                original = small(section, source, boolean=True)
                source_axes = frame_site(section, source)
                output_axes = frame_site(section, output)
                if original is not None and source_axes and output_axes:
                    values, valid = original
                    actual, actual_valid = result
                    n = values.shape[1]
                    geometry = require(section, n >= 2 and actual.shape ==
                        (values.shape[0], n * (n - 1) // 2, values.shape[2]) and
                        source.schema.repeat_domain == output.schema.repeat_domain and
                        source.schema.cell_domain == output.schema.cell_domain, "survival_pair_geometry")
                    if geometry:
                        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                        expected_valid = np.stack([values[:, i] & valid[:, i] & valid[:, j]
                                                   for i, j in pairs], axis=1)
                        expected = expected_valid & np.stack([values[:, j] for i, j in pairs], axis=1)
                        require(section, np.array_equal(actual_valid, expected_valid), "survival_denominator")
                        require(section, np.array_equal(actual, expected), "survival_verdict")
                        axis = source.schema.point_domain.axes[0]
                        coords = [axis.coordinate_at(code) for code in source.schema.point_domain.codes(axis.axis_id)]
                        labels = [value if isinstance(value, str) else "?" if value is None else f"{value:g}"
                                  for value in coords]
                        expected_labels = tuple(f"{labels[i]}-{labels[j]}" for i, j in pairs)
                        require(section, output.schema.point_domain.axes[0].coordinate_labels == expected_labels,
                                "survival_pair_labels")

    for name, section in sections.items():
        section["status"] = ("failed" if any(item["section"] == name for item in findings) else
                             "unchecked" if section["unchecked"] else "checked")
    return {"sections": sections, "findings": findings,
            "status": "failed" if findings else "unchecked" if any(
                section["unchecked"] for section in sections.values()) else "checked",
            "scope": "exact event consistency; calibration accuracy and displayed pixels are not re-evaluated"}


def check_overlay_pixels(panel, card, *, before=None, max_sites=128):
    """Check occupied-ring colour in an installed, exact-paired grey image.

    No rendering, event pumping, newest-status lookup, or RGBA serialization.
    This detects the occupied ring, not the weaker white empty/unknown ring.
    Small/clipped rings and ambiguous inputs are explicitly unchecked.
    """
    import numpy as np
    from matplotlib.colors import to_rgb
    from zlc_data.snapshot_projection import selection_indices, value_selection
    from zlc_plot.config import DEFAULTS
    from zlc_plot.data_contract import resolve_axis
    from zlc_plot.primitives import ImageFrame, PointStatus
    from zlc_plot.rendering import _point_ring_radius
    from zlc_plot.specs import FacetGridPlot, ImagePlot, semantic_spec

    result = {"status": "unchecked", "front": None, "sites": [], "findings": [],
              "unchecked": [], "transitions": [],
              "scope": "installed RGBA occupied-ring chroma; no calibration or native-window proof"}
    accepted = panel.accepted_surface
    surface = None if card is None else card.surface
    front = None if surface is None else surface.presented_front
    frame = None if accepted is None else accepted.plot_input
    description = None if accepted is None else accepted.description
    if front is None or description is None or not isinstance(frame, ImageFrame):
        result["unchecked"].append("no installed front with an accepted ImageFrame")
        return result
    result["front"] = {"host": front.identity.host_id, "sequence": front.identity.sequence,
                       "generation": front.identity.data_generation,
                       "revision": front.identity.data_revision,
                       "overlay_revision": front.identity.image_overlay_revision}
    identity, ref = front.identity, frame.snapshot.ref
    if (accepted.host is None or accepted.host.host_id != identity.host_id
            or str(ref.stream_generation.value) != identity.data_generation
            or int(ref.revision.value) != identity.data_revision
            or frame.overlay.revision != identity.image_overlay_revision):
        result["unchecked"].append("installed pixels do not identify this accepted image/overlay")
        return result
    spec, state = description.spec, description.display_state.values
    if (not isinstance(semantic_spec(spec), ImagePlot)
            or state.get("colormap") != "gray"
            or state.get("presentation", "heatmap") != "heatmap"
            or description.fit.get("model") is not None or front.interaction.selectors
            or panel.configuration is not None or surface._pointer_button is not None
            or surface._gesture_front is not None):
        result["unchecked"].append("requires settled grey Image/Facet Image without fit or selectors")
        return result
    overlay = frame.overlay
    if overlay.status is None or overlay.count < 2:
        result["unchecked"].append("requires dynamic statuses and at least two real sites")
        return result
    facets = None
    if isinstance(spec, FacetGridPlot) and spec.facet is not None:
        schema = frame.snapshot.block.schema
        facet = resolve_axis(schema, spec.facet)
        if spec.facet.domain.value not in ("repeat", "point"):
            result["unchecked"].append("status facet must belong to Repeat or Point")
            return result
        coordinates = np.asarray(facet.coordinates)
        if coordinates.dtype.kind not in "biuf" or not np.all(np.isfinite(coordinates)):
            result["unchecked"].append("non-numeric/non-finite facet ordering is outside this oracle")
            return result
        terms = {resolve_axis(schema, ref).axis_id: coordinate for ref, coordinate in spec.scope}
        try:
            if terms:
                repeats, points, _data = selection_indices(schema, value_selection(schema, terms))
            else:
                repeats, points = range(schema.repeat_domain.size), range(schema.point_domain.size)
        except (ValueError, TypeError) as error:
            result["unchecked"].append(f"facet scope cannot be resolved: {error}")
            return result
        rows = repeats if spec.facet.domain.value == "repeat" else points
        # DataView's retained domains enumerate USED declared codes in ascending
        # code order, not physical row order or a sort of coordinate values.
        # Explicit codes are ordinary camera geometry, not evidence of ambiguity.
        codes = facet.domain.codes(facet.axis_id)[np.asarray(tuple(rows), dtype=np.int64)]
        facets = tuple(facet.coordinates[int(code)] for code in np.unique(codes))
        # The declared values are canonical even with a unit annotation;
        # only the independent display values undergo unit conversion.
        result["facet_coordinates"] = _plain(facets)
    style = DEFAULTS.style.artists
    radius = _point_ring_radius(overlay.coordinates,
                               fraction=style.point_auto_radius_fraction, fallback=float("nan"))
    token = style.point_occupied
    colour = np.asarray(to_rgb(token.color), dtype=float) * 255.0
    direction = colour - colour.mean()
    norm = float(np.dot(direction, direction))
    if not np.isfinite(radius) or radius <= 0 or norm <= 0 or token.alpha <= 0:
        result["unchecked"].append("ring geometry/style has no measurable occupied chroma")
        return result
    rgba = front.buffer.as_rgba(copy=False)
    height, width = rgba.shape[:2]
    axes = tuple(axis for axis in front.interaction.axes
                 if axis.role in ("main", "image", "facet_cell"))
    if not axes:
        result["unchecked"].append("no installed image axes")
        return result
    old_front = (before or {}).get("front") or {}
    old_sites = ((before or {}).get("sites", ()) if
                 old_front.get("host") == identity.host_id and
                 old_front.get("generation") == identity.data_generation else ())
    previous = {(site["cell"], site["site"]): site for site in old_sites
                if site.get("observed") is not None}
    for axis in axes:
        cell = axis.cell_index
        if axis.x_scale != "linear" or axis.y_scale != "linear":
            result["unchecked"].append(f"cell {cell}: nonlinear mapping")
            continue
        if facets is not None and (cell is None or not 0 <= cell < len(facets)):
            result["unchecked"].append(f"cell {cell}: unknown facet coordinate")
            continue
        facet_value = None if facets is None else facets[cell]
        statuses = overlay.statuses_for(spec, facet_value)
        if statuses is None:
            result["unchecked"].append(f"cell {cell}: no uniquely selected shot status")
            continue
        left, top, right, bottom = axis.bounds
        x0, x1 = axis.canonical_x_limits
        y0, y1 = axis.canonical_y_limits
        if x1 == x0 or y1 == y0:
            result["unchecked"].append(f"cell {cell}: degenerate canonical bounds")
            continue
        # Invert the installed linear canonical transform, not a current view.
        scale_x = width * (right - left) / (x1 - x0)
        scale_y = -height * (bottom - top) / (y1 - y0)
        rx, ry = abs(radius * scale_x), abs(radius * scale_y)
        stroke_width = token.linewidth * front.logical_dpi * front.device_pixel_ratio / 72.0
        # The stroke straddles its centreline. One physical pixel bounds AA;
        # a whole linewidth here used to turn small annuli into filled discs.
        padding = 0.5 * stroke_width + 1.0
        labelled = bool(state.get("show_point_labels", True)) and any(
            status is PointStatus.OCCUPIED and (
                (overlay.labels is not None and overlay.labels[i]) or
                (overlay.point_ids is not None and overlay.point_ids[i]))
            for i, status in enumerate(statuses))
        for index, (point, status) in enumerate(zip(overlay.coordinates, statuses)):
            if len(result["sites"]) >= max_sites:
                result["unchecked"].append(f"site budget {max_sites} reached")
                break
            px = width * left + (float(point[0]) - x0) * scale_x
            py = height * top + (float(point[1]) - y1) * scale_y
            point_id = str(index) if overlay.point_ids is None else overlay.point_ids[index]
            site = {"cell": cell, "site": point_id, "expected": status.value,
                    "expected_occupied": status is PointStatus.OCCUPIED, "observed": None,
                    "center": [round(px, 2), round(py, 2)],
                    "radius_px": [round(rx, 3), round(ry, 3)],
                    "stroke_halfwidth_plus_aa_px": round(padding, 3)}
            result["sites"].append(site)
            if (min(rx, ry) < 4.0 or px - rx - padding <= width * left
                    or px + rx + padding >= width * right
                    or py - ry - padding <= height * top
                    or py + ry + padding >= height * bottom):
                site["unchecked"] = "ring is subpixel-sized or clipped by image axes"
                continue
            if labelled:
                # RasterFront carries no glyph bounds. Labels use the SAME
                # colour as occupied rings, so chroma cannot exclude a label
                # overlapping this site. Keep this conservative until the
                # operator uses the existing Point labels display switch.
                site["unchecked"] = "point labels may overlap; their paint bounds are not in this front"
                continue
            distances = np.hypot((overlay.coordinates[:, 0] - point[0]) * scale_x,
                                 (overlay.coordinates[:, 1] - point[1]) * scale_y)
            distances[index] = np.inf
            clearance = float(np.min(distances))
            site["nearest_site_distance_px"] = round(clearance, 3)
            # Conservative enclosing circles also cover anisotropic ellipses.
            # Do not attribute another site's real stroke to this one.
            if clearance <= 2.0 * (max(rx, ry) + padding):
                site["unchecked"] = "neighbour ring stroke/AA can intersect the detection annulus"
                continue
            ix0, ix1 = int(np.floor(px - rx - padding)), int(np.ceil(px + rx + padding)) + 1
            iy0, iy1 = int(np.floor(py - ry - padding)), int(np.ceil(py + ry + padding)) + 1
            rgb = rgba[iy0:iy1, ix0:ix1, :3].astype(float)
            yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
            dx, dy = (xx + 0.5 - px) / rx, (yy + 0.5 - py) / ry
            band = abs(np.sqrt(dx * dx + dy * dy) - 1.0) <= padding / min(rx, ry)
            chroma = rgb - rgb.mean(axis=2, keepdims=True)
            amount = np.sum(chroma * direction, axis=2) / norm
            error = np.linalg.norm(chroma - amount[..., None] * direction, axis=2)
            # Grey compositing preserves the token's chroma direction. Allow
            # byte-rounding error, but require visible (>= 8/255) colour spread.
            spread = np.ptp(rgb, axis=2)
            orange = band & (amount > 0) & (spread >= 8) & (error <= 3.0)
            foreign = band & (spread >= 8) & ~orange
            sectors = (dx >= 0).astype(np.int8) + 2 * (dy >= 0).astype(np.int8)
            hits = [int(np.count_nonzero(orange & (sectors == sector))) for sector in range(4)]
            site.update(coloured_pixels=int(np.count_nonzero(orange)), quadrant_hits=hits)
            if np.any(foreign):
                site["unchecked"] = "another chromatic foreground/background intersects the ring"
                continue
            if not np.any(orange):
                site["observed"] = False
            elif sum(hit >= 2 for hit in hits) >= 3:
                site["observed"] = True
            else:
                site["unchecked"] = "too little ring coverage to distinguish marker from interference"
                continue
            if site["observed"] != site["expected_occupied"]:
                result["findings"].append({"cell": cell, "site": point_id,
                                           "expected": site["expected_occupied"],
                                           "observed": site["observed"]})
            old = previous.get((cell, point_id))
            if (old is not None and old["expected_occupied"] != site["expected_occupied"]
                    and old.get("center") == site["center"]):
                result["transitions"].append({"cell": cell, "site": point_id,
                    "expected": [old["expected_occupied"], site["expected_occupied"]],
                    "observed": [old["observed"], site["observed"]]})
    for site in result["sites"]:
        site["status"] = ("unchecked" if "unchecked" in site or site["observed"] is None else
                          "checked" if site["observed"] == site["expected_occupied"] else "failed")
    result["site_counts"] = {status: sum(site["status"] == status for site in result["sites"])
                             for status in ("checked", "failed", "unchecked")}
    unchecked = result["unchecked"] or any("unchecked" in site for site in result["sites"])
    result["status"] = ("failed" if result["findings"] else "unchecked" if unchecked
                        or not result["sites"] else "checked")
    return result


def _install_feedback_scope_probe(bench, record):
    """Temporary coarse B-process scopes; no payloads, GUI calls or new work.

    Futures are observed only after completion, never awaited. Class wrappers
    avoid storing a bound-method closure on a host (which would manufacture
    the very GC cycles this probe investigates). A/C PIDs come from the two
    existing factory closures; there is deliberately no child injection/JIT
    claim. All patches and the GC callback are removed by the outer cleanup.
    """
    import gc
    import inspect
    import itertools
    import os
    import threading
    import weakref
    from time import perf_counter_ns, thread_time_ns
    from zlc_atom.nodes.slm_feedback import task as feedback
    from zlc_plot.render_process import RenderProcess, _RemoteRasterPlotHost

    active = True
    patches, totals, gc_starts = [], {}, {}
    serial = itertools.count(1)
    lock = threading.RLock()
    first_created, first_front = weakref.WeakSet(), weakref.WeakSet()
    gc_counts = {"cycles": 0, "slow_spans": 0, "slow_ns": 0}

    def emit(event, **facts):
        if active:
            record(event, process_role="B", pid=os.getpid(),
                   thread=threading.get_ident(), native_thread=threading.get_native_id(), **facts)

    def finish(name, token, started, cpu_started, facts, exception=None):
        elapsed, cpu = perf_counter_ns() - started, thread_time_ns() - cpu_started
        with lock:
            row = totals.setdefault(name, {"calls": 0, "wall_ns": 0, "thread_cpu_ns": 0})
            row["calls"] += 1
            row["wall_ns"] += elapsed
            row["thread_cpu_ns"] += cpu
        emit("scope.end", scope=name, call=token, start_ns=started, elapsed_ns=elapsed,
             thread_cpu_ns=cpu, exception=exception, **facts)

    def timed(original, name, describe, result_facts=None):
        def wrapped(*args, **kwargs):
            token = next(serial)
            facts = describe(args, kwargs)
            emit("scope.begin", scope=name, call=token, **facts)
            started, cpu_started = perf_counter_ns(), thread_time_ns()
            exception = None
            try:
                answer = original(*args, **kwargs)
                if result_facts is not None:
                    facts.update(result_facts(answer))
                return answer
            except BaseException as error:
                exception = type(error).__name__
                raise
            finally:
                finish(name, token, started, cpu_started, facts, exception)
        return wrapped

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        patches.append((owner, name, original, replacement))
        setattr(owner, name, replacement)

    def host_facts(host):
        return {"host": host.host_id, "remote_pid": host.process_pid,
                "remote_name": host.process_name}

    def describe_call(original):
        def wrapped(process, host, method, args, kwargs):
            if method != "describe_display":
                return original(process, host, method, args, kwargs)
            facts = host_facts(host)
            token, started = next(serial), perf_counter_ns()
            emit("scope.begin", scope="describe_display", call=token, **facts)
            try:
                pending = original(process, host, method, args, kwargs)
            except BaseException as error:
                emit("scope.future.end", scope="describe_display", call=token,
                     elapsed_ns=perf_counter_ns() - started, exception=type(error).__name__, **facts)
                raise
            emit("scope.future.queued", scope="describe_display", call=token,
                 future=id(pending), enqueue_ns=perf_counter_ns() - started, **facts)

            def completed(done):
                if not active:
                    return
                cancelled = done.cancelled()
                # add_done_callback guarantees completion; exception() here
                # cannot block and does not inspect the operation payload.
                error = None if cancelled else done.exception()
                emit("scope.future.end", scope="describe_display", call=token,
                     future=id(done), elapsed_ns=perf_counter_ns() - started,
                     cancelled=cancelled, exception=None if error is None else type(error).__name__, **facts)

            pending.add_done_callback(completed)
            return pending
        return wrapped

    def created(original):
        def wrapped(host, description):
            answer = original(host, description)
            if host not in first_created:
                first_created.add(host)
                emit("scope.remote.initial_metadata", **host_facts(host))
            return answer
        return wrapped

    def accepted_front(original):
        def wrapped(host, front):
            answer = original(host, front)
            if host not in first_front and host.front is front:
                first_front.add(host)
                identity = front.identity
                emit("scope.remote.first_front", sequence=identity.sequence,
                     generation=identity.data_generation, revision=identity.data_revision,
                     **host_facts(host))
            return answer
        return wrapped

    def garbage_collection(phase, info):
        generation = int(info["generation"])
        if phase == "start":
            gc_starts[generation] = perf_counter_ns()
        elif phase == "stop":
            started = gc_starts.pop(generation, None)
            gc_counts["cycles"] += 1
            if started is not None:
                elapsed = perf_counter_ns() - started
                if elapsed > 20_000_000:
                    gc_counts["slow_spans"] += 1
                    gc_counts["slow_ns"] += elapsed
                    emit("scope.gc", generation=generation, start_ns=started, elapsed_ns=elapsed,
                         collected=int(info["collected"]), uncollectable=int(info["uncollectable"]))

    def cleanup():
        nonlocal active
        if not active:
            return {"enabled": True, "totals": dict(totals), "gc": dict(gc_counts)}
        active = False
        if garbage_collection in gc.callbacks:
            gc.callbacks.remove(garbage_collection)
        for owner, name, original, replacement in reversed(patches):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
        patches.clear()
        gc_starts.clear()
        return {"enabled": True, "totals": {name: dict(row) for name, row in totals.items()},
                "gc": dict(gc_counts), "A_compile_observed": False}

    try:
        services = {}
        for role, attribute, cell in (("A", "_make_monitor_host", "monitor_render"),
                                      ("C", "_make_editor_host", "editor_render")):
            factory = getattr(bench.presenter, attribute)
            service = inspect.getclosurevars(factory).nonlocals.get(cell)
            services[role] = (None if service is None else {
                "pid": service._process.pid, "name": service.name, "owner_id": id(service)})
        emit("scope.services", services=services, A_compile_observed=False)
        for name in ("_readout_frames", "_fit_contrasts"):
            def dimensions(args, kwargs):
                value = args[0]
                shape = (value.block.values.shape if hasattr(value, "block") else value.shape)
                return {"input_id": id(value), "shape": list(shape), "shots": kwargs.get("shots")}
            patch(feedback, name, timed(getattr(feedback, name), name, dimensions))
        patch(feedback.SlmFeedbackTask, "_shoot", timed(feedback.SlmFeedbackTask._shoot, "_shoot",
            lambda args, kwargs: {"object_id": id(args[0]), "node": args[0].instance_id,
                                  "iteration": args[3], "shots": args[0].shots}))
        factory = bench.presenter._make_monitor_host
        patch(bench.presenter, "_make_monitor_host", timed(factory, "_make_monitor_host",
            lambda args, kwargs: {"input_id": id(args[0]), "signal": args[1].signal,
                                  "kind": args[1].kind},
            lambda host: host_facts(host) if isinstance(host, _RemoteRasterPlotHost)
                         else {"returned_type": type(host).__name__}))
        patch(RenderProcess, "_call", describe_call(RenderProcess._call))
        patch(_RemoteRasterPlotHost, "_created", created(_RemoteRasterPlotHost._created))
        patch(_RemoteRasterPlotHost, "_accept_front", accepted_front(_RemoteRasterPlotHost._accept_front))
        gc.callbacks.append(garbage_collection)
    except BaseException:
        cleanup()
        raise
    return cleanup
