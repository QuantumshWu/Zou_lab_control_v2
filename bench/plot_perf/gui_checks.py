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


def _snapshot_shape(snapshot):
    """Derive title factors from Dataset axes and compact validity directly.

    Does not call Workbench panel_data_shape or the card's title formatter.
    Explicit sparse-domain row counts are NOT products of coordinate sizes.
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
    repeat = schema.repeat_domain
    landed = []
    if isinstance(validity, Invalid):
        landed = [0 for _axis in repeat.axes]
    else:
        rows = (np.arange(repeat.size) if isinstance(validity, Valid) else
                np.flatnonzero(np.asarray(validity.mask).reshape(repeat.size, -1).any(axis=1)))
        for axis in repeat.axes:
            # One coordinate counts if any stored row for it landed. This
            # also handles missing/duplicate combinations in explicit codes.
            landed.append(int(np.unique(repeat.codes(axis.axis_id)[rows]).size))
    sizes, names = [], []
    for index, group in enumerate(structure):
        if group:
            counts = landed if index == 0 else [size for _name, size in group]
            sizes.append("(" + " × ".join(map(str, counts)) + ")")
            names.append("(" + " × ".join(name for name, _size in group) + ")")
    return {"structure": _plain(structure), "landed": landed,
            "domain_shapes": [list(domain.shape) for domain in domains],
            "values_shape": list(block.values.shape),
            "schema_physical_shape": list(schema.physical_shape),
            "title_sizes": " × ".join(sizes), "title_names": " × ".join(names)}


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
        result["shape"] = _snapshot_shape(snapshot)
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
            if card_state["structure"] != shape["structure"] or card_state["landed"] != shape["landed"]:
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
    """
    from time import perf_counter_ns
    from zlc_plot.backends import Qt5PlotWidget

    counts = {"install": 0, "paint": 0, "events": 0, "observer_errors": 0,
              "paint_data_mismatches": 0}
    connections, originals = [], {}

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
            for key, panel in bench.presenter.panels.items():
                card = bench.view._cards.get(key)
                editor = bench.view._panel_editors.get(key)
                if card is not None and card.surface is widget:
                    accepted = panel.accepted_surface
                    facts.update(panel=key, owner="live", pending=panel.configuration is not None)
                elif editor is not None and editor._surface is widget:
                    accepted = panel.frozen_data
                    facts.update(panel=key, owner="editor", pending=panel.editor_configuration is not None)
                else:
                    continue
                if accepted is not None:
                    data = getattr(accepted.plot_input, "snapshot", accepted.plot_input)
                    facts["accepted_data"] = {"generation": data.ref.stream_generation.value,
                                              "revision": data.ref.revision.value}
                break
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
        record("observer.canary", **result)
        result["observer_errors"] = counts["observer_errors"]
        result["passed"] = result["passed"] and counts["observer_errors"] == 0
        return result

    cleanup.counts = counts
    try:
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
