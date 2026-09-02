"""Profile the real four-panel MOT ROI chain in TaskConsole.

The scenario is intentionally specific: CameraMeasurement's own FacetGrid
preview, a 40x500 Area-derived ROI, a 40-shot Histogram history lease, a
source-index FacetGrid of Image, Curve or Histogram cells with live fits, and
a Curve over source index grouped by the ROI's 40-sample spatial-y axis.

Run from the repository root on the real display::

    python -m bench.plot_perf.run_mot_roi_chain --seconds 15
    python -m bench.plot_perf.run_mot_roi_chain --panel3 histogram \
        --fit-model bimodal_poisson_gaussian
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
import threading
import time

import numpy as np

from . import guards, probe
from .common import Pointer, axis_center, provenance, stats, write_result
from .run_console import ConsoleBench, render_cost


def _normalized(text: object) -> str:
    return str(text).strip().lower().replace("_", "-")


def _semantic_field(panel, label: str) -> dict:
    wanted = _normalized(label)
    fields = tuple(panel.parameter_surface.get("semantic", ()))
    matches = [field for field in fields if _normalized(field.get("label")) == wanted]
    if len(matches) != 1:
        raise guards.HarnessError(
            f"{panel.panel_id} has {len(matches)} semantic rows called {label!r}: "
            f"{[field.get('label') for field in fields]!r}"
        )
    return dict(matches[0])


def _role_value(field: dict, role: str) -> object:
    for label, value in tuple(field.get("choices") or ()):
        if value == role or _normalized(label).strip("()") == _normalized(role):
            return value
    raise guards.HarnessError(
        f"semantic row {field.get('label')!r} offers no {role!r} fate: "
        f"{field.get('choices')!r}"
    )


def _assign_roles(bench: ConsoleBench, panel, assignments: dict[str, str]) -> dict:
    changes = {}
    expected = {}
    for axis_label, role in assignments.items():
        field = _semantic_field(panel, axis_label)
        key = str(field["key"])
        value = _role_value(field, role)
        expected[key] = value
        if panel.state.semantic.get(key) != value:
            changes[key] = value
    if changes:
        bench.edit_setting(panel, "semantic", **changes)
    held = dict(panel.state.semantic)
    refused = {key: held.get(key) for key, value in expected.items() if held.get(key) != value}
    if refused:
        raise guards.HarnessError(
            f"{bench.label(panel)} semantic roles were not accepted: {refused!r}"
        )
    return {field: held.get(key) for field, key in (
        (label, str(_semantic_field(panel, label)["key"])) for label in assignments
    )}


def _image_payload(bench: ConsoleBench, panel):
    payload = bench.renderer(panel)._last_payload
    cells = tuple(getattr(payload, "cells", ()))
    if cells:
        payload = getattr(cells[0], "payload", cells[0])
    if not all(hasattr(payload, name) for name in ("x", "y", "z")):
        raise guards.HarnessError("Camera AutoPlot is not holding an Image payload")
    return payload


def _image_transform(bench: ConsoleBench, panel):
    widget = bench.surface(panel)
    front = None if widget is None else widget.presented_front
    if front is None:
        raise guards.HarnessError("Camera AutoPlot has no presented front")
    candidates = [
        axis
        for axis in front.interaction.axes
        if axis.role in {"main", "facet_cell"}
    ]
    session = panel.host._session
    active = bench.renderer(panel).primary_axes
    matched = [
        transform
        for transform in candidates
        if session._axis_for_transform(transform) is active
    ]
    if len(matched) != 1:
        raise guards.HarnessError(
            "Camera AutoPlot has no unique primary image transform: "
            f"{[(item.role, item.cell_index) for item in candidates]!r}"
        )
    return matched[0]


def _focus_autoplot_cell(bench: ConsoleBench, panel) -> None:
    if panel.host._session.facet_focus_index is not None:
        return
    from PyQt5 import QtCore, QtGui

    widget = bench.surface(panel)
    transform = _image_transform(bench, panel)
    pointer = Pointer(widget, bench.app, QtCore, QtGui)
    pointer.post = True
    pointer.dclick(*axis_center(transform))
    bench._until(
        lambda: (
            panel.host._session.facet_focus_index is not None
            and panel.state.focused_cell is not None
            and bench.surface(panel) is not None
            and bench.surface(panel).presented_front is not None
            and bench.surface(panel).presented_front.interaction.facet_focus_index
            is not None
        ),
        "Camera AutoPlot focused cell",
    )
    bench._pump(0.5)


def _draw_mot_roi(
    bench: ConsoleBench,
    panel,
    *,
    height: int = 40,
    width: int = 500,
) -> dict:
    """Draw the requested ROI through real QMouseEvents and publish roi_frame."""

    from PyQt5 import QtCore, QtGui

    _focus_autoplot_cell(bench, panel)
    payload = _image_payload(bench, panel)
    x = np.asarray(getattr(payload.x, "display", payload.x), dtype=float).reshape(-1)
    y = np.asarray(getattr(payload.y, "display", payload.y), dtype=float).reshape(-1)
    z = np.asarray(getattr(payload.z, "display", payload.z), dtype=float)
    valid = np.broadcast_to(
        np.asarray(getattr(payload, "valid", np.ones(z.shape, dtype=bool)), dtype=bool),
        z.shape,
    )
    if z.shape != (y.size, x.size) or y.size < height or x.size < width:
        raise guards.HarnessError(
            f"MOT image {z.shape!r} cannot carry a {height}x{width} ROI"
        )
    score = np.where(valid & np.isfinite(z), z, -np.inf)
    peak_row, peak_column = np.unravel_index(int(np.argmax(score)), score.shape)
    row_start = int(np.clip(peak_row - height // 2, 0, y.size - height))
    column_start = int(np.clip(peak_column - width // 2, 0, x.size - width))
    row_stop = row_start + height
    column_stop = column_start + width
    x_bounds = tuple(sorted((float(x[column_start]), float(x[column_stop - 1]))))
    y_bounds = tuple(sorted((float(y[row_start]), float(y[row_stop - 1]))))

    transform = _image_transform(bench, panel)
    first = transform.display_to_normalized(x_bounds[0], y_bounds[0])
    second = transform.display_to_normalized(x_bounds[1], y_bounds[1])
    widget = bench.surface(panel)
    pointer = Pointer(widget, bench.app, QtCore, QtGui)
    pointer.post = True
    answers = []

    def observe(result) -> None:
        answers.append(
            {
                "action": str(result[1]),
                "error": None if result[4] is None else str(result[4]),
                "has_state": result[2] is not None,
            }
        )

    widget._gesture_ready.connect(observe)
    before = guards.committed_region(panel)
    try:
        pointer.press(*first, button=QtCore.Qt.LeftButton)
        bench._pump(0.05)
        pointer.move(*second)
        bench._pump(0.05)
        pointer.release(*second, button=QtCore.Qt.LeftButton)
        bench._pump(0.8)
    finally:
        try:
            widget._gesture_ready.disconnect(observe)
        except TypeError:
            pass
    after = guards.committed_region(panel)
    if before == after:
        raise guards.HarnessError(
            "MOT ROI drag was not committed: "
            f"first={first!r}, second={second!r}, "
            f"transform={(transform.role, transform.cell_index, transform.bounds)!r}, "
            f"answers={answers!r}"
        )

    bench.presenter.update_panel_published_outputs(panel.panel_id, {"roi_frame": True})
    name = f"@logic/{panel.panel_id}/roi_frame"
    bench._until(
        lambda: bench.session.signal_plane.latest_publication(name) is not None,
        "MOT ROI signal",
    )
    snapshot = bench.session.signal_plane.current_dataset(name)
    actual_shape = tuple(map(int, snapshot.block.values.shape))
    tolerance = (
        max(2, int(round(height * 0.15))),
        max(2, int(round(width * 0.15))),
    )
    if any(
        abs(actual - wanted) > allowed
        for actual, wanted, allowed in zip(
            actual_shape[-2:], (height, width), tolerance, strict=True
        )
    ):
        raise guards.HarnessError(
            f"requested {height}x{width} ROI, product published {actual_shape!r}"
        )
    return {
        "signal": name,
        "peak_index_yx": [int(peak_row), int(peak_column)],
        "bounds_xy": [list(x_bounds), list(y_bounds)],
        "shape": list(actual_shape),
    }


def _source_index_size(bench: ConsoleBench, signal: str) -> int:
    from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

    snapshot = bench.session.signal_plane.current_dataset(signal)
    column = next(
        (
            item
            for item in snapshot.block.schema.point_table.columns
            if item.coordinate_id == PRIMARY_INDEX_AXIS_ID
        ),
        None,
    )
    return 0 if column is None else len(set(column.values))


def _revision(bench: ConsoleBench, signal: str) -> int | None:
    value = bench.session.signal_plane.freeze().value(signal)
    ref = getattr(getattr(value, "snapshot", value), "ref", None)
    revision = getattr(ref, "revision", None)
    return None if revision is None else int(revision.value)


class CausalTimeline:
    """Low-overhead prepare-to-Qt-accept timing keyed by causal root."""

    def __init__(self, bench: ConsoleBench) -> None:
        self._bench = bench
        self._lock = threading.Lock()
        self._events: dict[object, dict[str, dict[str, float]]] = {}

    def _key(self, publication) -> object:
        roots = self._bench.session.signal_plane.publication_roots(publication)
        return frozenset(roots)

    def attach(self, panel, label: str) -> None:
        port = panel.port
        original_prepare = port.prepare
        original_accept = port.accept

        def prepare(value, publication, front):
            now = time.perf_counter()
            key = self._key(publication)
            with self._lock:
                slot = self._events.setdefault(key, {}).setdefault(label, {})
                slot.setdefault("prepare", now)
            return original_prepare(value, publication, front)

        def accept(update, operation):
            result = original_accept(update, operation)
            surface = port.accepted_surface()
            if surface is not None and surface.publication is update.publication:
                now = time.perf_counter()
                key = self._key(update.publication)
                with self._lock:
                    self._events.setdefault(key, {}).setdefault(label, {})[
                        "accept"
                    ] = now
            return result

        port.prepare = prepare
        port.accept = accept

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def summary(self, labels: tuple[str, ...]) -> dict:
        with self._lock:
            events = {
                key: {name: dict(stages) for name, stages in rows.items()}
                for key, rows in self._events.items()
            }
        per_panel = defaultdict(list)
        cohort, prepare_skew, accept_skew = [], [], []
        complete = 0
        for rows in events.values():
            for label, stages in rows.items():
                if "prepare" in stages and "accept" in stages:
                    per_panel[label].append(stages["accept"] - stages["prepare"])
            if not all(
                label in rows
                and "prepare" in rows[label]
                and "accept" in rows[label]
                for label in labels
            ):
                continue
            complete += 1
            starts = [rows[label]["prepare"] for label in labels]
            stops = [rows[label]["accept"] for label in labels]
            cohort.append(max(stops) - min(starts))
            prepare_skew.append(max(starts) - min(starts))
            accept_skew.append(max(stops) - min(stops))
        return {
            "causal_roots_seen": len(events),
            "complete_four_panel_roots": complete,
            "panel_prepare_to_accept": {
                label: stats(per_panel[label]) for label in labels
            },
            "four_panel_critical_path": stats(cohort),
            "prepare_start_skew": stats(prepare_skew),
            "accept_finish_skew": stats(accept_skew),
        }


def _seam(rows: list[dict], name: str) -> dict | None:
    return next((row for row in rows if row["seam"] == name), None)


def _stage_summary(rows: list[dict], labels: tuple[str, ...], frames: dict) -> list[dict]:
    render_rows = {row["panel"]: row for row in render_cost(rows, frames)}
    answer = []
    for label in labels:
        names = {
            "route_materialize": f"PanelPort[{label}]._project_input",
            "data_projection": f"PlotSession[{label}]._prepare_live_frame_worker",
            "fit_total": f"PlotSession[{label}]._solve_live_pair",
            "fit_batch_setup": f"PlotSession[{label}]._fit_facet_batch",
            "numeric_fit_batch": f"FitEngine[{label}].fit_batch",
            "numeric_fit_single": f"FitEngine[{label}].fit",
            "commit_and_render": f"PlotSession[{label}].commit_live_frame",
            "renderer_present": f"MatplotlibRenderer[{label}].present",
            "compose": f"MatplotlibRenderer[{label}]._compose_frame",
            "qt_accept": f"PanelPort[{label}]._put_on_screen",
            "qt_widget": f"QtWidget[{label}].present_front",
        }
        stages = {}
        for stage, seam_name in names.items():
            row = _seam(rows, seam_name)
            if row is not None:
                stages[stage] = {
                    "calls": row["calls"],
                    "wall_ms_per_call": row["gross_ms_per_call"],
                    "cpu_ms_per_call": row["gross_cpu_ms_per_call"],
                    "self_ms_per_call": row["self_ms_per_call"],
                }
        answer.append(
            {
                "panel": label,
                "frames": frames.get(label, 0),
                "stages": stages,
                "render_self_rollup": render_rows.get(label),
            }
        )
    return answer


def _window_revision_rate(
    bench: ConsoleBench,
    signal: str,
    timeline: CausalTimeline,
):
    state = {}

    def begin() -> None:
        timeline.reset()
        state["start"] = _revision(bench, signal)

    def finish(elapsed: float) -> dict:
        state["stop"] = _revision(bench, signal)
        start, stop = state.get("start"), state.get("stop")
        count = 0 if start is None or stop is None else max(0, stop - start)
        return {
            "start_revision": start,
            "stop_revision": stop,
            "revisions": count,
            "per_second": round(count / elapsed, 2) if elapsed else 0.0,
        }

    return begin, finish


#: The fit each panel3 cell kind is measured under when none is named: the
#: default model of that cell's fit target.
_PANEL3_FITS = {
    "image": "anisotropic_gaussian_center",
    "curve": "gaussian_offset",
    "histogram": "bimodal_gaussian",
}
_PANEL3_LABELS = {
    "image": "panel3 facet-fit-40",
    "curve": "panel3 facet-curve-fit-40",
    "histogram": "panel3 facet-histogram-fit-40",
}


def run(
    *,
    seconds: float,
    baseline_seconds: float,
    panel3: str,
    fit_model: str | None = None,
) -> dict:
    if panel3 not in _PANEL3_FITS:
        raise ValueError("panel3 must be 'image', 'curve' or 'histogram'")
    panel3_label = _PANEL3_LABELS[panel3]
    fit_model = fit_model or _PANEL3_FITS[panel3]
    labels = (
        "panel1 camera-grid",
        "panel2 histogram-40",
        panel3_label,
        "panel4 curve-40",
    )
    bench = ConsoleBench()
    payload = {
        "scenario": "mot-roi-history-four-panel",
        "provenance": provenance(),
        "requested": {
            "camera": "mot_camera",
            "exposure_seconds": 0.1,
            "roi_yx": [40, 500],
            "history_window": 40,
            "panel_size": "2x2",
            "panel3": panel3,
            "fit": fit_model,
            "numba_threads": int(os.environ.get("NUMBA_NUM_THREADS", "0") or 0),
            "numba_worker_threads": int(
                os.environ.get("ZLC_NUMBA_WORKER_THREADS", "0") or 0
            ),
        },
    }
    with bench:
        bench.start(
            camera="mot_camera",
            exposure=0.1,
            clear_preview_panels=False,
        )
        panels = tuple(bench.presenter.panels.values())
        if len(panels) != 1:
            raise guards.HarnessError(
                f"CameraMeasurement opened {len(panels)} previews, expected one"
            )
        camera = panels[0]
        if camera.state.kind != "facet_grid" or camera.state.signal != bench.signal:
            raise guards.HarnessError(
                "CameraMeasurement preview is not its own FacetGrid frames panel"
            )
        bench._labels[camera.panel_id] = labels[0]
        bench._kinds[camera.panel_id] = "facet_grid:image"
        bench._until(
            lambda: (
                camera.host is not None
                and bench.surface(camera) is not None
                and bench.surface(camera).presented_front is not None
            ),
            "CameraMeasurement AutoPlot surface",
        )
        # The preview can finish mounting after ConsoleBench.start enabled
        # selectors.  Re-state the product toggle on the fully mounted card;
        # an interaction-disabled widget would make every synthetic drag a
        # no-op and the benchmark guard must never call that a chain.
        bench.presenter.set_deriving(True)
        bench._pump(0.3)
        if not bench.surface(camera).interaction_enabled:
            raise guards.HarnessError(
                "CameraMeasurement AutoPlot interaction gate stayed disabled"
            )
        if camera.state.size != "2x2":
            bench.view.panel_state_changed.emit(camera.panel_id, {"size": "2x2"})
            bench._until(
                lambda: (
                    camera.state.size == "2x2"
                    and camera.host is not None
                    and bench.surface(camera) is not None
                    and bench.surface(camera).presented_front is not None
                ),
                "AutoPlot 2x2 size",
            )

        roi = _draw_mot_roi(bench, camera)
        roi_signal = roi["signal"]

        histogram = bench.add_panel_on(roi_signal, "histogram", size="2x2")
        bench._labels[histogram.panel_id] = labels[1]
        history_edit = bench.edit_setting(histogram, "display", window=40)
        bench._until(
            lambda: _source_index_size(bench, roi_signal) == 40,
            "40-shot ROI history",
            timeout=30.0,
        )

        if panel3 in {"curve", "histogram"}:
            grid = bench.presenter.add_selected_panel("facet_grid")
            bench.view.panel_state_changed.emit(
                grid.panel_id,
                {"cell_kind": panel3, "signal": roi_signal, "size": "2x2"},
            )
            bench._until(
                lambda: (
                    grid.state.cell_kind == panel3
                    and grid.host is not None
                    and bench.surface(grid) is not None
                    and bench.surface(grid).presented_front is not None
                ),
                f"{panel3.capitalize()}-cell FacetGrid",
            )
            bench._name(grid, f"facet_grid:{panel3}")
        else:
            grid = bench.add_panel_on(roi_signal, "facet_grid", size="2x2")
        bench._labels[grid.panel_id] = labels[2]
        bench._pump(3.0)
        grid_assignments = {"source index": "facet"}
        if panel3 == "curve":
            grid_assignments.update({"spatial-x": "x", "spatial-y": "reduced"})
        grid_roles = _assign_roles(bench, grid, grid_assignments)
        fit_activation = bench.edit_setting(
            grid,
            "fit",
            model=fit_model,
        )
        bench._until(
            lambda: (
                grid.host is not None
                and grid.host._session.last_fit is not None
                and len(tuple(getattr(grid.host._session.last_fit, "results", ()))) == 40
            ),
            f"40-cell {fit_model} fit",
            timeout=60.0,
        )

        curve = bench.add_panel_on(roi_signal, "curve", size="2x2")
        bench._labels[curve.panel_id] = labels[3]
        curve_roles = _assign_roles(
            bench,
            curve,
            {"source index": "x", "spatial-x": "reduced", "spatial-y": "group"},
        )

        panels = (camera, histogram, grid, curve)
        guards.require_panels(bench.presenter, 4)
        guards.require_distinct_labels(bench.label(panel) for panel in panels)
        bench._pump(4.0)
        payload.update(
            roi=roi,
            history_edit=history_edit,
            grid_roles=grid_roles,
            curve_roles=curve_roles,
            fit_activation=fit_activation,
            density={bench.label(panel): bench.density(panel) for panel in panels},
            actual_panel_state={
                bench.label(panel): {
                    "kind": panel.state.kind,
                    "cell_kind": panel.state.cell_kind,
                    "signal": panel.state.signal,
                    "semantic": dict(panel.state.semantic),
                    "display": dict(panel.state.display),
                    "fit": dict(panel.state.fit),
                }
                for panel in panels
            },
        )

        timeline = CausalTimeline(bench)
        for panel, label in zip(panels, labels):
            timeline.attach(panel, label)

        baseline_begin, baseline_finish = _window_revision_rate(
            bench, bench.signal, timeline
        )
        baseline = bench.live_all(
            panels,
            baseline_seconds,
            window_start=baseline_begin,
        )
        baseline["source_rate"] = baseline_finish(baseline["window_s"])
        baseline["causal_timeline"] = timeline.summary(labels)
        payload["baseline_uninstrumented"] = baseline

        instrumented = {}
        for panel in panels:
            bench.instrument(panel, module_seams=False)
            instrumented[bench.label(panel)] = bench.instrument_pipeline(panel)
        # ``probe.watch`` replaces the instance method wrappers used by the
        # baseline timeline.  Reattach outside those timed wrappers so the
        # instrumented window retains both measurements without bypassing
        # either one.
        timeline = CausalTimeline(bench)
        for panel, label in zip(panels, labels):
            timeline.attach(panel, label)
        main_begin, main_finish = _window_revision_rate(bench, bench.signal, timeline)
        measured = bench.live_all(
            panels,
            seconds,
            window_start=main_begin,
        )
        measured["source_rate"] = main_finish(measured["window_s"])
        measured["causal_timeline"] = timeline.summary(labels)
        frames = {row["panel"]: row["frames"] for row in measured["panels"]}
        measured["render_cost"] = render_cost(measured["seams"], frames)
        measured["stage_summary"] = _stage_summary(
            measured["seams"], labels, frames
        )
        payload["instrumented_bindings"] = instrumented
        payload["measured"] = measured
        payload["problems"] = [
            {"severity": severity, "message": message}
            for severity, message in bench.problems()
        ]
    payload["threads_left_running"] = list(bench.surviving_threads())
    return payload


def _print(payload: dict) -> None:
    baseline = payload["baseline_uninstrumented"]
    measured = payload["measured"]
    print("MOT ROI four-panel TaskConsole chain")
    print(f"ROI physical shape: {tuple(payload['roi']['shape'])}")
    print(
        "baseline %.1fs, instrumented %.1fs, source %.2f / %.2f revisions/s"
        % (
            baseline["window_s"],
            measured["window_s"],
            baseline["source_rate"]["per_second"],
            measured["source_rate"]["per_second"],
        )
    )
    for title, block in (("baseline", baseline), ("instrumented", measured)):
        print(f"\n{title}:")
        for row in block["panels"]:
            gap = row["frame_gap"]
            print(
                "  %-23s %5.2f fps  gap p50/p90/max %s/%s/%s ms  stalls %d"
                % (
                    row["panel"],
                    row["frames_per_second"],
                    gap.get("median_ms"),
                    gap.get("p90_ms"),
                    gap.get("max_ms"),
                    row["stalls_over_two_beats"],
                )
            )
        print("  four-panel critical path:", block["causal_timeline"]["four_panel_critical_path"])
    print("\ninclusive stage wall ms/call:")
    for panel in measured["stage_summary"]:
        print(" ", panel["panel"])
        for stage, row in panel["stages"].items():
            print(
                "    %-22s %8.2f wall  %8.2f cpu  (%d calls)"
                % (stage, row["wall_ms_per_call"], row["cpu_ms_per_call"], row["calls"])
            )
    if payload["problems"]:
        print("\nconsole problems:")
        for problem in payload["problems"]:
            print("  %(severity)s: %(message)s" % problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--baseline-seconds", type=float, default=6.0)
    parser.add_argument(
        "--panel3", choices=tuple(_PANEL3_FITS), default="curve"
    )
    parser.add_argument(
        "--fit-model",
        default="",
        help="the fit model on panel3's forty cells; default is the cell "
             "kind's own default model (see _PANEL3_FITS)",
    )
    arguments = parser.parse_args()
    fit_model = str(arguments.fit_model).strip() or None
    payload = run(
        seconds=max(3.0, float(arguments.seconds)),
        baseline_seconds=max(3.0, float(arguments.baseline_seconds)),
        panel3=str(arguments.panel3),
        fit_model=fit_model,
    )
    _print(payload)
    label = f"console-mot-roi-four-panel-{arguments.panel3}"
    if fit_model is not None and fit_model != _PANEL3_FITS[arguments.panel3]:
        label += f"-{fit_model}"
    path = write_result(payload, label)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
