"""Isolated PlotSession costs driven by one SimulationWorld MOT source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import probe
from .common import ROOT, stats, write_result
from .run_console import renderer_seams


def _simulation_feeds(*, updates: int) -> dict:
    """One MOT world feeding the full frame and one rolling 40-shot ROI."""

    import zou_lab_control  # noqa: F401 - current checkout owns every package
    from zlc_atom.devices.camera.contract import CameraFrameRecord
    from zlc_atom.devices.simulation import (
        SimulationWorld,
        VirtualCamera,
        VirtualCameraConfig,
    )
    from zlc_atom.nodes.camera_measurement.measurement import frames_snapshot
    from zlc_data import BlockId, StreamGenerationId, owned_snapshot_from_arrays
    from zlc_runtime.plane import _indexed_schema

    height, width = 1200, 1920
    roi_height, roi_width = 40, 500
    roi_y = (height - roi_height) // 2
    roi_x = (width - roi_width) // 2
    total = 40 + updates + 2
    world = SimulationWorld()
    frame_source = lambda ordinal, exposure: world.render_mot_frame(
        ordinal,
        exposure_seconds=exposure,
        frame_shape_yx=(height, width),
    )
    full_camera = VirtualCamera(
        VirtualCameraConfig(
            frame_shape_yx=(height, width),
            exposure_seconds=0.1,
            frame_dtype="|u1",
        ),
        frame_source=frame_source,
        free_running=True,
    )
    roi_camera = VirtualCamera(
        full_camera.config,
        frame_source=frame_source,
        free_running=True,
    )
    roi_camera.set_roi((roi_x, roi_y, roi_width, roi_height))

    full_events = []
    roi_events = []
    for ordinal in range(total):
        frame = world.render_mot_frame(
            ordinal,
            exposure_seconds=0.1,
            frame_shape_yx=(height, width),
        )
        roi_record = CameraFrameRecord(
            frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width],
            ordinal,
            ordinal + 1,
        )
        if ordinal >= 39:
            full_record = CameraFrameRecord(frame, ordinal, ordinal + 1)
            full_events.append(
                frames_snapshot(
                    ((full_record,),),
                    producer="camera_measurement",
                    generation="sim-world-mot",
                    revision=ordinal + 1,
                    working_point=full_camera.working_point(),
                    value_unit="count",
                )
            )
        roi_events.append(
            frames_snapshot(
                ((roi_record,),),
                producer="panel-1",
                generation="sim-world-roi",
                revision=ordinal + 1,
                working_point=roi_camera.working_point(),
                value_unit="count",
            )
        )

    relative = tuple(range(-39, 1))
    indexed_schema = _indexed_schema(roi_events[0].block.schema, relative)
    roi_history = []
    camera = []
    for revision in range(1, updates + 4):
        start = revision - 1
        window = roi_events[start : start + 40]
        values = np.concatenate(
            [snapshot.block.values for snapshot in window], axis=1
        )
        roi_history.append(
            owned_snapshot_from_arrays(
                indexed_schema,
                values,
                revision,
                block_id=BlockId(f"sim-world-roi-history-{revision}"),
                stream_generation=StreamGenerationId("sim-world-roi-history"),
            )
        )
        camera.append(full_events[start])
    return {
        "camera": tuple(camera),
        "roi_history": tuple(roi_history),
        "roi_xywh": (roi_x, roi_y, roi_width, roi_height),
    }


def _specs(feeds: dict) -> dict:
    from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
    from zlc_plot import (
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
    )

    camera_schema = feeds["camera"][0].block.schema
    roi_schema = feeds["roi_history"][0].block.schema
    frame_ref = AxisRef.point(str(camera_schema.point_table.columns[0].coordinate_id))
    camera_y, camera_x = (
        AxisRef.data(str(axis.axis_id)) for axis in camera_schema.cell_schema.data_axes
    )
    roi_y, roi_x = (
        AxisRef.data(str(axis.axis_id)) for axis in roi_schema.cell_schema.data_axes
    )
    source = AxisRef.point(str(PRIMARY_INDEX_AXIS_ID))
    camera_image = ImagePlot(camera_x, camera_y)
    roi_image = ImagePlot(roi_x, roi_y)
    return {
        "panel1 camera-grid": FacetGridPlot(frame_ref, camera_image),
        "panel1 standalone-image": camera_image,
        "panel2 histogram-40": HistogramPlot(),
        "panel3 facet-fit-40": FacetGridPlot(source, roi_image),
        "panel3b facet-curve-40": FacetGridPlot(source, CurvePlot(roi_x)),
        "panel3c facet-histogram-40": FacetGridPlot(source, HistogramPlot()),
        "panel4 curve-40": CurvePlot(source, group=roi_y),
    }


def _advance(session, snapshot) -> dict[str, float]:
    started = time.perf_counter()
    prepared = session.prepare_live_frame(snapshot).result()
    projected = time.perf_counter()
    fit_future = session.solve_live_frame(prepared)
    solved = None if fit_future is None else fit_future.result()
    fitted = time.perf_counter()
    finalization = session.commit_live_frame(prepared, solved)
    if finalization is None:
        raise RuntimeError("isolated live frame was not committed")
    rendered = time.perf_counter()
    session.publish_live_frame(finalization)
    finished = time.perf_counter()
    return {
        "projection": projected - started,
        "fit": fitted - projected,
        "render": rendered - fitted,
        "publish": finished - rendered,
        "total": finished - started,
    }


def _case(
    label: str,
    feed,
    spec,
    *,
    updates: int,
    parameters: dict | None = None,
    fit: dict | None = None,
    facet_focus: int | None = None,
    selector_facet: int | None = None,
    roi_xywh: tuple[int, int, int, int] | None = None,
) -> dict:
    from zlc_plot import PlotSession
    from zlc_plot.selectors import (
        NumericRange,
        RectangleRange,
        SelectorKind,
        SelectorState,
    )

    session = PlotSession(
        feed[0],
        spec,
        size="2x2",
        parameters=parameters,
        device_pixel_ratio=3.0,
    )
    try:
        selectors = ()
        if roi_xywh is not None:
            roi_x, roi_y, roi_width, roi_height = roi_xywh
            selectors = (
                SelectorState(
                    SelectorKind.AREA,
                    RectangleRange(
                        NumericRange(roi_x, roi_x + roi_width - 1),
                        NumericRange(roi_y, roi_y + roi_height - 1),
                    ),
                    facet_index=selector_facet,
                ),
            )
        session.configure(
            selectors=selectors,
            facet_focus=facet_focus,
            fit={} if fit is None else fit,
            fit_live=True,
        )
        renderer = session._renderer
        probe.watch(
            renderer,
            *renderer_seams(type(renderer)),
            prefix=f"MatplotlibRenderer[{label}]",
        )
        _advance(session, feed[1])
        _advance(session, feed[2])
        probe.reset()
        samples = [_advance(session, feed[index + 3]) for index in range(updates)]
        rows = probe.rows(sum(sample["total"] for sample in samples))
        render_row = next(
            row
            for row in rows
            if row["seam"] == f"MatplotlibRenderer[{label}].present"
        )
        return {
            "label": label,
            "source_shape": list(map(int, feed[0].block.values.shape)),
            "figure_px": [
                int(round(float(renderer.figure.bbox.width))),
                int(round(float(renderer.figure.bbox.height))),
            ],
            "dpr": float(renderer.plan.device_pixel_ratio),
            "stages": {
                stage: stats([sample[stage] for sample in samples])
                for stage in ("projection", "fit", "render", "publish", "total")
            },
            "renderer_present": render_row,
            "seams": rows,
        }
    finally:
        session.close()


def run(*, updates: int) -> dict:
    feeds = _simulation_feeds(updates=updates)
    specs = _specs(feeds)
    camera = feeds["camera"]
    roi = feeds["roi_history"]
    cases = [
        _case(
            "panel1 camera-grid",
            camera,
            specs["panel1 camera-grid"],
            updates=updates,
            facet_focus=0,
            selector_facet=0,
            roi_xywh=feeds["roi_xywh"],
        ),
        _case(
            "panel1 standalone-image",
            camera,
            specs["panel1 standalone-image"],
            updates=updates,
            roi_xywh=feeds["roi_xywh"],
        ),
        _case(
            "panel2 histogram-40",
            roi,
            specs["panel2 histogram-40"],
            updates=updates,
            parameters={"window": 40},
        ),
        _case(
            "panel3 facet-fit-40",
            roi,
            specs["panel3 facet-fit-40"],
            updates=updates,
            fit={"model": "anisotropic_gaussian_center", "fit_all_facets": True},
        ),
        _case(
            "panel3b facet-curve-40",
            roi,
            specs["panel3b facet-curve-40"],
            updates=updates,
            parameters={"uncertainty": False},
        ),
        _case(
            "panel3c facet-histogram-40",
            roi,
            specs["panel3c facet-histogram-40"],
            updates=updates,
        ),
        _case(
            "panel4 curve-40",
            roi,
            specs["panel4 curve-40"],
            updates=updates,
            parameters={"uncertainty": True},
        ),
    ]
    console_path = ROOT / "bench" / "results" / "console-mot-roi-four-panel.json"
    console = json.loads(console_path.read_text(encoding="utf-8"))
    console_stages = {
        row["panel"]: row for row in console["measured"]["stage_summary"]
    }
    comparison = {}
    for case in cases:
        console_row = console_stages.get(case["label"])
        if console_row is None:
            continue
        isolated_render = case["stages"]["render"]["median_ms"]
        console_render = console_row["stages"]["renderer_present"][
            "wall_ms_per_call"
        ]
        comparison[case["label"]] = {
            "isolated_render_ms": isolated_render,
            "console_render_ms": console_render,
            "console_over_isolated": round(console_render / isolated_render, 2),
        }
    return {
        "scenario": "mot-simulation-world-isolated",
        "source": {
            "world": "SimulationWorld.render_mot_frame",
            "exposure_seconds": 0.1,
            "roi_xywh": list(feeds["roi_xywh"]),
            "history": 40,
        },
        "cases": cases,
        "comparison": comparison,
    }


def _print(result: dict) -> None:
    print("isolated PlotSession from one SimulationWorld MOT source")
    for case in result["cases"]:
        stages = case["stages"]
        print(
            "%-25s proj %7.2f  fit %7.2f  render %7.2f  total %7.2f ms"
            % tuple(
                [case["label"]]
                + [
                    stages[name].get("median_ms", 0.0)
                    for name in ("projection", "fit", "render", "total")
                ]
            )
        )
    print("console / isolated render:")
    for label, row in result["comparison"].items():
        print(
            "  %-23s %7.2f / %7.2f = %.2fx"
            % (
                label,
                row["console_render_ms"],
                row["isolated_render_ms"],
                row["console_over_isolated"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=8)
    arguments = parser.parse_args()
    result = run(updates=max(4, int(arguments.updates)))
    _print(result)
    path = write_result(result, "mot-roi-isolated-same-source")
    print("wrote", path)


if __name__ == "__main__":
    main()
