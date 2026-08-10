"""Static Curve + selector + unit switch + fit example.

Run from the repository root::

    python examples/static_and_fit.py --headless --output static_fit.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

# This checkout must win over any installed zlc_* distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import zou_lab_control_v2  # noqa: F401

import numpy as np

try:
    from examples._role_data import Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology
except ModuleNotFoundError:
    from _role_data import Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology
from zlc_plot import (
    AxisRef,
    CurvePlot,
    NumericRange,
    PlotLabels,
    PlotSession,
    SelectorKind,
)


DEMO_GENERATION = "zlc-plot-example-generation"


def make_demo_snapshot(revision: int = 1, *, phase: float = 0.0) -> DatasetSnapshot:
    """Return a small immutable ``(R, P, site)`` demo snapshot.

    ``P`` is the flattened Cartesian product of the producer-declared
    ``field`` and ``time`` scan dimensions.  The separate PointTopology makes
    that structure explicit; the plotting layer never infers it from unique
    PointTable values.
    """

    time_axis = Axis.create(
        "time",
        values=np.linspace(-2.0e-3, 2.0e-3, 31),
        label="Time",
        canonical_unit="s",
        display_unit="ms",
    )
    field_axis = Axis.create(
        "field",
        values=np.linspace(-2.0e-3, 2.0e-3, 3),
        label="Field",
        canonical_unit="V",
        display_unit="mV",
    )
    field_rows = np.repeat(field_axis.coordinates, time_axis.size)
    time_rows = np.tile(time_axis.coordinates, field_axis.size)
    points = PointTable.from_columns(
        {"field": field_rows, "time": time_rows},
        units={"field": "V", "time": "s"},
        display_units={"field": "mV", "time": "ms"},
        labels={"field": "Field", "time": "Time"},
    )
    topology = PointTopology.from_cartesian(
        (field_axis, time_axis),
        point_table=points,
    )
    repeat = Axis.create("repeat", values=np.arange(64), label="Repeat")
    site = Axis.create("site", values=np.arange(3), label="Site")
    schema = DatasetSchema.create(
        repeat,
        points,
        data_axes=(site,),
        point_topology=topology,
        value_label="Signal",
        canonical_unit="V",
        display_unit="mV",
        dtype=np.float64,
        generation=DEMO_GENERATION,
        metadata={"example": "static-and-live"},
    )

    center = 0.25e-3 * np.sin(float(phase))
    gaussian = 0.4e-3 + 2.0e-3 * np.exp(
        -0.5 * ((time_rows - center) / 0.62e-3) ** 2
    )
    field_offset = 0.04 * field_rows
    values = np.empty(schema.shape, dtype=np.float64)
    for repeat_index in range(schema.R):
        for site_index in range(site.size):
            values[repeat_index, :, site_index] = (
                gaussian
                + field_offset
                + 0.08e-3
                * np.sin(2.0 * np.pi * repeat_index / repeat.size + float(phase))
                + site_index * 0.015e-3
            )
    return DatasetSnapshot(schema, values, revision=revision)


def run_static_demo(output: str | Path | None = None) -> dict[str, Any]:
    """Render and fit one snapshot, optionally saving a PNG."""

    time_ref = AxisRef.point_dimension("time")
    spec = CurvePlot(
        time_ref,
        group=AxisRef.data("site"),
        labels=PlotLabels(
            title="Static Gaussian fit",
            x="Time",
            y="Signal",
        ),
    )
    with PlotSession(make_demo_snapshot(), spec, size="2x2") as session:
        # The initial display unit is ms. Selector state is stored canonically.
        session.set_area_selector(
            NumericRange(-1.4, 1.4),
            NumericRange(0.6, 2.3),
            display=True,
        )
        selection = session.selector_data(SelectorKind.AREA)
        session.set_axis_unit(time_ref, "s")
        fit = session.fit("gaussian_offset")
        if output is not None:
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            session.save(target)
        rgba_shape = tuple(int(value) for value in session.rgba().shape)
        return {
            "fit_success": fit.success,
            "fit_scope": session.fit_selection("gaussian_offset").scope.value,
            "center_s": float(fit.parameters["center"]),
            "selected_samples": int(selection.canonical_values.size),
            "rgba_shape": rgba_shape,
            "preset": session.surface_plan.preset,
        }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="force the Agg backend")
    parser.add_argument("--output", type=Path, default=Path("static_fit.png"))
    args = parser.parse_args()
    if args.headless:
        import matplotlib

        matplotlib.use("Agg", force=True)
    summary = run_static_demo(args.output)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
