# Data boundary

`zlc_data` is the sole owner of scientific data.  `zlc_plot` consumes its
immutable role-axis objects and owns only presentation state: display-unit
conversion, selectors, fit projections, fixed-size raster surfaces and the
active+latest exact-pair live handoff. One solve may be active and only the
latest complete successor is retained; Runtime, not Plot, owns skipped-index
validity and history.

## Role-axis snapshots

Construct a dataset with `AxisSpec`/`PointColumn`, `PointTable`, an optional
`GridTopology`, and a `ValueSchema`:

```python
import numpy as np
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, PointColumn, PointTable,
    REPEAT, SCAN_POINT, COMPONENT, ValidityContract, ValueSchema,
    owned_snapshot_from_arrays,
)

repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
point = PointColumn(
    AxisId("scan"), "scan", SCAN_POINT, PointColumn.NUMERIC,
    (-1.0, 0.0, 1.0), "V",
)
value_axis = AxisSpec(AxisId("value"), "value", COMPONENT, 1, (0,))
cell = ValueSchema(
    (value_axis,), ValidityContract.value(), np.dtype("float64"), "V",
)
schema = DatasetSchema(repeat, PointTable(3, (point,)), None, cell)
snapshot = owned_snapshot_from_arrays(
    schema=schema, values=np.zeros((2, 3, 1)), revision=0,
)
```

The physical geometry is `schema.physical_shape == (R, P, *data_dim)`; the
scalar carrier is an explicit trailing size-one data axis.  `OwnedSnapshot`
stores immutable arrays, a schema fingerprint, block id, stream generation and
monotonic revision.  Validity is a typed data-layer contract and can be
materialized as a dense physical mask with `snapshot.expanded_validity()`.

`GridTopology` is producer-authored.  Its dimension ids and coordinate domains
may be absent from `PointTable`; `zlc_plot` never infers a Cartesian grid from
repeated column values.  Use `AxisRef.point_dimension("b_x")` only when the
producer declared that topology dimension.

## Units are presentation-owned

`zlc_data` keeps unit annotations as strings.  The plotting package resolves
those strings through its presentation registry and converts canonical values
only for display, selectors and fit overlays; stored arrays are never mutated.
The built-in registry accepts both `"1"` and `"arb"` for dimensionless data:

```python
from zlc_plot import DEFAULT_UNITS, resolve_unit
display = resolve_unit("mV", DEFAULT_UNITS)
assert resolve_unit("arb", DEFAULT_UNITS).compatible_with(resolve_unit("1"))
```

Applications may pass a `UnitRegistry` to `PlotSession` when a producer uses a
custom unit symbol.  Labels and selected display units are plot/session state,
not data-schema state.

## Runtime handoff

Acquisition history, cadence and partial/final lifecycle belong to Runtime.
Plot receives immutable revisions through its existing host/session update
transaction. `PulseTimelineData` remains an immutable presentation payload;
`ImageFrame` carries an image and its point annotations as one plot input.

## Persistence and application ownership

Scientific NPZ persistence belongs to `zlc_data`:

```python
from zlc_data import load_npz, save_npz
save_npz("run.npz", snapshot)
restored = load_npz("run.npz")
```

Project files, device calls, Logic routes, causal shot joins and archive
manifests remain application responsibilities.  `zlc_plot` only saves a
rendered raster/front through its public `PlotSession`/`RasterFront` APIs.
