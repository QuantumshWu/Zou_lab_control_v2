# Data boundary

`zlc_data` is the sole owner of scientific data.  `zlc_plot` consumes its
immutable role-axis objects and owns only presentation state: display-unit
conversion, selectors, fit projections, fixed-size raster surfaces and the
active+latest exact-pair live handoff. One solve may be active and only the
latest complete successor is retained; Runtime, not Plot, owns skipped-index
validity and history.

## Role-axis snapshots

Construct a dataset from three `DomainSpec` values and one `ValueSchema`.
Repeat and Point are flat physical carriers whose axis-major codes map every
row to each logical coordinate; Cell-data is a dense domain whose axes map
directly to tensor dimensions:

```python
import numpy as np
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, DomainSpec,
    REPEAT, SCAN_POINT, SCALAR_DOMAIN, ValidityContract, ValueSchema,
    owned_snapshot_from_arrays,
)

repeat_axis = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
repeat = DomainSpec((2,), (repeat_axis,), ((0, 1),))
scan = AxisSpec(AxisId("scan"), "scan", SCAN_POINT, 3, (-1.0, 0.0, 1.0), "V")
point = DomainSpec((3,), (scan,), ((0, 1, 2),))
value = ValueSchema(ValidityContract.value(), np.dtype("float64"), "V")
schema = DatasetSchema(repeat, point, SCALAR_DOMAIN, value)
snapshot = owned_snapshot_from_arrays(
    schema=schema, values=np.zeros((2, 3, 1)), revision=0,
)
```

The physical geometry is `schema.physical_shape == (R, P, *cell_shape)`.
A scalar value declares `SCALAR_DOMAIN` as its Cell domain: its trailing
size-one carrier axis is representation, consumed automatically, and never
appears in Plot fate/title vocabulary.  A size-one `COMPONENT` axis is
different -- it is an information axis with a fate of its own
(`fate:cell_data:<id>`), offered like any other. `OwnedSnapshot`
stores immutable arrays, a schema fingerprint, block id, stream generation and
monotonic revision.  Validity is a typed data-layer contract and can be
materialized as a dense physical mask with `snapshot.expanded_validity()`.

Each logical axis is declared exactly once. `DomainSpec.codes(axis_id)` returns
only the one-dimensional code vector along that axis's physical carrier:
length `R` or `P` for mapped Repeat/Point domains, and an implicit identity of
length `Di` for a dense Cell axis. Plot projection broadcasts these small
vectors by stride and never materializes a full coordinate plane merely to
recover geometry. Axis references are correspondingly just
`AxisRef.repeat(id)`, `AxisRef.point(id)`, or `AxisRef.cell_data(id)`.

## Units are defined by Data and chosen for display by Plot

`zlc_data` keeps unit annotations as strings and owns the registry that
resolves them: `zlc_data.units` defines the units, their prefixes and the
conversions between them.  The plotting package chooses a display unit per
axis through that registry and converts canonical values only for display,
selectors and fit overlays; stored arrays are never mutated.  The built-in
registry accepts both `"1"` and `"arb"` for dimensionless data:

```python
from zlc_data.units import DEFAULT_UNITS, resolve_unit
display = resolve_unit("mV", DEFAULT_UNITS)
assert resolve_unit("arb", DEFAULT_UNITS).compatible_with(resolve_unit("1"))
```

Applications may pass a `zlc_data.units.UnitRegistry` to `PlotSession` when a
producer uses a custom unit symbol.  Labels and selected display units are
plot/session state, not data-schema state.

## Runtime handoff

Acquisition history, cadence and partial/final lifecycle belong to Runtime.
Plot receives immutable revisions through its existing host/session update
transaction. `PulseTimelineData` remains an immutable presentation payload;
`ImageFrame` carries an image and its point annotations as one plot input.

## Persistence and application ownership

`zlc_data` owns scientific NPZ encoding and decoding; `zlc_durable` owns
atomic path publication:

```python
from zlc_data import load_npz, save_npz
from zlc_durable import atomic_write_file

atomic_write_file("run.npz", lambda stream: save_npz(stream, snapshot))
restored = load_npz("run.npz")
```

Project files, device calls, Logic routes, causal shot joins and archive
manifests remain application responsibilities.  `zlc_plot` owns the exact Plot
recipe of a figure: its public Figure API (`save_figure_artifact`,
`encode_plot_recipe` / `decode_plot_recipe`, `read_figure_plot`,
`build_figure_host` / `open_figure_host`) writes the typed archive first and
the preview second, through `zlc_data`'s Figure NPZ grammar and
`zlc_durable`'s atomic publication, and a rendered front is saved through the
public `PlotSession` / `RasterFront` APIs.  An application adds its own
workflow and provenance records beside that archive; it never rewrites the
recipe.
