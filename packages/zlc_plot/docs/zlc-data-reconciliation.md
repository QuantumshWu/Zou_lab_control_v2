# zlc_plot / zlc-data reconciliation

Status: migration in progress.  The required API reconciliation is complete
against the W-round `zlc-data` checkout.  The role-axis package now owns the
immutable schema/topology/value/revision/validity contract.  Presentation-only
unit conversion and latest-only transport remain in `zlc_plot`, as specified
by `zlc-data/docs/contract.md`.

## Compared sources

| owner | checkout / version | role |
| --- | --- | --- |
| pre-I1 private data contract | `zlc_plot` at the pre-I1 worktree | name-axis `(R, P, *data_dim)` contract previously consumed by the plot and fit code |
| `zlc_data` | `C:\Users\eadri\Dropbox\WorkCode\Github\zlc_data`, W-round git `85d3c05`, package `0.1.0` | independent role-axis data contract |

The installed `zlc_data` resolves to the second checkout above.  It is not a
renamed copy of the private package: its public objects, identity model,
validity model, and persistence format are different.

## Object mapping

| Current private contract | Role-axis contract | Mapping | Missing or incompatible semantics |
| --- | --- | --- | --- |
| `Axis` | `AxisSpec` or `PointColumn` | A repeat/data `Axis` could become an `AxisSpec`; a point-table `Axis` could become a `PointColumn`. | One old object carries name, label, canonical unit, display unit, aliases, affine conversion, and metadata. `AxisSpec` has name/role/unit/frame/index origin; `PointColumn` has name/role/value kind/unit/frame. Neither has label, display unit, affine conversion, or metadata. |
| `Axis.create(...)` | direct `AxisSpec(...)` / `PointColumn(...)` | Construction must be rewritten by role and by axis owner. | `zlc_data` has no unit registry or `arb`/`1` conversion vocabulary. `unit` is a string annotation, not a conversion object. |
| `PointTable.from_columns(...)` | `PointTable(row_count, columns)` | Build a `PointColumn` for every column and choose a role/value kind. | The old convenience constructor infers numeric arrays and stores labels/units/display units. The new table permits zero columns and does not provide old `axis`, `names`, `column(display=...)` behavior. |
| `PointTopology` | `GridTopology` | `dimensions` become `dimension_ids` plus `coordinate_domains`; `row_to_cell` becomes tuples. | The old contract intentionally permits topology dimensions absent from the point table (for example `b_x,b_y,b_z` declared only in topology). `zlc_data._validate_grid_topology` calls `point_table.column(dimension_id)` for every grid dimension, so that valid V1 representation cannot be represented. |
| `DatasetSchema` | `zlc_data.DatasetSchema` + `ValueSchema` | `repeat_axis` maps to an `AxisSpec(role=REPEAT)`; old `data_axes` map to `ValueSchema.data_axes`; old dtype/value unit map partly to `ValueSchema`. | The old schema owns `value_label`, canonical/display units, metadata, a simple generation string, and `shape/R/P/ndim/validate_values`. The new schema owns a fingerprint, `cell_schema`, and a role/validity contract; it has no value label, display unit, or metadata. |
| `DatasetSnapshot` | `DataBlock` + `OwnedSnapshot` | `values` can become `DataBlock.values`; dense validity can be compacted with `compact_dataset_validity`; revision can become `DatasetRevision`. | New snapshots require `BlockId`, `StreamGenerationId`, schema fingerprint, and typed validity. There is no immutable `(schema, values, revision, validity, metadata)` snapshot constructor. Plot code currently reads `snapshot.validity` as a dense ndarray and `snapshot.generation` as a string. |
| `DatasetSnapshot.validity` | `VALID`/`INVALID`, `CellValidity`, `DatasetComponentValidity` | `expand_dataset_validity` can materialize a dense mask for a compatible new block. | This is an explicit semantic change and must be threaded through `DataView`, selectors, fit, and live transport; it cannot be solved by aliasing a class. |
| `LatestRevisionChannel` | none | Could move the plot transport implementation out of the data package. | `zlc-data` does not own a latest-only live ingress/channel. Existing tests and `zlc_plot.live` use the private channel API directly. This is runtime plumbing, not a data-schema alias. |
| `Unit`, `UnitRegistry`, `resolve_unit` | `AxisSpec.unit: str | None`, `ValueSchema.value_unit: str | None` | Only the textual annotation survives. | `zlc_plot` currently performs canonical/display conversion, compatibility checks, inverse-unit lookup, and selector/fit range conversion. Reimplementing those in `zlc_plot` would create the forbidden second unit system. |
| `save_npz/load_npz(snapshot)` | `save_npz/load_npz(OwnedSnapshot)` | Both are NPZ persistence entry points. | Formats are not compatible. The new format persists role-axis fingerprints/references and typed validity; the old format persists the name-axis schema, units, metadata, and dense snapshot. No migration adapter is supplied by either package. |
| `SchemaError`, `RevisionError`, `UnitError`, `SerializationError` | mostly `ValueError`/`TypeError` plus `NPZFormatError` | Error translation is possible at a boundary. | Error identity and messages are part of current tests and public behavior; translating them during every projection would be a new compatibility layer. |

## Consumer requirements found in zlc_plot

The following are not hypothetical uses; they are read directly by the
current projection/session/fit paths and examples:

1. `DataView` resolves `repeat`, point rows, point-table coordinates,
   topology dimensions, and arbitrary data axes into coordinate arrays.  It
   needs topology dimensions even when those dimensions are not redundant
   point-table columns.
2. `AxisTransform` and `PlotSession` convert canonical values to a selected
   display unit and back.  This is used by axis labels, selector ranges,
   histogram edges, fit inputs, and `set_axis_unit`/`set_value_unit`.
3. `DatasetSnapshot` is the ingress payload for `RasterPlotHost`,
   `LivePlotController`, `ImageFrame`, selectors, fit projection, notebook
   frames, and all public examples.  The payload is required to retain one
   fixed schema identity and an integer revision while keeping arrays
   immutable.
4. `LatestRevisionChannel` provides the bounded/latest-only transport used by
   `LivePlotController` and its channel tests.
5. NPZ round-tripping is currently an application-facing snapshot operation,
   not a transform/materialization operation.

## I1 decision gate — resolved

The W round closed the three data-boundary decisions that blocked migration:

* `GridTopology` dimensions may be declared without same-named point-table
  columns, so producer-authored `b_x/b_y/b_z` topology is representable;
* `owned_snapshot_from_arrays` provides direct immutable `OwnedSnapshot`
  construction, while `OwnedSnapshot.expanded_validity()` provides the dense
  physical mask needed by the projection layer;
* canonical/display unit conversion and latest-only ingress are explicitly
  presentation/runtime responsibilities, not duplicated in `zlc-data`.

The implementation phase is therefore unblocked.  The migration must use the
real `OwnedSnapshot` object and `zlc_data` schema classes, move only
presentation units/live transport into `zlc_plot`, and delete the private
name-axis package rather than adding an adapter or shadow data model.

### Runtime probe

The following read-only probe was run against the installed editable
`zlc-data 0.1.0` checkout:

```text
zlc-data package: 0.1.0
topology-without-point-column: accepted
direct immutable snapshot: OwnedSnapshot
expanded validity shape: (R, P, *data_dim)
```

The probe now exercises the W-round behavior rather than the former negative
case.  Unit conversion and latest-only transport intentionally remain absent
from `zlc-data`; they are moved to presentation-owned modules in this package.

## Migration execution order

The safe migration order is:

1. Port `zlc_plot` imports and `DataView`/fit validity access to the public
   role-axis objects, with one presentation conversion boundary.
2. Move the unit registry and latest-only channel into presentation-owned
   modules, then delete the pre-I1 private data directory and every legacy import.
3. Rewrite the NPZ and namespace guards against real `zlc_data` objects, run
   the full offscreen suite, execute `notebooks/usage.ipynb`, and compare
   every golden pixel before closing I1.
