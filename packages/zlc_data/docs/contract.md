# zlc_data cross-package contract

This document is the ownership boundary between the role-axis `zlc_data`
package and presentation-layer consumers such as `zlc_plot`. It is normative
for the I1 reconciliation: consumers may adapt their inputs at the boundary,
but they must not create a second data or unit authority.

## 单位与 live 的归属

### Unit ownership

`AxisSpec.unit`, `PointColumn.unit`, and `ValueSchema.value_unit` are optional
strings annotating the **canonical unit** of the stored coordinates or values.
They do not describe a display unit and they do not carry conversion behavior.
All physical computation represented by `zlc_data`—validity, selection, and
persistence—operates on canonical values.

`zlc_data` deliberately does not own a unit registry, unit objects, affine
conversion rules, compatibility checks, inverse-unit lookup, or `arb`/`1`
dimensionless conversion semantics. No display conversion logic is to be
added to this package.

Display-unit choice, canonical/display conversion, labels, and conversion of
user-facing selector or fit ranges belong to the presentation layer. For the
I1 integration, `zlc_plot` is the authority for those operations and must
convert at its presentation boundary while keeping the `zlc_data` snapshot in
canonical units.

### Live ownership

`zlc_data` owns immutable schemas, topology, values, typed validity,
revision references, codecs, NPZ persistence, and the direct
`OwnedSnapshot` construction convenience. It does not own a live channel,
latest-only replacement policy, ingress queue, patch/rolling state, or any
other mutable transport state.

Latest-only live ingress and its replacement/patch semantics belong to the
presentation/runtime transport owner. For I1, that owner is `zlc_plot` (or a
separately specified runtime owner), not `zlc_data`. A live consumer may hand
one completed immutable `OwnedSnapshot` to the data contract; it must not add
live state or a second snapshot schema inside this package.

This section is the cross-repository contract for the `zlc_plot` I1
reconciliation. The package boundary remains intentionally narrow: canonical
unit annotations and immutable snapshot data cross it, while display-unit
conversion and latest-only live ingress stay on the presentation side.

## Public API allow-list

The following JSON array is the exact set of names exported by
`zlc_data.__all__`. Implementations that are useful only as return types,
internal policies, role constants, or low-level helpers remain available from
their owning submodules when needed; they are not re-exported by the package
facade.

```json
[
  "AxisId",
  "AxisRoleId",
  "AxisSourceRef",
  "AxisSpec",
  "BlockId",
  "COMPONENT",
  "CellValidity",
  "ComponentValidity",
  "CoordinateFrameId",
  "DataBlock",
  "DatasetComponentValidity",
  "DatasetRevision",
  "DatasetRevisionRef",
  "DatasetSchema",
  "GridTopology",
  "INVALID",
  "IndexSelection",
  "Invalid",
  "NPZFormatError",
  "OwnedSnapshot",
  "PointColumn",
  "PointTable",
  "point_ordinal_axis",
  "READOUT_EVENT",
  "REPEAT",
  "SCAN_POINT",
  "SITE",
  "SPATIAL_X",
  "SPATIAL_Y",
  "Selection",
  "SelectionChange",
  "StreamGenerationId",
  "VALID",
  "Valid",
  "ValidityContract",
  "ValidityMode",
  "Value",
  "ValuePayloadContract",
  "ValueSchema",
  "__version__",
  "canonical_text",
  "compact_dataset_validity",
  "digest_text",
  "exact_mapping",
  "expand_component_validity",
  "expand_dataset_validity",
  "expand_snapshot_validity",
  "finite_real",
  "integer",
  "is_intrinsically_immutable_array",
  "load_npz",
  "materialize_derived_dataset",
  "materialize_scalar_dataset",
  "materialize_value_dataset",
  "nonnegative_integer",
  "owned_snapshot_from_arrays",
  "positive_integer",
  "resolve_selection_indices",
  "save_npz",
  "snapshot_from_manifest",
  "snapshot_manifest"
]
```

### Export review note

The adversarial API review deliberately retained `AxisRoleId`, `ValidityMode`,
`ValuePayloadContract`, `AxisSourceRef`, and `__version__` at the facade. They
are required by downstream production contracts or by the authoritative
construction path; the notebook demonstrates their use. The withdrawn names
are return carriers, policy internals, unused role constants, or low-level
helpers and remain owned by their original submodules.
