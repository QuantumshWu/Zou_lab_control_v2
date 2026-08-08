# zlc-data

`zlc-data` is the **role-axis** data-model layer inside the
`Zou_lab_control_v2` monorepo. This package directory is its only current
source of truth; old standalone repositories and v1 trees are historical
references, not alternate install or edit locations. Its runtime dependency is
NumPy only.

## Test the monorepo package

Run from the repository root and bootstrap that checkout before importing any
`zlc_*` name:

```text
python -c "import zou_lab_control_v2; import zlc_data; print(zlc_data.__file__)"
python -m pytest -q packages/zlc_data/tests
```

The package has no compatibility shim for an old name-axis `zlc_data` copy.
Import-identity tests require this checkout's `packages/zlc_data/src/zlc_data`
directory.

The cross-package ownership boundary for canonical units, display conversion,
and latest-only live ingress is documented in [`docs/contract.md`](docs/contract.md).

## Module map

| Module | Role |
| --- | --- |
| `axis.py` | Role identifiers, axis identities, coordinate frames, and axis specifications. |
| `schema.py` | Point rows, grid topology, value schemas, dataset schemas, and point-row resolution. |
| `validity.py` | Value-, cell-, and named-component validity contracts and immutable masks. |
| `value.py` | `Value`, `DataBlock`, `OwnedSnapshot`, direct array-to-snapshot construction, revision identities, and validity expansion/compaction. |
| `selection.py` | Axis-named index/coordinate selections and their tree representation. |
| `transform.py` | Committed selections, reductions, histograms, schema resolution, and transform application. |
| `codec.py` | Strict schema and revision-reference tree codecs plus schema fingerprints. |
| `transform_codec.py` | Strict tree codecs for selections and committed transforms. |
| `snapshot_projection.py` | Explicit scalar, value, and derived dataset materialization helpers. |
| `output_contract.py` | Stable semantic ids for explicit authoritative dataset projections. |
| `io.py` | Pickle-free NPZ persistence through `save_npz`/`load_npz`. |
| `validation.py` | The seven data-layer validators: `canonical_text`, `exact_mapping`, `finite_real`, `integer`, `nonnegative_integer`, `positive_integer`, and `digest_text`. |
| `numeric.py` | Canonical numeric dtypes and checked reduction arithmetic. |
| `_arrays.py`, `_diagnostic.py`, `_tree.py` | Private array immutability, exact diagnostic formatting, and deterministic primitive-tree support. |

## Public API

The stable convenience surface is available from `zlc_data`:

| Area | Main names |
| --- | --- |
| Axis and schema | `AxisId`, `AxisRoleId`, `AxisSourceRef`, `AxisSpec`, `COMPONENT`, `READOUT_EVENT`, `REPEAT`, `SCAN_POINT`, `SITE`, `SPATIAL_X`, `SPATIAL_Y`, `CoordinateFrameId`, `DatasetSchema`, `GridTopology`, `PointColumn`, `PointTable`, `ValueSchema` |
| Values and snapshots | `BlockId`, `DataBlock`, `DatasetRevision`, `DatasetRevisionRef`, `OwnedSnapshot`, `StreamGenerationId`, `Value`, `ValuePayloadContract` |
| Validity | `VALID`, `INVALID`, `Valid`, `Invalid`, `ValidityContract`, `ValidityMode`, `CellValidity`, `ComponentValidity`, `DatasetComponentValidity`, `compact_dataset_validity`, `expand_dataset_validity`, `expand_snapshot_validity` |
| Snapshot construction | `owned_snapshot_from_arrays`, `OwnedSnapshot.expanded_validity`, `OwnedSnapshot.exactly_equals` |
| Selection and transforms | `Selection`, `DataTransformSpec`, `CommittedTransform`, `ReductionSpec`, `HistogramSpec`, `commit_transform`, `apply_transform`, `resolve_transformed_schema` |
| Persistence | `NPZFormatError`, `save_npz`, `load_npz` |

Some transform and codec functions are intentionally module-scoped; import
them from `zlc_data.transform`, `zlc_data.codec`, or
`zlc_data.transform_codec` when tree-level authority is required. The seven
validation primitives, including `integer`, are module-scoped in
`zlc_data.validation`. Likewise, `dataset_cell_value`,
`expand_component_validity`, and `expand_value_validity` are module-scoped
helpers in `zlc_data.value`. The exact facade allow-list, including the
module-scoped boundary, is recorded in [`docs/contract.md`](docs/contract.md)
and guarded by tests.

Schema fingerprints in this package are existing BLAKE2b-128 content names
defined by `_tree.digest`. They identify a schema used by `zlc_data` codecs and
committed transforms; they are not Calibration or Panel Save provenance, a
security feature, or authority to add fingerprints/hashes to those product
artifacts. Panel archives can rebuild the runtime revision reference from the
schema they already store instead of persisting the derived name again.

## Executable usage

[`notebooks/usage.ipynb`](notebooks/usage.ipynb) is a tutorial organized by
capability: role-axis schema/topology, direct immutable snapshot construction,
named validity, transforms, Value projection, tree codecs, NPZ persistence, and
scalar validation. Each code cell prints a result; rejection cases belong to
the tests. In an environment with notebook tooling installed, run:

```text
python -m nbconvert --to notebook --execute --inplace notebooks/usage.ipynb
```
