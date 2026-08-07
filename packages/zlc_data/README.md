# zlc-data

`zlc-data` is the independent distribution of the **role-axis** data model used
by Zou Lab Control. Its source of truth is the role-axis package in the
migration tree:

`..\Zou_lab_control_v1_claude\Zou_lab_control_v1\zlc_data`

This repository is the sole owner of that package from its first commit. The
runtime dependency is NumPy only; development adds pytest only.

## Important installation warning

The independent `zlc_plot` repository also distributes a top-level package
named `zlc_data`, but that is a different, name-axis implementation. Never
install both distributions in editable mode in the same environment. The
`zlc_plot` migration and pinning work is outside this repository.

## Install and test

```text
python -m pip install -e ".[dev]"
pytest -q
```

The package has no compatibility shim for the other `zlc_data` owner. Import
identity is guarded by `tests/test_package_guards.py`, which checks the
distribution version and resolves the imported package to this checkout's
`src/zlc_data` directory.

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
| `validation.py` | The seven data-layer validators: `canonical_text`, `exact_mapping`, `finite_real`, `integer`, `nonnegative_integer`, `positive_integer`, and `sha256_text`. |
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

Schema fingerprints in this package are defined by `_tree.digest` (JSON plus
SHA-256), not by the migration tree's `canonical_digest`; the values differ and
schema fingerprints produced by the old tree are not cross-comparable.

## Executable usage

[`notebooks/usage.ipynb`](notebooks/usage.ipynb) is a tutorial organized by
capability: role-axis schema/topology, direct immutable snapshot construction,
named validity, transforms, Value projection, tree codecs, NPZ persistence, and
scalar validation. Each code cell prints a result; rejection cases belong to
the tests. In an environment with notebook tooling installed, run:

```text
python -m nbconvert --to notebook --execute --inplace notebooks/usage.ipynb
```
