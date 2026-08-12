"""Stable role-axis contracts for named multidimensional data."""

__version__ = "0.1.0"

from .axis import (
    COMPONENT,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisRoleId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
    point_ordinal_axis,
)
from .schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from .selection import (
    # What a selection IS, and how one becomes row indices.  Two packages
    # publish and consume selections across the wire; both reached in here
    # for the vocabulary, and an enum matched by string VALUE rather than by
    # identity is a shared contract nobody declared.
    IndexSelection,
    Selection,
    SelectionChange,
    resolve_selection_indices,
)
from .validation import (
    canonical_text,
    digest_text,
    exact_mapping,
    finite_real,
    integer,
    nonnegative_integer,
    positive_integer,
)
from ._arrays import is_intrinsically_immutable_array
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
    ValidityMode,
)
from .value import (
    expand_component_validity,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    OwnedSnapshot,
    StreamGenerationId,
    Value,
    ValuePayloadContract,
    compact_dataset_validity,
    expand_dataset_validity,
    expand_snapshot_validity,
    owned_snapshot_from_arrays,
)
# How a snapshot is written down and read back.  A figure archive IS a
# snapshot plus its manifest, so the layer that writes one needs both, and
# reaching into the submodule for them left the format's owner unable to see
# who depended on it.
from .io import (
    NPZFormatError,
    load_npz,
    save_npz,
    snapshot_from_manifest,
    snapshot_manifest,
)
from .snapshot_projection import (
    materialize_derived_dataset,
    materialize_scalar_dataset,
    materialize_value_dataset,
)

__all__ = [
    "IndexSelection",
    "is_intrinsically_immutable_array",
    "SelectionChange",
    "resolve_selection_indices",
    "snapshot_from_manifest",
    "snapshot_manifest",
    "canonical_text",
    "digest_text",
    "exact_mapping",
    "expand_component_validity",
    "finite_real",
    "integer",
    "nonnegative_integer",
    "positive_integer",
    "AxisId",
    "AxisRoleId",
    "AxisSourceRef",
    "AxisSpec",
    "BlockId",
    "COMPONENT",
    "CellValidity",
    "ComponentValidity",
    "CoordinateFrameId",
    "point_ordinal_axis",
    "DataBlock",
    "DatasetComponentValidity",
    "DatasetRevision",
    "DatasetRevisionRef",
    "DatasetSchema",
    "GridTopology",
    "INVALID",
    "Invalid",
    "OwnedSnapshot",
    "PointColumn",
    "PointTable",
    "READOUT_EVENT",
    "REPEAT",
    "SCAN_POINT",
    "SITE",
    "SPATIAL_X",
    "SPATIAL_Y",
    "Selection",
    "StreamGenerationId",
    "VALID",
    "Valid",
    "ValidityContract",
    "ValidityMode",
    "Value",
    "ValuePayloadContract",
    "ValueSchema",
    "NPZFormatError",
    "__version__",
    "compact_dataset_validity",
    "expand_dataset_validity",
    "expand_snapshot_validity",
    "load_npz",
    "materialize_derived_dataset",
    "materialize_value_dataset",
    "materialize_scalar_dataset",
    "owned_snapshot_from_arrays",
    "save_npz",
]
