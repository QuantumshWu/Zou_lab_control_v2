"""Stable role-axis contracts for named multidimensional data."""

from .axis import (
    COMPONENT,
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisRoleId,
    AxisSpec,
    CoordinateScalar,
    CoordinateSelector,
    CoordinateFrameId,
    LATEST_COORDINATE,
    canonical_coordinate_scalar,
    point_ordinal_axis,
)
from .schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    point_domain_admits,
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
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
    ValidityMode,
)
from .value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    OwnedSnapshot,
    StreamGenerationId,
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
    "finite_real",
    "integer",
    "nonnegative_integer",
    "positive_integer",
    "AxisId",
    "AxisRoleId",
    "AxisSpec",
    "BlockId",
    "COMPONENT",
    "CellValidity",
    "CoordinateFrameId",
    "CoordinateScalar",
    "CoordinateSelector",
    "LATEST_COORDINATE",
    "canonical_coordinate_scalar",
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
    "point_domain_admits",
    "PointTable",
    "PRIMARY_INDEX",
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
    "ValueSchema",
    "NPZFormatError",
    "compact_dataset_validity",
    "expand_dataset_validity",
    "expand_snapshot_validity",
    "load_npz",
    "materialize_derived_dataset",
    "owned_snapshot_from_arrays",
    "save_npz",
]
