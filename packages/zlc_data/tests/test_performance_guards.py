


def test_a_schema_catalogues_its_axes_once() -> None:
    """The catalog is a fact about the schema; it is built once.

    Every AxisSpec the catalog builds re-runs canonical_coordinate_scalar
    over a tuple PointColumn has already canonicalised.  Measured on a
    200x200 scan that is 26 ms of the 41 ms selection_indices costs, and
    an operator holding a committed ROI pays it on every publication for
    a map that cannot change -- the same shape GridTopology.cell_indices
    rejects by name and caches for.
    """

    import time

    import numpy as np

    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointColumn,
        PointTable,
        ValueSchema,
    )
    from zlc_data.snapshot_projection import axis_catalog

    rows = 40_000
    schema = DatasetSchema(
        AxisSpec(AxisId("scan.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(
            rows,
            (
                PointColumn(
                    AxisId("scan.x"), "x", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(index % 200) for index in range(rows)),
                ),
                PointColumn(
                    AxisId("scan.y"), "y", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(index // 200) for index in range(rows)),
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )

    first = axis_catalog(schema)
    started = time.perf_counter()
    again = axis_catalog(schema)
    elapsed = time.perf_counter() - started

    assert again is first
    # The rebuild it replaces is tens of milliseconds; anything in this
    # range is a lookup rather than a second walk over 40 000 rows.
    assert elapsed < 1e-3, elapsed
