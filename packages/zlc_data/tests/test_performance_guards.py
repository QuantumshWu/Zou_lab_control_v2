


def test_a_schema_catalogues_its_axes_once() -> None:
    """The catalog is a fact about the schema; it is built once.

    Repeat and Point axes are already canonical and the catalog is cached;
    a 200x200 scan must not rebuild per-row coordinates on every selection.
    """

    import time

    import numpy as np

    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        DomainSpec,
        SCALAR_DOMAIN,
        ValueSchema,
    )
    from zlc_data.snapshot_projection import axis_catalog

    rows = 40_000
    repeat = AxisSpec(AxisId("scan.repeat"), "repeat", REPEAT, 1, (0,))
    x = AxisSpec(AxisId("scan.x"), "x", SCAN_POINT, 200)
    y = AxisSpec(AxisId("scan.y"), "y", SCAN_POINT, 200)
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec(
            (rows,),
            (x, y),
            (
                tuple(index % 200 for index in range(rows)),
                tuple(index // 200 for index in range(rows)),
            ),
        ),
        SCALAR_DOMAIN,
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
