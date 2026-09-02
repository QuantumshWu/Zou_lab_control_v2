"""The window mask and the rolling trace read ONE history layout.

"Which rows belong to the last N shots" used to be answered twice: the
rolling trace mapped the shot column through the domain machinery while
every histogram and facet path walked the rows in Python and asked numpy
to compare objects.  Both now read the schema's indexed-history layout, so
a grid of site histograms selects exactly the shots the rolling trace draws.
"""

from __future__ import annotations

import numpy as np
from zlc_data import (
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointColumn,
    PointTable,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import AxisRef
from zlc_plot.data_view import DataView

SITES = 3
FRAMES = 2


def _history(shots: int) -> DataView:
    """A Runtime-shaped history: shots x frames rows, sites in the cell."""

    offsets = tuple(int(offset) for offset in np.repeat(np.arange(-shots + 1, 1), FRAMES))
    primary = PointColumn(
        PRIMARY_INDEX_AXIS_ID, "source index", PRIMARY_INDEX, PointColumn.NUMERIC, offsets
    )
    frame = PointColumn(
        AxisId("frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(range(FRAMES)) * shots,
        coordinate_labels=("before", "after") * shots,
    )
    site = AxisSpec(AxisId("site"), "site", SITE, SITES, tuple(range(SITES)))
    schema = DatasetSchema(
        AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(shots * FRAMES, (primary, frame)),
        None,
        ValueSchema(
            (site,), ValidityContract.components(AxisId("site")), np.dtype("<f8"), "count"
        ),
    )
    values = np.arange(shots * FRAMES * SITES, dtype=np.float64).reshape(
        1, shots * FRAMES, SITES
    )
    return DataView(owned_snapshot_from_arrays(schema, values, 1, stream_generation="g"))


def test_the_window_mask_selects_the_shots_the_rolling_trace_draws() -> None:
    view = _history(5)
    history = view.rolling_history(group=AxisRef.data("site"))
    assert history.source_indices.tolist() == [-4, -3, -2, -1, 0]
    for window in (1, 2, 5, 9):
        mask = view.history_validity(window)
        assert mask.shape == (1, 5 * FRAMES, SITES)
        rows = np.flatnonzero(mask[0, :, 0])
        shots_kept = history.source_indices[-window:]
        # Every row of a kept shot, and only those, is inside the window.
        assert rows.tolist() == [
            row for row in range(5 * FRAMES) if -(5 - 1) + row // FRAMES in shots_kept
        ]
        # And the trace's shot planes are the same rows, reduced per site.
        expected = view.samples.value.canonical[0].reshape(5, FRAMES, SITES).mean(axis=1)
        np.testing.assert_allclose(np.asarray(history.values), expected)


def test_the_mask_is_built_once_per_view_however_many_projections_ask() -> None:
    view = _history(4)
    first = view.history_validity(3)
    assert view.history_validity(3) is first
    assert view.history_validity(2) is not first


def test_a_labelled_point_coordinate_names_each_distinct_value_from_its_first_row() -> None:
    view = _history(3)
    domain = view._domain(AxisRef.point("frame"), view._all_positions())
    assert [value.canonical for value in domain.values] == [0, 1]
    assert [value.label for value in domain.values] == ["frame=before", "frame=after"]
