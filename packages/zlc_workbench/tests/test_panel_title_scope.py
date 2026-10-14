from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, DomainSpec, READOUT_EVENT, REPEAT, SITE, ValidityContract, ValueSchema,
)
from zlc_plot.semantics import fate_field_name, scope_fate
from zlc_plot.kinds import AxisDomain, AxisRef

from zlc_workbench.panel_state import panel_data_shape


def _schema(labels):
    repeat = AxisSpec(AxisId("cam.repeat"), "repeat", REPEAT, 1, (0,))
    pair = AxisSpec(AxisId("survival.pair"), "pair", READOUT_EVENT, 3, (0, 1, 2), coordinate_labels=labels)
    site = AxisSpec(AxisId("calibration.site"), "Site", SITE, 2, coordinates=(1, 2))
    return DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((3,), (pair,), ((0, 1, 2),)),
        DomainSpec((2,), (site,)),
        ValueSchema(ValidityContract.components(site.axis_id), np.dtype(bool), "1", name="survived"),
    )


def _pinned(schema, coordinate):
    field = SimpleNamespace(
        name=fate_field_name(AxisRef(AxisDomain.POINT, "survival.pair")),
        label="pair", value=scope_fate(coordinate), choices=(),
    )
    return panel_data_shape(schema, SimpleNamespace(semantics=SimpleNamespace(fields=(field,))))["data_scope"]


def test_a_pinned_labelled_coordinate_is_named_by_its_label() -> None:
    """A Frame Survival pair is 0-5 in the panel's title, never its place 3 on
    the pair axis; an axis without labels still shows its coordinate."""

    assert _pinned(_schema(("0-1", "0-2", "1-2")), 1) == (("pair", "0-2"),)
    assert _pinned(_schema(None), 1) == (("pair", "1"),)
