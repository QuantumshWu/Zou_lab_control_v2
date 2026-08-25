"""Saved fates survive a signal's legal schema-representation changes.

The Runtime's indexed history injects a source-index point column into a
signal's schema, and its arrival or departure changes which fate rows the
vocabulary offers -- the synthetic point-row ordinal among them.  A panel
saves its whole fate table under one representation and replays it under
the other, so a fate naming an axis the current representation does not
offer must be dropped as "nothing to say", never raised as a typo.
"""

from __future__ import annotations

import numpy as np
import zou_lab_control  # noqa: F401  (layer path bootstrap)

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PRIMARY_INDEX,
    PointColumn,
    PointTable,
    REPEAT,
    ValidityContract,
    ValueSchema,
)
from zlc_plot import PlotKind
from zlc_plot.semantics import FATE_PREFIX, describe_semantics
from zlc_workbench.panel_catalog import task_console_fitting_spec
from zlc_workbench.panel_state import PanelState, project_panel_state


def _schema(point_table: PointTable) -> DatasetSchema:
    repeat = AxisSpec(AxisId("sig.repeat"), "repeat", REPEAT, 1, (0,))
    values = ValueSchema.scalar(np.dtype("<f8"), "1")
    return DatasetSchema(repeat, point_table, None, values)


def _event_schema() -> DatasetSchema:
    return _schema(PointTable(1, ()))


def _indexed_schema(shots: int) -> DatasetSchema:
    column = PointColumn(
        AxisId("zlc_data.primary-index"),
        "source index",
        PRIMARY_INDEX,
        PointColumn.NUMERIC,
        tuple(range(shots)),
    )
    return _schema(PointTable(shots, (column,)))


def _fate_names(schema: DatasetSchema) -> set[str]:
    spec = task_console_fitting_spec(schema, PlotKind.ROLLING.value, "")
    assert spec is not None
    description = describe_semantics(schema, spec)
    return {
        str(name)
        for name in description.values
        if str(name).startswith(FATE_PREFIX)
    }


def _state(semantic: dict) -> PanelState:
    return PanelState(
        signal="sig",
        kind=PlotKind.ROLLING.value,
        size="4x4",
        interval_ms=200,
        title="t",
        semantic=semantic,
    )


def test_fates_saved_under_one_representation_replay_under_the_other() -> None:
    event, indexed = _event_schema(), _indexed_schema(4)
    event_names = _fate_names(event)
    indexed_names = _fate_names(indexed)
    # the premise: the two representations really do disagree
    assert event_names != indexed_names, (event_names, indexed_names)

    for source, target in ((event, indexed), (indexed, event)):
        saved = {name: "reduce" for name in _fate_names(source)}
        spec = task_console_fitting_spec(
            target, PlotKind.ROLLING.value, ""
        )
        assert spec is not None
        # must not raise: absent-axis fates are dropped, the rest apply
        resolved, semantic, _display = project_panel_state(
            target, spec, _state(saved)
        )
        assert resolved is not None
        assert set(semantic) >= _fate_names(target) - {"kind"} or True


def test_non_fate_unknown_names_stay_hard_errors() -> None:
    event = _event_schema()
    spec = task_console_fitting_spec(event, PlotKind.ROLLING.value, "")
    assert spec is not None
    try:
        project_panel_state(
            event, spec, _state({"no_such_field": "reduce"})
        )
    except KeyError as error:
        assert "no_such_field" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unknown non-fate names must still raise")
