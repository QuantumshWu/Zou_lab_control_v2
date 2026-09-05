"""A scan's Repeat domain is how the scan was executed, and nothing else."""

from __future__ import annotations

import numpy as np
import pytest
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    ValidityContract,
    ValueSchema,
)

from zlc_atom.nodes.scan import scan_dataset_schema, scan_repeat_domain


def _source_schema(*, shots: int) -> DatasetSchema:
    """A source publishing ``shots`` per event over five sites."""

    repeat = AxisSpec(AxisId("shot"), "repeat", REPEAT, shots, tuple(range(shots)))
    event = AxisSpec(AxisId("event"), "event", READOUT_EVENT, 1, (0,))
    site = AxisSpec(AxisId("site"), "site", SITE, 5, tuple(range(5)))
    return DatasetSchema(
        DomainSpec((shots,), (repeat,), (tuple(range(shots)),)),
        DomainSpec((1,), (event,), ((0,),)),
        DomainSpec((site.size,), (site,)),
        ValueSchema(ValidityContract.components(site.axis_id), np.dtype("<f8"), "1"),
    )


def test_the_repeat_domain_is_exactly_the_two_execution_facts() -> None:
    """Scan repeats outer, run repeats inner -- the board's two counters.

    The domain used to carry the source's own Repeat carrier as a third
    axis named "repeat", of size one, beside these: a scan point's value is
    one shot, so that axis said nothing, and said it in every scan signal's
    title.
    """

    domain = scan_repeat_domain(scan_repeats=2, run_repeats=3)
    assert tuple(axis.name for axis in domain.axes) == ("scan repeat", "run repeat")
    assert tuple(axis.size for axis in domain.axes) == (2, 3)
    assert all(axis.role == REPEAT for axis in domain.axes)
    assert domain.shape == (6,)
    assert domain.axis_codes == (
        (0, 0, 0, 1, 1, 1),
        (0, 1, 2, 0, 1, 2),
    ), "scan repeat is the outer index, run repeat the inner"
    with pytest.raises(ValueError, match="scan_repeats"):
        scan_repeat_domain(scan_repeats=0, run_repeats=1)
    with pytest.raises(ValueError, match="run_repeats"):
        scan_repeat_domain(scan_repeats=1, run_repeats=0)


def test_a_scan_dataset_carries_no_repeat_axis_but_its_own() -> None:
    schema = scan_dataset_schema(
        _source_schema(shots=1),
        ((0.0,), (1.0,), (2.0,)),
        (("bias", "code"),),
        scan_repeats=4,
        run_repeats=2,
    )
    assert tuple(axis.name for axis in schema.repeat_domain.axes) == (
        "scan repeat",
        "run repeat",
    )
    assert schema.repeat_domain.size == 8


def test_a_source_publishing_more_than_one_shot_is_refused_by_name() -> None:
    """The scan consumes its source's Repeat carrier; it cannot carry two."""

    with pytest.raises(ValueError, match=r"Repeat carrier \(shot\) holds 3"):
        scan_dataset_schema(
            _source_schema(shots=3),
            ((0.0,), (1.0,)),
            (("bias", "code"),),
        )
