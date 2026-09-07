"""Derive: one expression over the outputs of one publication.

The operator writes what a panel should see -- the counts of the occupied
sites, how many sites were occupied, one frame minus another -- as an
expression over the bound producer's outputs.  "Not selected" is validity,
so the histogram downstream leaves those cells out; the result's shape is
typed before any value exists; and a name the producer never publishes,
a unit that cannot be added, a geometry that does not match, are refused
by name.
"""

from __future__ import annotations

import numpy as np
import pytest
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    READOUT_EVENT,
    REPEAT,
    SCALAR_DOMAIN,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import DatasetCoverage, MonitorCoverage, SignalValue

from zlc_atom.nodes.derive import (
    DERIVE_OUTPUT,
    LOGIC_NODE,
    DeriveProcessor,
    ExpressionError,
    Operand,
    evaluate,
    referenced_outputs,
)

COUNTS = "@logic/occupancy/counts"
OCCUPIED = "@logic/occupancy/occupied"


def _schema(cycles: int, frames: int, sites: int, dtype: np.dtype, unit: str) -> DatasetSchema:
    site_axis = AxisSpec(AxisId("occupancy.site"), "site", SITE, sites)
    cycle_axis = AxisSpec(AxisId("camera.cycle"), "cycle", REPEAT, cycles)
    frame_axis = AxisSpec(
        AxisId("camera.frames.frame"),
        "frame",
        READOUT_EVENT,
        frames,
        tuple(range(frames)),
        coordinate_labels=tuple(f"frame {index}" for index in range(frames)),
    )
    return DatasetSchema(
        DomainSpec((cycles,), (cycle_axis,), (tuple(range(cycles)),)),
        DomainSpec((frames,), (frame_axis,), (tuple(range(frames)),)),
        DomainSpec((sites,), (site_axis,)),
        ValueSchema(ValidityContract.components(site_axis.axis_id), dtype, unit),
    )


def _snapshot(values: np.ndarray, unit: str, valid: np.ndarray | None = None, *, revision: int = 0):
    cycles, frames, sites = values.shape
    schema = _schema(cycles, frames, sites, values.dtype, unit)
    return owned_snapshot_from_arrays(
        schema,
        values,
        revision,
        validity=np.ones(values.shape, dtype=bool) if valid is None else valid,
    )


def _operand(values: np.ndarray, unit: str, valid: np.ndarray | None = None) -> Operand:
    snapshot = _snapshot(values, unit, valid)
    return Operand(
        snapshot.block.schema,
        np.asarray(snapshot.block.values),
        np.asarray(snapshot.expanded_validity(), dtype=bool),
    )


COUNT_VALUES = np.array(
    [
        [[10.0, 200.0, 30.0], [11.0, 210.0, 31.0]],
        [[12.0, 220.0, 32.0], [13.0, 230.0, 33.0]],
        [[14.0, 240.0, 34.0], [15.0, 250.0, 35.0]],
    ]
)
OCCUPIED_VALUES = np.array(
    [
        [[False, True, False], [False, True, True]],
        [[True, True, False], [False, False, False]],
        [[False, False, False], [True, True, True]],
    ]
)


def _outputs(valid_counts: np.ndarray | None = None) -> dict[str, Operand]:
    return {
        "counts": _operand(COUNT_VALUES, "count", valid_counts),
        "occupied": _operand(OCCUPIED_VALUES, "1"),
    }


def test_the_expression_names_the_outputs_it_reads() -> None:
    assert referenced_outputs("a.counts.frame(1).where(a.occupied.frame(1))") == (
        "counts",
        "occupied",
    )
    assert referenced_outputs("a.occupied.frame('frame 1').count('site') > 2") == ("occupied",)


def test_the_counts_of_the_occupied_sites_are_the_counts_with_validity() -> None:
    """The example the node exists for: a frame's photon counts, only where
    that frame judged the site occupied.  The values are untouched, the
    others are invalid and NaN, the frame keeps its coordinate."""

    result = evaluate("a.counts.frame(1).where(a.occupied.frame(1))", _outputs())
    assert result.values is not None and result.valid is not None
    assert result.values.shape == (3, 1, 3)
    expected_valid = OCCUPIED_VALUES[:, 1:2, :]
    np.testing.assert_array_equal(result.valid, expected_valid)
    np.testing.assert_array_equal(
        np.isnan(result.values), ~expected_valid
    )
    np.testing.assert_array_equal(result.values[expected_valid], COUNT_VALUES[:, 1:2, :][expected_valid])
    schema = result.schema
    assert schema.point_domain.size == 1
    frame_axis = schema.point_domain.axes[0]
    assert frame_axis.coordinates == (1,) and frame_axis.coordinate_labels == ("frame 1",)
    assert schema.cell_domain.axes[0].role == SITE
    assert schema.value_schema.value_unit == "count"
    assert schema.value_schema.dtype == np.dtype(np.float64)


def test_a_frame_is_addressed_by_label_as_well_as_by_index() -> None:
    by_index = evaluate("a.counts.frame(0)", _outputs())
    by_label = evaluate("a.counts.frame('frame 0')", _outputs())
    np.testing.assert_array_equal(by_index.values, by_label.values)
    assert by_index.schema == by_label.schema


def test_how_many_sites_were_occupied_is_a_count_per_cycle() -> None:
    result = evaluate("a.occupied.frame(1).count('site')", _outputs())
    assert result.values is not None and result.valid is not None
    np.testing.assert_array_equal(result.values[:, 0, 0], [2, 0, 3])
    assert result.values.dtype == np.dtype(np.int64)
    assert result.valid.all()
    assert result.schema.cell_domain == SCALAR_DOMAIN
    assert result.schema.value_schema.value_unit == "1"
    assert result.schema.value_schema.validity_contract == ValidityContract.value()


def test_reductions_leave_invalid_cells_out() -> None:
    valid = np.ones(COUNT_VALUES.shape, dtype=bool)
    valid[0, 1, :] = False  # a cycle whose frame 1 was never judged
    valid[1, 1, 0] = False
    outputs = _outputs(valid)
    mean = evaluate("a.counts.frame(1).mean('site')", outputs)
    total = evaluate("a.counts.frame(1).sum('site')", outputs)
    assert mean.values is not None and mean.valid is not None
    assert total.values is not None and total.valid is not None
    assert not mean.valid[0, 0, 0] and np.isnan(mean.values[0, 0, 0])
    assert mean.valid[1, 0, 0] and mean.values[1, 0, 0] == pytest.approx((230.0 + 33.0) / 2)
    assert total.values[1, 0, 0] == pytest.approx(230.0 + 33.0)
    assert total.values[2, 0, 0] == pytest.approx(15.0 + 250.0 + 35.0)
    any_bright = evaluate("(a.counts.frame(1) > 100).any('site')", outputs)
    assert any_bright.values is not None
    np.testing.assert_array_equal(any_bright.values[:, 0, 0], [False, True, True])
    assert any_bright.valid is not None and not any_bright.valid[0, 0, 0]


def test_arithmetic_keeps_the_geometry_and_the_unit() -> None:
    # Frame 0 minus frame 1 is a subtraction an operator means; the result
    # is the left operand's frame.
    difference = evaluate("a.counts.frame(0) - a.counts.frame(1)", _outputs())
    assert difference.values is not None
    np.testing.assert_allclose(difference.values[:, 0, :], COUNT_VALUES[:, 0, :] - COUNT_VALUES[:, 1, :])
    assert difference.schema.value_schema.value_unit == "count"
    assert difference.schema.point_domain.axes[0].coordinates == (0,)
    scaled = evaluate("a.counts * 2 + 1", _outputs())
    assert scaled.values is not None
    np.testing.assert_allclose(scaled.values, COUNT_VALUES * 2 + 1)
    assert scaled.schema.value_schema.value_unit == "count"
    ratio = evaluate("a.counts.frame(0) / a.counts.frame(1)", _outputs())
    assert ratio.schema.value_schema.value_unit == "1"
    bright = evaluate("a.counts > 100", _outputs())
    assert bright.values is not None and bright.values.dtype == np.dtype("?")
    np.testing.assert_array_equal(bright.values, COUNT_VALUES > 100)
    both = evaluate("(a.counts > 100) & a.occupied", _outputs())
    assert both.values is not None
    np.testing.assert_array_equal(both.values, (COUNT_VALUES > 100) & OCCUPIED_VALUES)
    negated = evaluate("~a.occupied", _outputs())
    assert negated.values is not None
    np.testing.assert_array_equal(negated.values, ~OCCUPIED_VALUES)


def test_the_typing_pass_knows_the_shape_without_any_values() -> None:
    schemas = {name: Operand(operand.schema) for name, operand in _outputs().items()}
    typed = evaluate("a.counts.frame(1).where(a.occupied.frame(1)).mean('site')", schemas)
    assert typed.values is None
    assert typed.schema.point_domain.size == 1 and typed.schema.cell_domain == SCALAR_DOMAIN
    assert typed.schema.value_schema.value_unit == "count"


@pytest.mark.parametrize(
    ("expression", "said"),
    [
        ("a.counts + a.occupied", "needs numeric operands"),
        ("a.counts.frame(0) - a.counts", "point domains differ"),
        ("a.counts.frame(0) + a.counts.frame(0).count('site')", "cell domains differ"),
        ("a.counts.where(a.counts)", "boolean operand as its mask"),
        ("a.counts.frame(7)", "outside this 2-frame cycle"),
        ("a.counts.frame('frame 9')", "no frame is labelled"),
        ("a.counts.sum('frame')", "cannot be reduced"),
        ("a.occupied.sum('site')", "count a boolean one"),
        ("a.counts.any('site')", "needs a boolean operand"),
        ("a.counts.smooth()", "is not a method of an operand"),
        ("a.brightness", "has no output 'brightness'"),
        ("b.counts", "is not an operand"),
        ("a.counts * a.counts", "cannot multiply count by count"),
        ("__import__('os')", "only an operand's own methods"),
        ("a.counts if True else 1", "not admitted"),
        ("", "is empty"),
        ("a.counts +", "cannot read the expression"),
    ],
)
def test_what_cannot_be_typed_is_refused_by_name(expression: str, said: str) -> None:
    with pytest.raises(ExpressionError) as refusal:
        evaluate(expression, _outputs())
    assert said in str(refusal.value)


def _signal(name: str, values: np.ndarray, unit: str, **placement) -> SignalValue:
    return SignalValue(name, _snapshot(values, unit, revision=3), **placement)


def test_the_processor_publishes_the_result_with_its_provenance() -> None:
    processor = DeriveProcessor(
        expression="a.counts.frame(1).where(a.occupied.frame(1))",
        primary_output="counts",
    )
    assert processor.dataset_input_siblings == ("occupied",)
    outputs = processor.evaluate_inputs(
        {
            "a": _signal(COUNTS, COUNT_VALUES, "count", coverage=None),
            "occupied": _signal(OCCUPIED, OCCUPIED_VALUES, "1", coverage=None),
        }
    )
    (output,) = outputs.values()
    assert output.declaration is DERIVE_OUTPUT
    assert output.snapshot.block.revision.value == 3
    assert output.snapshot.block.values.shape == (3, 1, 3)
    np.testing.assert_array_equal(
        np.asarray(output.snapshot.expanded_validity(), dtype=bool),
        OCCUPIED_VALUES[:, 1:2, :],
    )
    assert output.coverage == DatasetCoverage(3, 3)
    assert output.canonical_schema == output.snapshot.block.schema
    assert output.cell_origin == (0, 0)
    assert output.run_record["parameters"]["expression"] == processor.expression
    assert output.run_record["parameters"]["inputs"] == {"counts": COUNTS, "occupied": OCCUPIED}


def test_a_finite_run_keeps_exact_bookkeeping_through_a_frame_selection() -> None:
    processor = DeriveProcessor(expression="a.counts.frame(0).mean('site')", primary_output="counts")
    assert processor.dataset_input_siblings == ()
    # A six-cycle run of which this event is cycles 1..3, eight of twelve
    # cells written so far.
    full = _schema(6, 2, 3, np.dtype(np.float64), "count")
    output = processor.evaluate(
        _signal(
            COUNTS,
            COUNT_VALUES,
            "count",
            coverage=DatasetCoverage(8, 12),
            canonical_schema=full,
            cell_origin=(1, 0),
        )
    )["value"]
    assert output.coverage == DatasetCoverage(4, 6)
    assert output.cell_origin == (1, 0)
    assert output.canonical_schema is not None
    assert output.canonical_schema.repeat_domain.size == 6
    assert output.canonical_schema.point_domain.size == 1
    assert output.canonical_schema.cell_domain == SCALAR_DOMAIN
    monitor = processor.evaluate(
        _signal(COUNTS, COUNT_VALUES, "count", coverage=MonitorCoverage(4, 6))
    )["value"]
    assert monitor.coverage == MonitorCoverage(2, 3)
    assert monitor.canonical_schema is None and monitor.cell_origin is None


def test_the_bound_output_must_be_one_the_expression_reads() -> None:
    with pytest.raises(ValueError, match="not one the expression reads"):
        DeriveProcessor(expression="a.occupied.count('site')", primary_output="counts")


class _Plane:
    """The one question a build asks the plane: which outputs the bound
    producer has, resolved to their signals."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def resolve_sibling_signals(self, signal_name: str, sibling_outputs: tuple[str, ...]) -> tuple[str, ...]:
        self.asked.append((signal_name, tuple(sibling_outputs)))
        missing = tuple(name for name in sibling_outputs if name not in self.outputs)
        if missing:
            raise ValueError(f"signal {signal_name!r} has no sibling outputs {missing!r}")
        return tuple(self.outputs[name] for name in sibling_outputs)


def test_the_build_asks_the_plane_which_outputs_exist_and_which_is_bound() -> None:
    plane = _Plane({"counts": COUNTS, "occupied": OCCUPIED, "frame_judged": "@logic/occupancy/frame_judged"})
    node = LOGIC_NODE.instantiate(
        signal_plane=plane,
        source_signal=COUNTS,
        expression="a.counts.frame(1).where(a.occupied.frame(1))",
    )
    assert isinstance(node, DeriveProcessor)
    assert node.primary_output == "counts"
    assert node.dataset_input_siblings == ("occupied",)
    assert plane.asked == [(COUNTS, ("counts", "occupied"))]

    bound_elsewhere = LOGIC_NODE.instantiate(
        signal_plane=plane,
        source_signal=OCCUPIED,
        expression="a.counts.frame(1).where(a.occupied.frame(1))",
    )
    assert bound_elsewhere.primary_output == "occupied"
    assert bound_elsewhere.dataset_input_siblings == ("counts",)

    with pytest.raises(ValueError, match="no sibling outputs \\('brightness',\\)"):
        LOGIC_NODE.instantiate(
            signal_plane=plane, source_signal=COUNTS, expression="a.brightness.frame(1)"
        )
    with pytest.raises(ValueError, match="missing required authoring field 'expression'"):
        LOGIC_NODE.instantiate(signal_plane=plane, source_signal=COUNTS, expression="")
