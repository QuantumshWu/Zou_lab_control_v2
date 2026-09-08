"""Derive: named signals, each an expression, over the outputs of one publication.

The operator writes what a panel should see -- the counts of the occupied
sites, how many sites were occupied, one frame minus another, whether two
verdicts agree -- as signals over the bound producer's outputs, each a row
of a name and an expression, and every signal publishes under its name.
"Not selected" is
validity, so the histogram downstream leaves those cells out; the result's
shape is typed before any value exists; and a name the producer never
publishes, a unit that cannot be added, a geometry that does not match, are
refused by name.
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
    LOGIC_NODE,
    DeriveProcessor,
    ExpressionError,
    Operand,
    evaluate,
    published_names,
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


def _rows(*signals: tuple[str, str]) -> tuple[dict[str, str], ...]:
    """Signals as the form hands them over: a row of name and expression each."""

    return tuple({"name": name, "expression": expression} for name, expression in signals)


def _one(expression: str, outputs: dict[str, Operand] | None = None) -> Operand:
    """One expression, published as the one signal of a derive."""

    return evaluate(_rows(("value", expression)), _outputs() if outputs is None else outputs)["value"]


AGREEMENT = _rows(
    ("agree", "a.occupied.frame(0) == a.occupied.frame(2)"),
    ("counts", "a.counts.frame(1).where(agree)"),
    ("occupied", "a.occupied.frame(0).where(agree)"),
)
BRIGHT = _rows(("bright", "a.counts.frame(1).where(a.occupied.frame(1))"))


def test_the_signals_name_what_they_publish_and_what_they_read() -> None:
    assert published_names(AGREEMENT) == ("agree", "counts", "occupied")
    assert referenced_outputs(AGREEMENT) == ("occupied", "counts")
    assert referenced_outputs(
        _rows(("n", "a.occupied.frame('frame 1').count('site') > 2"))
    ) == ("occupied",)


def test_the_counts_of_the_occupied_sites_are_the_counts_with_validity() -> None:
    """The example the node exists for: a frame's photon counts, only where
    that frame judged the site occupied.  The values are untouched, the
    others are invalid and NaN, the frame keeps its coordinate."""

    result = _one("a.counts.frame(1).where(a.occupied.frame(1))")
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
    by_index = _one("a.counts.frame(0)")
    by_label = _one("a.counts.frame('frame 0')")
    np.testing.assert_array_equal(by_index.values, by_label.values)
    assert by_index.schema == by_label.schema


def test_how_many_sites_were_occupied_is_a_count_per_cycle() -> None:
    result = _one("a.occupied.frame(1).count('site')")
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
    mean = _one("a.counts.frame(1).mean('site')", outputs)
    total = _one("a.counts.frame(1).sum('site')", outputs)
    assert mean.values is not None and mean.valid is not None
    assert total.values is not None and total.valid is not None
    assert not mean.valid[0, 0, 0] and np.isnan(mean.values[0, 0, 0])
    assert mean.valid[1, 0, 0] and mean.values[1, 0, 0] == pytest.approx((230.0 + 33.0) / 2)
    assert total.values[1, 0, 0] == pytest.approx(230.0 + 33.0)
    assert total.values[2, 0, 0] == pytest.approx(15.0 + 250.0 + 35.0)
    any_bright = _one("(a.counts.frame(1) > 100).any('site')", outputs)
    assert any_bright.values is not None
    np.testing.assert_array_equal(any_bright.values[:, 0, 0], [False, True, True])
    assert any_bright.valid is not None and not any_bright.valid[0, 0, 0]


def test_arithmetic_keeps_the_geometry_and_the_unit() -> None:
    # Frame 0 minus frame 1 is a subtraction an operator means; the result
    # is the left operand's frame.
    difference = _one("a.counts.frame(0) - a.counts.frame(1)")
    assert difference.values is not None
    np.testing.assert_allclose(difference.values[:, 0, :], COUNT_VALUES[:, 0, :] - COUNT_VALUES[:, 1, :])
    assert difference.schema.value_schema.value_unit == "count"
    assert difference.schema.point_domain.axes[0].coordinates == (0,)
    scaled = _one("a.counts * 2 + 1")
    assert scaled.values is not None
    np.testing.assert_allclose(scaled.values, COUNT_VALUES * 2 + 1)
    assert scaled.schema.value_schema.value_unit == "count"
    ratio = _one("a.counts.frame(0) / a.counts.frame(1)")
    assert ratio.schema.value_schema.value_unit == "1"
    bright = _one("a.counts > 100")
    assert bright.values is not None and bright.values.dtype == np.dtype("?")
    np.testing.assert_array_equal(bright.values, COUNT_VALUES > 100)
    both = _one("(a.counts > 100) & a.occupied")
    assert both.values is not None
    np.testing.assert_array_equal(both.values, (COUNT_VALUES > 100) & OCCUPIED_VALUES)
    negated = _one("~a.occupied")
    assert negated.values is not None
    np.testing.assert_array_equal(negated.values, ~OCCUPIED_VALUES)


def test_two_verdicts_agree_or_differ_as_boolean_equality() -> None:
    """``==`` between two verdicts is the logic of the verdicts: true where
    both say the same, valid where both were judged.  ``!=`` is its
    complement.  A verdict compared with a number, or ordered, is refused."""

    valid = np.ones(OCCUPIED_VALUES.shape, dtype=bool)
    valid[2, 1, 0] = False
    outputs = {"counts": _operand(COUNT_VALUES, "count"), "occupied": _operand(OCCUPIED_VALUES, "1", valid)}
    agree = _one("a.occupied.frame(0) == a.occupied.frame(1)", outputs)
    differ = _one("a.occupied.frame(0) != a.occupied.frame(1)", outputs)
    for result in (agree, differ):
        assert result.values is not None and result.valid is not None
        assert result.values.dtype == np.dtype("?")
        assert result.schema.value_schema.value_unit == "1"
        assert result.schema.point_domain.axes[0].coordinates == (0,)
        np.testing.assert_array_equal(result.valid[:, 0, :], valid[:, 0, :] & valid[:, 1, :])
    np.testing.assert_array_equal(
        agree.values[:, 0, :], OCCUPIED_VALUES[:, 0, :] == OCCUPIED_VALUES[:, 1, :]
    )
    np.testing.assert_array_equal(differ.values, ~agree.values)
    assert not agree.valid[2, 0, 0]
    for expression, said in (
        ("a.occupied.frame(0) == 1", "== needs two boolean operands"),
        ("a.occupied.frame(0) != a.counts.frame(0)", "!= needs two boolean operands"),
        ("a.occupied.frame(0) < a.occupied.frame(1)", "< needs numeric operands"),
    ):
        with pytest.raises(ExpressionError, match=said):
            _one(expression, outputs)


def test_an_occupancy_agreement_is_three_signals_of_one_derive() -> None:
    """The counts of one frame, kept only where two other frames' verdicts
    agree, and the verdict they agreed on: what a dedicated processor used
    to hard-code as three frame indices is three signals anyone can edit.
    A site neither frame could judge is invalid all the way down."""

    occupied = np.asarray(
        [[[False, True, False, True, True],
          [True, False, True, False, True],
          [False, True, True, False, True]]],
        dtype=bool,
    )
    valid = np.ones_like(occupied)
    valid[:, :, 4] = False
    counts = np.asarray(
        [[[1, 2, 3, 4, 5], [11, 22, 33, 44, 55], [6, 7, 8, 9, 10]]],
        dtype=np.float64,
    )
    outputs = {"counts": _operand(counts, "count"), "occupied": _operand(occupied, "1", valid)}
    results = evaluate(AGREEMENT, outputs)
    assert tuple(results) == ("agree", "counts", "occupied")
    agreement = np.asarray([[[True, True, False, False, False]]])
    np.testing.assert_array_equal(results["occupied"].valid, agreement)
    np.testing.assert_array_equal(results["occupied"].values[:, 0, :2], [[False, True]])
    np.testing.assert_array_equal(results["counts"].valid, agreement)
    np.testing.assert_array_equal(results["counts"].values[:, 0, :2], [[11.0, 22.0]])
    assert np.isnan(results["counts"].values[:, 0, 2:]).all()
    assert results["counts"].schema.point_domain.axes[0].coordinates == (1,)
    assert results["occupied"].schema.point_domain.axes[0].coordinates == (0,)

    # One frame judged against itself keeps every judged site: the no-op.
    one_frame = {
        "counts": _operand(counts[:, :1, :], "count"),
        "occupied": _operand(occupied[:, :1, :], "1", valid[:, :1, :]),
    }
    kept = evaluate(
        _rows(
            ("agree", "a.occupied.frame(0) == a.occupied.frame(0)"),
            ("counts", "a.counts.frame(0).where(agree)"),
        ),
        one_frame,
    )["counts"]
    np.testing.assert_array_equal(kept.valid, valid[:, :1, :])
    np.testing.assert_array_equal(kept.values[:, 0, :4], counts[:, 0, :4])


def test_a_named_signal_is_an_operand_in_the_signals_below_it() -> None:
    results = evaluate(
        _rows(
            ("agree", "a.occupied.frame(0) == a.occupied.frame(1)"),
            ("how_many", "agree.count('site')"),
            ("all_agree", "~(~agree).any('site')"),
        ),
        _outputs(),
    )
    assert tuple(results) == ("agree", "how_many", "all_agree")
    np.testing.assert_array_equal(results["how_many"].values[:, 0, 0], [2, 1, 0])
    np.testing.assert_array_equal(results["all_agree"].values[:, 0, 0], [False, False, False])
    for rows, said in (
        ((), "no signal is written"),
        (_rows(("", "a.counts.frame(0)")), "signal 1 has no name"),
        (_rows(("x y", "a.counts")), "cannot name a signal"),
        (_rows(("class", "a.counts")), "cannot name a signal"),
        (_rows(("a", "a.counts")), "cannot be named after it"),
        (_rows(("x", "a.counts"), ("x", "a.occupied")), "'x' already names signal 1"),
        (_rows(("y", "x + 1"), ("x", "a.counts")), "'x' is not an operand"),
        (_rows(("k", "2")), "publishes no dataset"),
        (_rows(("x", "a")), "read one of its outputs"),
        (_rows(("x", "a.counts"), ("y", "x.counts")), "'x' has no outputs"),
        (_rows(("x", "")), r"signal 1 \(x\) has no expression"),
        (_rows(("x", "a.counts +")), "cannot read the expression"),
        ("x = a.counts", "rows of a name and an expression"),
    ):
        with pytest.raises(ExpressionError, match=said):
            evaluate(rows, _outputs())


def test_the_typing_pass_knows_the_shape_without_any_values() -> None:
    schemas = {name: Operand(operand.schema) for name, operand in _outputs().items()}
    typed = evaluate(
        _rows(
            ("agree", "a.occupied.frame(0) == a.occupied.frame(1)"),
            ("mean", "a.counts.frame(1).where(agree).mean('site')"),
        ),
        schemas,
    )
    assert typed["agree"].values is None and typed["agree"].is_boolean
    mean = typed["mean"]
    assert mean.values is None
    assert mean.schema.point_domain.size == 1 and mean.schema.cell_domain == SCALAR_DOMAIN
    assert mean.schema.value_schema.value_unit == "count"


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
        ("a.counts +", "cannot read the expression"),
    ],
)
def test_what_cannot_be_typed_is_refused_by_name(expression: str, said: str) -> None:
    with pytest.raises(ExpressionError) as refusal:
        _one(expression)
    assert said in str(refusal.value)


def _signal(name: str, values: np.ndarray, unit: str, **placement) -> SignalValue:
    return SignalValue(name, _snapshot(values, unit, revision=3), **placement)


def test_the_processor_publishes_every_signal_with_its_provenance() -> None:
    processor = DeriveProcessor(
        expressions=_rows(
            ("bright", "a.counts.frame(1).where(a.occupied.frame(1))"),
            ("how_many", "a.occupied.frame(1).count('site')"),
        ),
        primary_output="counts",
    )
    assert processor.dataset_input_siblings == ("occupied",)
    assert [
        (item.name, item.contract_id, item.index_by_source)
        for item in processor.dataset_output_declarations
    ] == [("bright", "derive.bright", True), ("how_many", "derive.how_many", True)]
    outputs = processor.evaluate_inputs(
        {
            "a": _signal(COUNTS, COUNT_VALUES, "count", coverage=None),
            "occupied": _signal(OCCUPIED, OCCUPIED_VALUES, "1", coverage=None),
        }
    )
    assert tuple(outputs) == ("bright", "how_many")
    bright, how_many = outputs["bright"], outputs["how_many"]
    assert bright.declaration is processor.dataset_output_declarations[0]
    assert how_many.declaration is processor.dataset_output_declarations[1]
    assert bright.snapshot.block.revision.value == 3
    assert bright.snapshot.block.values.shape == (3, 1, 3)
    np.testing.assert_array_equal(
        np.asarray(bright.snapshot.expanded_validity(), dtype=bool),
        OCCUPIED_VALUES[:, 1:2, :],
    )
    assert bright.coverage == DatasetCoverage(3, 3)
    assert bright.canonical_schema == bright.snapshot.block.schema
    assert bright.cell_origin == (0, 0)
    np.testing.assert_array_equal(how_many.snapshot.block.values[:, 0, 0], [2, 0, 3])
    assert how_many.run_record is bright.run_record
    assert bright.run_record["parameters"]["expressions"] == processor.expressions
    assert bright.run_record["parameters"]["inputs"] == {"counts": COUNTS, "occupied": OCCUPIED}


def test_a_finite_run_keeps_exact_bookkeeping_through_a_frame_selection() -> None:
    processor = DeriveProcessor(
        expressions=_rows(("mean", "a.counts.frame(0).mean('site')")), primary_output="counts"
    )
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
    )["mean"]
    assert output.coverage == DatasetCoverage(4, 6)
    assert output.cell_origin == (1, 0)
    assert output.canonical_schema is not None
    assert output.canonical_schema.repeat_domain.size == 6
    assert output.canonical_schema.point_domain.size == 1
    assert output.canonical_schema.cell_domain == SCALAR_DOMAIN
    monitor = processor.evaluate(
        _signal(COUNTS, COUNT_VALUES, "count", coverage=MonitorCoverage(4, 6))
    )["mean"]
    assert monitor.coverage == MonitorCoverage(2, 3)
    assert monitor.canonical_schema is None and monitor.cell_origin is None


def test_the_bound_output_must_be_one_the_signals_read() -> None:
    with pytest.raises(ValueError, match="not one the signals read"):
        DeriveProcessor(expressions=_rows(("n", "a.occupied.count('site')")), primary_output="counts")


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
        expressions=BRIGHT,
    )
    assert isinstance(node, DeriveProcessor)
    assert node.primary_output == "counts"
    assert node.dataset_input_siblings == ("occupied",)
    assert plane.asked == [(COUNTS, ("counts", "occupied"))]

    bound_elsewhere = LOGIC_NODE.instantiate(
        signal_plane=plane,
        source_signal=OCCUPIED,
        expressions=BRIGHT,
    )
    assert bound_elsewhere.primary_output == "occupied"
    assert bound_elsewhere.dataset_input_siblings == ("counts",)

    with pytest.raises(ValueError, match="no sibling outputs \\('brightness',\\)"):
        LOGIC_NODE.instantiate(
            signal_plane=plane, source_signal=COUNTS, expressions=_rows(("x", "a.brightness.frame(1)"))
        )
    with pytest.raises(ValueError, match="missing required authoring field 'expressions'"):
        LOGIC_NODE.instantiate(signal_plane=plane, source_signal=COUNTS, expressions=())


def test_the_descriptor_declares_what_a_draft_publishes_before_anything_runs() -> None:
    """A derive's outputs are the names of its signals, so the descriptor
    cannot list them once; it answers per draft, without a plane.  A draft
    that cannot be read publishes nothing yet, and the schema's validator
    is what says why -- to the editor, in the draft's own words."""

    assert LOGIC_NODE.outputs == () and LOGIC_NODE.declare_outputs is not None
    declared = LOGIC_NODE.outputs_for({"expressions": AGREEMENT})
    assert [(item.name, item.contract_id) for item in declared] == [
        ("agree", "derive.agree"),
        ("counts", "derive.counts"),
        ("occupied", "derive.occupied"),
    ]
    broken = _rows(("counts", "counts +"))
    assert LOGIC_NODE.outputs_for({"expressions": broken}) == ()
    assert LOGIC_NODE.outputs_for({}) == ()
    schema = LOGIC_NODE.authoring_schema
    with pytest.raises(ValueError, match="cannot read the expression"):
        schema.project_values({"expressions": broken})
    # A row is a form of its columns: a name still missing is refused as
    # such at the build, named by the row it is missing from.
    with pytest.raises(ValueError, match="Signals row 1: missing required authoring field 'name'"):
        schema.project_values({"expressions": _rows(("", "a.counts"))})
    # A draft still being typed is not complete, and completeness is the
    # build's law: with no row yet the validator waits; with a row being
    # written it says, in the editor, what the row still needs.
    assert schema.draft_values({"expressions": ()}) == {"expressions": ()}
    assert schema.draft_values({}) == {"expressions": ()}
    with pytest.raises(ValueError, match="signal 1 has no name"):
        schema.draft_values({"expressions": _rows(("", "a.counts"))})
    # Rows come back from a saved layout as lists of mappings; the value
    # is the same tuple of rows either way.
    assert schema.project_values({"expressions": [dict(row) for row in AGREEMENT]}) == {
        "expressions": AGREEMENT
    }
