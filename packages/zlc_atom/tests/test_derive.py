"""Derive executes ordinary Python on real three-domain Dataset operands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, DomainSpec, READOUT_EVENT, REPEAT,
    SCALAR_DOMAIN, SITE, ValidityContract, ValueSchema, owned_snapshot_from_arrays,
)
from zlc_runtime import DatasetCoverage, MonitorCoverage, SignalValue

from zlc_atom.nodes.derive import (
    LOGIC_NODE, DeriveProcessor, ExpressionError, Operand, evaluate, published_names,
)

COUNTS = "@logic/occupancy/counts"
OCCUPIED = "@logic/occupancy/occupied"


def _schema(cycles: int, frames: int, sites: int, dtype: np.dtype, unit: str) -> DatasetSchema:
    site = AxisSpec(AxisId("occupancy.site"), "site", SITE, sites)
    cycle = AxisSpec(AxisId("camera.cycle"), "cycle", REPEAT, cycles)
    frame = AxisSpec(AxisId("camera.frames.frame"), "frame", READOUT_EVENT, frames,
                     tuple(range(frames)), coordinate_labels=tuple(f"frame {i}" for i in range(frames)))
    return DatasetSchema(
        DomainSpec((cycles,), (cycle,), (tuple(range(cycles)),)),
        DomainSpec((frames,), (frame,), (tuple(range(frames)),)),
        DomainSpec((sites,), (site,)),
        ValueSchema(ValidityContract.components(site.axis_id), dtype, unit),
    )


def _snapshot(values, unit, valid=None, *, revision=0):
    schema = _schema(*values.shape, values.dtype, unit)
    return owned_snapshot_from_arrays(schema, values, revision,
        validity=np.ones(values.shape, bool) if valid is None else valid)


def _operand(values, unit="count", valid=None):
    snapshot = _snapshot(np.asarray(values), unit, valid)
    return Operand(snapshot.block.schema, snapshot.block.values, snapshot.expanded_validity())


def _rows(*signals):
    return tuple({"name": name, "code": code} for name, code in signals)


def _one(code, outputs):
    return evaluate(_rows(("answer", code)), outputs)["answer"]


def test_conditional_repeat_means_publish_one_35_site_vector() -> None:
    repeat = np.arange(50)[:, None]
    site = np.arange(35)[None, :]
    counts = np.broadcast_to(7 + site * 3 + repeat * .2 + 100 * (repeat % 2), (50, 35)).copy()
    values = np.stack((counts - 1, counts, counts + 1), axis=1)
    occupied = np.broadcast_to((repeat % 2).astype(bool)[:, None, :], (50, 3, 35)).copy()
    occupied[:, 2, 34] = False  # no jointly occupied samples for this site
    judged = np.ones(occupied.shape, bool)
    judged[:, :, 33] = False
    valid = np.ones(values.shape, bool)
    valid[1::5, 1, 32] = False
    outputs = {"counts": _operand(values, valid=valid),
               "occupied": _operand(occupied, "1", judged)}
    code = (
        "c = a.counts.isel(frame=1)\n"
        "o0 = a.occupied.isel(frame=0)\n"
        "o2 = a.occupied.isel(frame=2)\n"
        "result = c.mean('cycle', where=o0 & o2) - c.mean('cycle', where=~o0 & ~o2)"
    )
    result = _one(code, outputs)
    assert result.shape == (1, 1, 35)
    assert result.schema.repeat_domain.axes == result.schema.point_domain.axes == ()
    assert result.schema.cell_domain == outputs["counts"].schema.cell_domain
    assert result.unit == "count"
    for index in range(35):
        available = valid[:, 1, index] & judged[:, 0, index] & judged[:, 2, index]
        bright = available & occupied[:, 0, index] & occupied[:, 2, index]
        dark = available & ~occupied[:, 0, index] & ~occupied[:, 2, index]
        assert result.valid[0, 0, index] == bool(bright.any() and dark.any())
        if bright.any() and dark.any():
            assert result.values[0, 0, index] == pytest.approx(
                counts[bright, index].mean() - counts[dark, index].mean())
    assert np.isnan(result.values[0, 0, 33:]).all()


def test_scalar_isel_drops_only_its_named_axis_while_lists_keep_length_one() -> None:
    source = _operand(np.arange(12.0).reshape(2, 2, 3))
    scalar = source.isel(frame=1)
    kept = source.isel(frame=[1])
    assert scalar.shape == kept.shape == (2, 1, 3)
    assert scalar.schema.point_domain.axes == ()
    assert kept.schema.point_domain.axes[0].size == 1
    assert kept.schema.point_domain.axes[0].coordinates == (1,)
    assert kept.schema.point_domain.axes[0].coordinate_labels == ("frame 1",)
    np.testing.assert_array_equal(scalar.values, kept.values)
    np.testing.assert_array_equal(source.sel(frame=1).values, scalar.values)
    assert source.sel(frame=[1]).schema == kept.schema
    with pytest.raises(ValueError, match="axes/coordinates differ"):
        _ = scalar + kept
    one = source.isel(cycle=[0], frame=1, site=2)
    assert one.shape == (1, 1, 1)
    assert one.schema.repeat_domain.axes[0].name == "cycle"
    assert one.schema.point_domain.axes == ()
    assert one.schema.cell_domain == SCALAR_DOMAIN
    assert one.isel(cycle=0).schema.repeat_domain.axes == ()
    source = _operand(np.arange(18.0).reshape(2, 3, 3))
    for choice in ([2, 0], slice(None, None, -1)):
        indexes = [2, 0] if isinstance(choice, list) else [2, 1, 0]
        selected = source.isel(cycle=[1, 0], frame=choice, site=[2, 0])
        np.testing.assert_array_equal(selected.values,
            np.take(np.take(source.values[::-1], indexes, axis=1), [2, 0], axis=2))
        frame = selected.schema.point_domain.axes[0]
        assert frame.coordinates == tuple(indexes)
        assert frame.coordinate_labels == tuple(f"frame {i}" for i in indexes)
        assert tuple(selected.schema.point_domain.codes(frame.axis_id)) == tuple(range(len(indexes)))


def test_sparse_logical_axes_reduce_without_densifying_or_averaging_means() -> None:
    original = _schema(2, 5, 2, np.dtype(float), "count")
    a = AxisSpec(AxisId("scan.a"), "a", READOUT_EVENT, 2, (10, 20))
    b = AxisSpec(AxisId("scan.b"), "b", READOUT_EVENT, 3, (1, 2, 3))
    a_codes, b_codes = (0, 0, 1, 1, 1), (0, 2, 0, 1, 2)
    schema = replace(original, point_domain=DomainSpec((5,), (a, b), (a_codes, b_codes)))
    values = np.arange(20.0).reshape(2, 5, 2)
    valid = np.ones(values.shape, bool)
    valid[0, 0, :] = False
    valid[1, 2:4, 1] = False
    source = Operand(schema, values, valid)
    for reduced, remaining, codes in (("a", b, b_codes), ("b", a, a_codes)):
        result = source.mean(reduced)
        assert result.schema.point_domain.axes == (remaining,)
        order = tuple(dict.fromkeys(codes))
        assert result.shape == (2, len(order), 2)
        assert tuple(result.schema.point_domain.codes(remaining.axis_id)) == order
        for row, coordinate in enumerate(order):
            selected = np.asarray(codes) == coordinate
            for cycle in range(2):
                for site in range(2):
                    samples = values[cycle, selected, site][valid[cycle, selected, site]]
                    assert result.valid[cycle, row, site] == bool(samples.size)
                    if samples.size:
                        assert result.values[cycle, row, site] == pytest.approx(samples.mean())
    # Unequal sample counts across repeat, mapped Point groups and Cell-data:
    # divide once by total valid count, not by a count of intermediate means.
    result = source.mean(("cycle", "b", "site"))
    assert result.shape == (1, 2, 1)
    assert result.schema.repeat_domain.axes == () and result.schema.cell_domain == SCALAR_DOMAIN
    for index in (0, 1):
        selected = np.asarray(a_codes) == index
        samples = values[:, selected, :][valid[:, selected, :]]
        assert result.values[0, index, 0] == pytest.approx(samples.mean())
    std = source.std(("cycle", "a", "b", "site"))
    assert std.values.item() == pytest.approx(values[valid].std())
    selected = source.isel(a=1)
    assert selected.shape == (2, 3, 2)
    assert tuple(axis.name for axis in selected.schema.point_domain.axes) == ("b",)
    np.testing.assert_array_equal(selected.values, values[:, 2:, :])
    reordered = source.isel(b=[2, 0])
    assert reordered.schema.point_domain.axes[1].coordinates == (3, 1)
    assert tuple(reordered.schema.point_domain.codes(b.axis_id)) == (0, 0, 1, 1)
    np.testing.assert_array_equal(reordered.values, values[:, [1, 4, 0, 2], :])
    np.testing.assert_array_equal(reordered.valid, valid[:, [1, 4, 0, 2], :])


def test_where_and_reductions_keep_invalid_and_empty_groups_explicit(monkeypatch) -> None:
    values = np.array([[[10., 20., 30.]], [[40., 50., 60.]]])
    source = _operand(values, valid=np.array([[[False, False, False]], [[True, False, True]]]))
    mask = _operand(np.array([[[True, False, True]], [[False, True, True]]]), "1")
    filtered = source.where(mask)
    np.testing.assert_array_equal(filtered.values, values)
    np.testing.assert_array_equal(filtered.valid, source.valid & mask.values)
    assert filtered.masked.mask[1, 0].tolist() == [True, True, False]
    for operation, expected in (("sum", 100.), ("mean", 50.), ("min", 40.), ("max", 60.), ("std", 10.), ("count", 2)):
        result = getattr(source, operation)("site")
        assert not result.valid[0, 0, 0]
        assert result.valid[1, 0, 0] and result.values[1, 0, 0] == pytest.approx(expected)
    boolean = mask.with_values(mask.values, valid=source.valid)
    for operation, expected in (("count", 1), ("any", True), ("all", False)):
        result = getattr(boolean, operation)("site")
        assert not result.valid[0, 0, 0]
        assert result.valid[1, 0, 0] and result.values[1, 0, 0] == expected
        assert result.unit == "1"
    empty = mask.with_values(np.zeros(mask.shape, dtype=bool))
    for operation in ("sum", "mean", "min", "max", "std", "count"):
        assert not getattr(source, operation)(("cycle", "site"), where=empty).valid.any()
    for operation in ("all", "any", "count"):
        assert not getattr(mask, operation)(("cycle", "site"), where=empty).valid.any()
    from zlc_atom.nodes.derive import expression

    grouped = expression._group_reduce
    count_inputs = []

    def group_reduce(values, codes, count, axis, operation):
        count_inputs.append(values.dtype.kind)
        return grouped(values, codes, count, axis, operation)

    monkeypatch.setattr(expression, "_group_reduce", group_reduce)
    counted = source.count("cycle")
    assert count_inputs == ["b"], "numeric count consumes validity, not a discarded value sum"
    np.testing.assert_array_equal(counted.values, [[[1, 0, 1]]])


def test_python_locals_numpy_and_previous_outputs_do_not_create_a_second_schema() -> None:
    source = _operand(np.arange(6.0).reshape(1, 2, 3))
    rows = _rows(
        ("scaled", "x = a.counts\nlocal = np.clip(x.values, 1, 4)\nresult = x.with_values(local)"),
        ("difference", "scaled.isel(frame=1) - scaled.isel(frame=0)"),
        ("condition", "result = (difference > 0) & (difference < 5)"),
    )
    assert published_names(rows) == ("scaled", "difference", "condition")
    results = evaluate(rows, {"counts": source})
    assert tuple(results) == published_names(rows)
    assert results["scaled"].schema == source.schema
    np.testing.assert_array_equal(results["scaled"].values, np.clip(source.values, 1, 4))
    np.testing.assert_array_equal(results["difference"].values, [[[2., 3., 2.]]])
    assert results["difference"].unit == "count"
    assert results["condition"].values.all() and results["condition"].dtype == np.dtype(bool)
    assert not source.values.flags.writeable


def test_code_errors_are_named_without_escaping_the_worker() -> None:
    outputs = {"counts": _operand(np.ones((1, 1, 1)))}
    for code, message in (
        ("result = (", "line 1"),
        ("result = missing", "NameError"),
        ("raise ValueError('deliberate')", "ValueError: deliberate"),
        ("raise SystemExit('stop')", "SystemExit: stop"),
        ("raise KeyboardInterrupt('stop')", "KeyboardInterrupt: stop"),
        ("result = 3", "Dataset schema"),
        ("temporary = a.counts", "Dataset schema"),
        ("result = a.counts.with_values(np.zeros((2, 2)))", "three-domain physical shape"),
        ("result = a.counts.where(a.counts)", "boolean Dataset condition"),
    ):
        with pytest.raises(ExpressionError, match=message):
            _one(code, outputs)
    for rows in (
        (), ({"name": "old", "expression": "a.counts"},),
        _rows(("result", "a.counts")), _rows(("same", "a.counts"), ("same", "a.counts")),
    ):
        with pytest.raises(ExpressionError):
            evaluate(rows, outputs)


def test_processor_publishes_current_estimates_and_the_exact_program_provenance() -> None:
    values = np.arange(18.0).reshape(3, 2, 3)
    occupied = values % 2 == 0
    rows = _rows(("bright", "a.counts.isel(frame=1).where(a.occupied.isel(frame=1))"),
                 ("mean", "bright.mean('cycle')"))
    processor = DeriveProcessor(expressions=rows, primary_output="counts",
        input_outputs=("counts", "occupied"), input_view="run")
    assert processor.dataset_input_siblings == ("occupied",)
    assert processor.dataset_input_view == "run"
    primary = SignalValue(COUNTS, _snapshot(values, "count", revision=3),
        coverage=DatasetCoverage(6, 12), canonical_schema=_schema(6, 2, 3, values.dtype, "count"),
        cell_origin=(1, 0))
    outputs = processor.evaluate_inputs({
        "a": primary, "occupied": SignalValue(OCCUPIED, _snapshot(occupied, "1", revision=3),
            coverage=primary.coverage, canonical_schema=_schema(6, 2, 3, occupied.dtype, "1"),
            cell_origin=(1, 0)),
    })
    bright, mean = outputs["bright"], outputs["mean"]
    assert bright.snapshot.block.revision.value == 3
    assert bright.snapshot.block.values.shape == (3, 1, 3)
    assert bright.coverage == MonitorCoverage(3, 3)
    assert bright.canonical_schema is None and bright.cell_origin is None
    assert mean.snapshot.block.values.shape == (1, 1, 3)
    assert mean.run_record is bright.run_record
    assert bright.run_record["parameters"]["expressions"] == rows
    assert bright.run_record["parameters"]["input_view"] == "run"
    assert bright.run_record["parameters"]["inputs"] == {"counts": COUNTS, "occupied": OCCUPIED}
    np.testing.assert_array_equal(bright.snapshot.expanded_validity(), occupied[:, 1:2, :])
    with pytest.raises(RuntimeError, match="lost 'occupied'"):
        processor.evaluate(primary)


def test_descriptor_reads_only_names_and_syntax_without_evaluating_code() -> None:
    rows = _rows(("answer", "raise SystemExit('must not execute while editing')"))
    declared = LOGIC_NODE.outputs_for({"expressions": rows})
    assert [(item.name, item.contract_id) for item in declared] == [("answer", "derive.answer")]
    schema = LOGIC_NODE.authoring_schema
    authored = schema.project_values({"expressions": list(rows)})
    assert authored == {"input_view": "event", "window": 50, "expressions": rows}
    assert schema.draft_values({})["expressions"] == ()
    with pytest.raises(ValueError, match="line 1"):
        schema.project_values({"expressions": _rows(("broken", "result = ("))})
    with pytest.raises(ValueError):
        schema.project_values({"expressions": ({"name": "old", "expression": "a.counts"},)})
