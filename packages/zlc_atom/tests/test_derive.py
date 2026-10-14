"""Derive evaluates structured and code outputs on real three-domain Dataset operands."""

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
    LOGIC_NODE, DeriveError, DeriveProcessor, Operand,
)
from zlc_atom.nodes.derive.expression import compiled, execute, typed_outputs

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


def _snapshot(values, unit, valid=None, *, revision=0, name=None):
    schema = _schema(*values.shape, values.dtype, unit)
    schema = replace(schema, value_schema=replace(schema.value_schema, name=name))
    return owned_snapshot_from_arrays(schema, values, revision,
        validity=np.ones(values.shape, bool) if valid is None else valid)


def _operand(values, unit="count", valid=None):
    snapshot = _snapshot(np.asarray(values), unit, valid)
    return Operand(snapshot.block.schema, snapshot.block.values, snapshot.expanded_validity())


def _rows(*signals):
    return tuple({"name": name, "code": code} for name, code in signals)


def _one(code, outputs):
    return execute(compiled(_rows(("answer", code))), outputs)["answer"]


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
    assert tuple(kept.schema.point_domain.axes[0].coordinate_values()) == (1,)
    assert kept.schema.point_domain.axes[0].coordinate_labels == ("frame 1",)
    np.testing.assert_array_equal(scalar.values, kept.values)
    np.testing.assert_array_equal(source.sel(frame=1).values, scalar.values)
    assert source.sel(frame=[1]).schema == kept.schema
    # The pinned operand lacks only the frame axis: it repeats along it.
    assert (scalar + kept).schema == kept.schema
    with pytest.raises(ValueError, match="operand axes differ"):
        _ = source.isel(frame=[0]) + kept
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
        assert tuple(frame.coordinate_values()) == tuple(indexes)
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
    constant_values = np.empty((100, 1, 2), dtype=np.float64)
    constant_values[..., 0] = 0.0
    constant_values[..., 1] = 0.4328961258925487
    constant_std = _operand(constant_values).std("cycle")
    np.testing.assert_array_equal(constant_std.values, 0.0)
    local_spread = np.empty((20, 1, 2), dtype=np.float64)
    alternating = 1.0e-3 * np.tile((-1.0, 1.0), 10)
    local_spread[:, 0, 0] = alternating
    local_spread[:, 0, 1] = 1.0e6 + alternating
    local_std = _operand(local_spread).std("cycle")
    np.testing.assert_allclose(
        local_std.values,
        local_spread.std(axis=0)[None, ...],
        rtol=1.0e-13,
    )
    selected = source.isel(a=1)
    assert selected.shape == (2, 3, 2)
    assert tuple(axis.name for axis in selected.schema.point_domain.axes) == ("b",)
    np.testing.assert_array_equal(selected.values, values[:, 2:, :])
    reordered = source.isel(b=[2, 0])
    assert tuple(reordered.schema.point_domain.axes[1].coordinate_values()) == (3, 1)
    assert tuple(reordered.schema.point_domain.codes(b.axis_id)) == (0, 0, 1, 1)
    np.testing.assert_array_equal(reordered.values, values[:, [1, 4, 0, 2], :])
    np.testing.assert_array_equal(reordered.valid, valid[:, [1, 4, 0, 2], :])
    time = AxisSpec(
        AxisId("scan.time"), "time", READOUT_EVENT, 3, (0.1, 0.2, 0.4),
        "s", coordinate_of=b.axis_id,
    )
    paired = Operand(replace(schema, point_domain=DomainSpec(
        (5,), (a, b, time), (a_codes, b_codes, b_codes),
    )), values, valid)
    for coordinate in ("b", "time"):
        result = paired.mean(coordinate)
        assert result.schema == source.mean("b").schema
        np.testing.assert_array_equal(result.values, source.mean("b").values)
        selected = paired.isel({coordinate: 1})
        assert selected.schema == source.isel(b=1).schema
        np.testing.assert_array_equal(selected.values, source.isel(b=1).values)
        selected = paired.isel({coordinate: [2, 0]})
        points = selected.schema.point_domain
        assert tuple(points.axis(b.axis_id).coordinate_values()) == (3, 1)
        assert tuple(points.axis(time.axis_id).coordinate_values()) == (0.4, 0.1)
        assert points.codes(b.axis_id) is points.codes(time.axis_id)
        np.testing.assert_array_equal(selected.values, reordered.values)


def test_where_and_reductions_keep_invalid_and_empty_groups_explicit() -> None:
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
    # Booleans read as a plot reads them: sum is the number true, mean the
    # fraction true, count the valid samples -- of any dtype.
    for operation, expected in (("count", 2), ("sum", 1), ("mean", .5), ("any", True), ("all", False)):
        result = getattr(boolean, operation)("site")
        assert not result.valid[0, 0, 0]
        assert result.valid[1, 0, 0] and result.values[1, 0, 0] == expected
        assert result.unit == "1"
    for operation in ("min", "max", "std"):
        with pytest.raises(TypeError, match="needs numbers"):
            getattr(boolean, operation)("site")
    with pytest.raises(TypeError, match="do not mix with numbers"):
        _ = source == boolean
    with pytest.raises(TypeError, match="do not mix with numbers"):
        _ = boolean * 2
    empty = mask.with_values(np.zeros(mask.shape, dtype=bool))
    for operation in ("sum", "mean", "min", "max", "std", "count"):
        assert not getattr(source, operation)(("cycle", "site"), where=empty).valid.any()
    for operation in ("all", "any", "count", "sum", "mean"):
        assert not getattr(mask, operation)(("cycle", "site"), where=empty).valid.any()
    counted = source.count("cycle")
    np.testing.assert_array_equal(counted.values, [[[1, 0, 1]]])


def test_python_locals_numpy_and_previous_outputs_do_not_create_a_second_schema() -> None:
    source = _operand(np.arange(6.0).reshape(1, 2, 3))
    rows = _rows(
        ("scaled", "x = a.counts\nlocal = np.clip(x.values, 1, 4)\nresult = x.with_values(local)"),
        ("difference", "scaled.isel(frame=1) - scaled.isel(frame=0)"),
        ("condition", "result = (difference > 0) & (difference < 5)"),
    )
    programs = compiled(rows)
    assert tuple(row["name"] for row, _ in programs) == (
        "scaled",
        "difference",
        "condition",
    )
    results = execute(programs, {"counts": source})
    assert tuple(results) == tuple(row["name"] for row, _ in programs)
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
        with pytest.raises(DeriveError, match=f"answer · code: .*{message}"):
            _one(code, outputs)
    for rows in (
        (), ({"name": "old", "expression": "a.counts"},),
        _rows(("result", "a.counts")), _rows(("same", "a.counts"), ("same", "a.counts")),
        ({"name": "mixed", "value": "counts", "code": "a.counts"},),
    ):
        with pytest.raises(DeriveError):
            execute(compiled(rows), outputs)


def test_structured_outputs_type_exactly_as_they_evaluate() -> None:
    """Each row reads select -> mask -> combine -> reduce; the schemas alone
    give the same answer the data does, and refusals name row and stage."""
    values = np.arange(24.0).reshape(2, 3, 4)
    occupied = values % 3 != 0
    outputs = {"counts": _operand(values), "occupied": _operand(occupied, "1")}
    schemas = {name: value.schema for name, value in outputs.items()}
    rows = (
        {"name": "bright", "value": "counts", "at": "frame 1", "where": "occupied", "where_at": "frame 1"},
        {"name": "loaded", "value": "occupied", "at": "frame 0", "statistic": "sum", "over": "site"},
        {"name": "fraction", "value": "occupied", "at": "frame 0", "statistic": "mean", "over": "site"},
        {"name": "level", "value": "bright", "statistic": "mean", "over": "site"},
        # A per-shot level repeats along the sites it lacks.
        {"name": "norm", "value": "counts", "at": "frame 1", "op": "/", "with": "level"},
        {"name": "dark", "value": "counts", "at": "frame 2", "where": "occupied", "where_at": "frame 2",
         "where_is": False},
        {"name": "rise", "value": "counts", "at": "frame 2", "op": "-", "with": "counts", "with_at": "frame 0"},
        {"name": "high", "value": "counts", "at": "frame 1", "op": ">", "with": "15"},
        {"name": "agree", "value": "occupied", "at": "frame 0", "op": "==", "with": "occupied",
         "with_at": "frame 2"},
        {"name": "coded", "code": "result = norm * 2"},
        {"name": "after", "value": "coded", "statistic": "max", "over": "site"},
    )
    answers = typed_outputs(rows, schemas)
    results = execute(compiled(rows), outputs)
    for row, answer in zip(rows, answers):
        assert answer["error"] is None, answer["error"]
        assert answer["schema"] is None or answer["schema"] == results[row["name"]].schema, row["name"]
    assert answers[-2]["schema"] is answers[-1]["schema"] is None  # code: known after Start
    np.testing.assert_array_equal(results["bright"].valid, occupied[:, 1:2, :])
    np.testing.assert_array_equal(results["loaded"].values[:, 0, 0], occupied[:, 0, :].sum(axis=1))
    np.testing.assert_allclose(results["fraction"].values[:, 0, 0], occupied[:, 0, :].mean(axis=1))
    level = np.array([values[r, 1, occupied[r, 1]].mean() for r in range(2)])
    np.testing.assert_allclose(results["norm"].values[:, 0, :], values[:, 1, :] / level[:, None])
    np.testing.assert_array_equal(results["dark"].valid, ~occupied[:, 2:3, :])
    np.testing.assert_array_equal(results["rise"].values, values[:, 2:3, :] - values[:, 0:1, :])
    assert results["high"].dtype == bool and results["agree"].dtype == bool
    assert results["loaded"].shape == (2, 1, 1) and results["loaded"].dtype == np.int64
    for row, stage, message in (
        # A pin is the frame's label, as a plot shows it.
        ({"name": "x", "value": "counts", "at": "1"}, "value",
         "no frame '1'; frame labels: frame 0, frame 1, frame 2"),
        ({"name": "x", "value": "counts", "at": "frame 1", "where": "occupied"}, "where",
         "frame is only on the mask"),
        ({"name": "x", "value": "counts", "at": "frame 1", "op": "==", "with": "occupied", "with_at": "frame 1"},
         "combine", "do not mix with numbers"),
        ({"name": "x", "value": "counts", "at": "frame 1", "statistic": "mean", "over": "Site"},
         "statistic", "no axis 'Site'; axes: cycle, site"),
        ({"name": "x", "value": "counts", "statistic": "mean", "over": "cycle"},
         "statistic", "belong in a plot panel"),
        ({"name": "x", "value": "missing"}, "value", "no input or earlier output named 'missing'"),
        ({"name": "x", "value": "later"}, "value", "defined below"),
        ({"name": "x", "value": "counts", "op": "-"}, "combine", "choose what to combine with"),
        ({"name": "counts", "value": "occupied"}, "name", "an input's name"),
    ):
        (answer, *_rest) = typed_outputs((row, {"name": "later", "value": "counts"}), schemas)
        error = answer["error"]
        assert (error.output, error.stage) == (row["name"], stage) and message in error.detail, error


def test_processor_publishes_current_estimates_and_the_exact_program_provenance() -> None:
    values = np.arange(18.0).reshape(3, 2, 3)
    occupied = values % 2 == 0
    rows = ({"name": "bright", "value": "counts", "at": "frame 1", "where": "occupied", "where_at": "frame 1"},
            {"name": "total", "code": "bright.sum('site')"})
    processor = DeriveProcessor(outputs=rows, primary_output="counts",
        input_outputs=("counts", "occupied"))
    assert processor.dataset_input_siblings == ("occupied",)
    counts = _snapshot(values, "count", revision=3, name="counts")
    primary = SignalValue(COUNTS, counts, coverage=MonitorCoverage(6, 6))
    inputs = {"a": primary, "occupied": SignalValue(OCCUPIED, _snapshot(occupied, "1", revision=3),
                                                    coverage=primary.coverage)}
    outputs = processor.evaluate_inputs(inputs)
    bright, total = outputs["bright"], outputs["total"]
    assert bright.snapshot.block.schema.value_schema.name == "bright"
    assert total.snapshot.block.schema.value_schema.name == "total"
    assert primary.schema.value_schema.name == "counts"
    assert bright.snapshot.block.revision.value == 3
    assert bright.snapshot.block.values.shape == (3, 1, 3)
    assert bright.coverage == MonitorCoverage(3, 3)
    assert bright.canonical_schema is None and bright.cell_origin is None
    assert total.snapshot.block.values.shape == (3, 1, 1)
    # The provenance the runtime files for this run: the exact outputs, as
    # written, and which signal fed each input name.
    parameters = processor.describe_run(inputs)["parameters"]
    assert parameters["outputs"] == rows
    assert parameters["inputs"] == {"counts": COUNTS, "occupied": OCCUPIED}
    np.testing.assert_array_equal(bright.snapshot.expanded_validity(), occupied[:, 1:2, :])
    with pytest.raises(RuntimeError, match="lost 'occupied'"):
        processor.evaluate(primary)


def test_descriptor_reads_only_names_and_syntax_without_evaluating_code() -> None:
    rows = _rows(("answer", "raise SystemExit('must not execute while editing')"))
    declared = LOGIC_NODE.outputs_for({"outputs": rows}, {})
    assert [(item.name, item.contract_id) for item in declared] == [("answer", "derive.answer")]
    schema = LOGIC_NODE.authoring_schema
    authored = schema.project_values({"outputs": list(rows)})
    assert authored["outputs"][0]["code"] == rows[0]["code"] and authored["outputs"][0]["value"] == ""
    assert schema.draft_values({})["outputs"] == ()
    with pytest.raises(ValueError, match="broken · code: line 1"):
        schema.project_values({"outputs": _rows(("broken", "result = ("))})
    for row in ({"name": "old", "expression": "a.counts"}, {"name": "x", "value": "counts", "op": "%"}):
        with pytest.raises(ValueError):
            schema.project_values({"outputs": (row,)})


def test_the_editor_lists_what_the_data_offers_and_commands_are_buttons() -> None:
    """The frames come from the data, so ``at`` is a list with All first;
    adding an output, opening or closing a step and removing a row are
    commands.  The draft reads as the same rows either way."""
    from PyQt5 import QtCore, QtTest, QtWidgets
    from zlc_ui.fluent import FluentButton

    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["zlc-derive-editor-test"])
    editor = LOGIC_NODE.ui_contributions[0]()
    drafts = []
    editor.draft_changed.connect(lambda patch: drafts.append(patch["values"]["outputs"]))
    bundle = (("counts", _schema(2, 3, 4, np.dtype(float), "count"), None),
              ("occupied", _schema(2, 3, 4, np.dtype(bool), "1"), None))

    def project(rows):
        editor.update_projection({"form_values": {"outputs": rows}, "input_bundle": bundle})
        app.processEvents()
        return editor._cards

    def click(button):
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        app.processEvents()
        return drafts[-1]

    try:
        project(())
        (row,) = click(editor.add_output)
        assert row == {"name": "output_1", "value": "counts"}
        (card,) = project((row,))
        assert [card.at.itemText(index) for index in range(card.at.count())] == [
            "All", "frame 0", "frame 1", "frame 2"]
        assert card.at.currentIndex() == 0, "no pin reads every frame"
        card.at.setCurrentIndex(2)
        card.at.activated.emit(2)
        (row,) = drafts[-1]
        assert row["at"] == "frame 1"
        project((row,))
        (row,) = click(card.add["where"])
        assert (row["where"], row["where_at"]) == ("occupied", "frame 1"), "a verdict at the value's own frame"
        project((row,))
        assert card.where_at.currentText() == "frame 1" and not card.lines["where"].isHidden()
        (close,) = card.lines["where"].findChildren(FluentButton)
        assert click(close) == ({"name": "output_1", "value": "counts", "at": "frame 1"},)
        remove = card.remove
        assert remove.text() == "×" and remove.role() == "destructive"
        assert click(remove) == () and editor._cards == []
        assert click(editor.add_code) == ({"name": "output_1"},)
        (code,) = project(drafts[-1])
        assert code.kind == "code" and not code.insert.isHidden()
    finally:
        editor.close()
        editor.deleteLater()
        app.processEvents()


def test_a_typing_error_is_said_once_under_its_line_where_the_result_stood() -> None:
    """A row the schemas cannot type says why in red under the line it is
    about, and names that problem -- in the draft's words -- to the page,
    whose header counts it instead of saying it again.  The error stands
    where the row's result stood: the same gap under the card's last line,
    so the card keeps its height and nothing below it moves."""
    from PyQt5 import QtCore, QtWidgets

    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["zlc-derive-editor-test"])
    # Hosted as the Edit page hosts it: at its own height, the slack below.
    host = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(host)
    editor = LOGIC_NODE.ui_contributions[0]()
    column.addWidget(editor)
    column.addStretch(1)
    schemas = {"counts": _schema(2, 3, 4, np.dtype(float), "count"),
               "occupied": _schema(2, 3, 4, np.dtype(bool), "1")}
    bundle = tuple((name, schema, None) for name, schema in schemas.items())
    typed = {"name": "bright", "value": "counts", "at": "frame 1", "where": "occupied", "where_at": "frame 1"}
    unpinned = dict(typed, where_at="")

    def project(row):
        editor.update_projection({"form_values": {"outputs": (row,)}, "input_bundle": bundle})
        (card,) = editor._cards
        for _ in range(3):
            app.processEvents()
        return card

    def top(widget, card):
        return widget.mapTo(card, QtCore.QPoint()).y()

    host.resize(1400, 900)
    host.show()
    try:
        card = project(typed)
        assert editor.shown_problems() == {}
        assert card.preview.isVisible()
        result_y, height = top(card.preview, card), card.height()
        card = project(unpinned)
        (answer,) = typed_outputs((unpinned,), schemas)
        assert answer["error"] is not None and answer["error"].stage == "where"
        assert editor.shown_problems() == {str(answer["error"]): "Outputs"}
        error = card.errors["where"]
        assert error.isVisible() and not card.preview.isVisible()
        assert (top(error, card), card.height()) == (result_y, height)
    finally:
        host.close()
        host.deleteLater()
        app.processEvents()
