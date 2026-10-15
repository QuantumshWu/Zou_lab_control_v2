"""Derive evaluates variables -- picks, operators, statistics -- on real three-domain Dataset operands."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, DomainSpec, READOUT_EVENT, REPEAT,
    SCALAR_DOMAIN, SCAN_POINT, SITE, ValidityContract, ValueSchema, owned_snapshot_from_arrays,
)
from zlc_runtime import MonitorCoverage, SignalValue

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.nodes.derive import DERIVE_SCHEMA, LOGIC_NODE, DeriveError, DeriveProcessor, Operand
from zlc_atom.nodes.derive.expression import (
    NO_PUBLISHED, checked_variables, compact_variable, describe_schema, error_stage, evaluate,
    input_members, typed_variables,
)

COUNTS = "@logic/occupancy/counts"
OCCUPIED = "@logic/occupancy/occupied"
#: The draft's rows completed column by column, without the validator.
_ROWS = AuthoringSchema(DERIVE_SCHEMA.fields)


def _schema(repeats: int, frames: int, sites: int, dtype, unit: str) -> DatasetSchema:
    site = AxisSpec(AxisId("calibration.site"), "Site", SITE, sites, tuple(range(1, sites + 1)))
    repeat = AxisSpec(AxisId("camera.repeat"), "repeat", REPEAT, repeats)
    frame = AxisSpec(AxisId("camera.frames.frame"), "frame", READOUT_EVENT, frames, tuple(range(frames)))
    return DatasetSchema(
        DomainSpec((repeats,), (repeat,), (tuple(range(repeats)),)),
        DomainSpec((frames,), (frame,), (tuple(range(frames)),)),
        DomainSpec((sites,), (site,)),
        ValueSchema(ValidityContract.components(site.axis_id), np.dtype(dtype), unit),
    )


def _operand(values, unit="count", valid=None):
    values = np.asarray(values)
    return Operand(_schema(*values.shape, values.dtype, unit), values, valid)


def _rows(*variables):
    return _ROWS.project_values({"variables": variables})["variables"]


def _snapshot(values, unit, valid=None, *, revision=0, name=None):
    schema = _schema(*values.shape, values.dtype, unit)
    schema = replace(schema, value_schema=replace(schema.value_schema, name=name))
    return owned_snapshot_from_arrays(schema, values, revision,
                                      validity=np.ones(values.shape, bool) if valid is None else valid)


def _evaluated(rows, members):
    """Every LIVE variable evaluated, and each typed answer equal to it."""
    results = evaluate(rows, members)
    for row, answer in zip(rows, typed_variables(rows, {name: value.schema for name, value in members.items()})):
        assert answer["error"] is None, answer["error"]
        if answer["live"]:
            assert answer["schema"] == results[row["name"]].schema, row["name"]
    return results


def _problem(rows, members):
    """The first problem typing finds, as (variable, stage, detail)."""
    for answer in typed_variables(rows, {name: value.schema for name, value in members.items()}):
        if answer["error"] is not None:
            error = answer["error"]
            return error.variable, error.stage, error.detail
    return None


def _occupancy(repeats=5, sites=6, seed=3):
    """Counts and verdicts of a 3-frame cycle: some counts invalid or not a
    number, and frames the classifier did not judge at some sites."""
    rng = np.random.default_rng(seed)
    counts = (rng.random((repeats, 3, sites)) * 100).astype(np.float32)
    counts[1, 2, 0] = np.nan
    counted = np.ones(counts.shape, bool)
    counted[2, 1, 3] = False
    occupied = rng.random((repeats, 3, sites)) > .5
    judged = np.ones(occupied.shape, bool)
    judged[0, 0, 1] = judged[3, 2, 4] = False
    return {"counts": _operand(counts, "count", counted), "occupied": _operand(occupied, "1", judged)}


def test_the_users_example_keeps_the_counts_of_sites_whose_frames_0_and_2_agree() -> None:
    """same = occupied at frame 0 = occupied at frame 2, internal; kept =
    counts on every frame where same holds, published.  Only kept is a
    signal; the run record keeps both as written."""
    members = _occupancy()
    draft = [
        {"name": "same", "publish": False, "terms": [
            {"of": "occupied", "at": "frame", "labels": ["0"]},
            {"op": "==", "of": "occupied", "at": "frame", "labels": ["2"]}]},
        {"name": "kept", "terms": [{"of": "counts"}, {"op": "where", "of": "same"}]},
    ]
    rows = DERIVE_SCHEMA.project_values({"variables": draft})["variables"]
    assert [compact_variable(row) for row in rows] == [
        {"name": "same", "publish": False, "terms": (
            {"of": "occupied", "at": "frame", "labels": ("0",)},
            {"op": "==", "of": "occupied", "at": "frame", "labels": ("2",)})},
        {"name": "kept", "terms": ({"of": "counts"}, {"op": "where", "of": "same"})}]
    results = _evaluated(rows, members)
    occupied, judged = members["occupied"].values, members["occupied"].valid
    counts, counted = members["counts"].values, members["counts"].valid
    same = results["same"]
    np.testing.assert_array_equal(same.values[:, 0], occupied[:, 0] == occupied[:, 2])
    np.testing.assert_array_equal(same.valid[:, 0], judged[:, 0] & judged[:, 2])
    assert describe_schema(same.schema) == "repeat 5 × Site 6 · bool · unit 1"
    kept = results["kept"]
    assert describe_schema(kept.schema) == "repeat 5 × frame 3 × Site 6 · float32 · unit count"
    np.testing.assert_array_equal(kept.values, counts)
    agree = (same.valid & same.values)[:, :, :]
    np.testing.assert_array_equal(kept.valid, counted & np.isfinite(counts) & agree)
    assert not kept.valid[1, 2, 0], "a count that is not a number is never valid"
    answers = typed_variables(rows, {name: value.schema for name, value in members.items()})
    assert [(a["live"], a["reads"], a["read_by"], a["grouping"]) for a in answers] == [
        (True, (), ("kept",), None), (True, ("same",), (), None)]
    assert answers[1]["terms"][1]["left"] == members["counts"].schema
    assert [item.name for item in LOGIC_NODE.outputs_for({"variables": draft}, {})] == ["kept"]

    processor = DeriveProcessor(variables=rows, primary_output="counts", input_outputs=("counts", "occupied"))
    assert processor.dataset_input_siblings == ("occupied",)
    primary = SignalValue(COUNTS, _snapshot(counts, "count", counted, revision=3, name="counts"),
                          coverage=MonitorCoverage(15, 15))
    inputs = {"bundle": primary, "occupied": SignalValue(
        OCCUPIED, _snapshot(occupied, "1", judged, revision=3), coverage=primary.coverage)}
    outputs = processor.evaluate_inputs(inputs)
    assert tuple(outputs) == ("kept",)
    published = outputs["kept"]
    assert published.snapshot.block.schema.value_schema.name == "kept"
    assert published.snapshot.block.revision.value == 3 and published.coverage == MonitorCoverage(15, 15)
    np.testing.assert_array_equal(published.snapshot.expanded_validity(), kept.valid)
    parameters = processor.describe_run(inputs)["parameters"]
    assert parameters["variables"] == tuple(compact_variable(row) for row in rows)
    assert parameters["inputs"] == {"counts": COUNTS, "occupied": OCCUPIED}
    with pytest.raises(RuntimeError, match="lost 'occupied'"):
        processor.evaluate(primary)


def test_operators_keep_their_precedence_and_repeat_within_one_variable() -> None:
    """Arithmetic binds tighter than one comparison, which binds tighter than
    and/and not, then or, then where; each level reads left to right.  The
    note says the grouping only when it differs from reading left to right."""
    members = _occupancy()
    c = members["counts"].values.astype(np.float64)
    o, judged = members["occupied"].values, members["occupied"].valid

    def at(name, *labels):
        return {"of": name, "at": "frame", "labels": list(labels)}

    rows = _rows(
        {"name": "brightened", "terms": [at("counts", "2"), {"op": "-", **at("counts", "0")},
                                         {"op": ">", "number": 30}, {"op": "and", **at("occupied", "0")}]},
        {"name": "window", "terms": [at("counts", "1"), {"op": ">", "number": 10}, {"op": "and", **at("counts", "1")},
                                     {"op": "<", "number": 50}, {"op": "and", **at("occupied", "1")}]},
        {"name": "same", "publish": False, "terms": [at("occupied", "0"), {"op": "==", **at("occupied", "2")}]},
        {"name": "steady", "terms": [at("counts", "1"), {"op": ">", "number": 30}, {"op": "and", **at("counts", "2")},
                                     {"op": ">", "number": 30}, {"op": "or", "of": "same"}]},
        {"name": "rise", "terms": [at("counts", "2", "1"), {"op": "-", **at("counts", "0")}]},
        {"name": "lost", "terms": [at("occupied", "0"), {"op": "and not", **at("occupied", "2")}],
         "statistic": "sum", "over": ["Site"]},
        {"name": "empty0", "terms": [at("occupied", "0"), {"op": "==", "boolean": "false"}]},
        {"name": "kept_both", "terms": [{"of": "counts"}, {"op": "where", **at("occupied", "0")},
                                        {"op": "and", **at("occupied", "2")}]},
        {"name": "scaled", "terms": [at("counts", "0"), {"op": "+", **at("counts", "1")}, {"op": "*", "number": 2}]},
        {"name": "drift", "terms": [at("counts", "0"), {"op": "-", **at("counts", "1")}, {"op": "-", **at("counts", "2")}]},
        {"name": "either", "terms": [at("counts", "0"), {"op": ">", "number": 1}, {"op": "and", **at("counts", "1")},
                                     {"op": "<", "number": 2}, {"op": "or", **at("occupied", "2")}]},
        {"name": "dark", "terms": [at("counts", "1"), {"op": "where not", **at("occupied", "1")}]},
    )
    results = _evaluated(rows, members)
    answers = {row["name"]: answer for row, answer in zip(rows, typed_variables(
        rows, {name: value.schema for name, value in members.items()}))}
    assert {name: answer["grouping"] for name, answer in answers.items() if answer["grouping"]} == {
        "window": "reads (counts[frame 1] > 10) and (counts[frame 1] < 50) and occupied[frame 1]",
        "steady": "reads ((counts[frame 1] > 30) and (counts[frame 2] > 30)) or same",
        "kept_both": "reads counts where (occupied[frame 0] and occupied[frame 2])",
        "scaled": "reads counts[frame 0] + (counts[frame 1] × 2)",
        "either": "reads ((counts[frame 0] > 1) and (counts[frame 1] < 2)) or occupied[frame 2]",
    }
    # A number takes the unit of the other side of + − and comparisons, and
    # is a plain factor beside × ÷.
    assert answers["brightened"]["terms"][2]["number_unit"] == "count"
    assert answers["scaled"]["terms"][2]["number_unit"] is None
    np.testing.assert_array_equal(results["brightened"].values[:, 0], (c[:, 2] - c[:, 0] > 30) & o[:, 0])
    assert describe_schema(results["brightened"].schema) == "repeat 5 × Site 6 · bool · unit 1"
    np.testing.assert_array_equal(results["window"].values[:, 0], (c[:, 1] > 10) & (c[:, 1] < 50) & o[:, 1])
    same = o[:, 0] == o[:, 2]
    np.testing.assert_array_equal(results["steady"].values[:, 0], ((c[:, 1] > 30) & (c[:, 2] > 30)) | same)
    rise = results["rise"]
    assert describe_schema(rise.schema) == "repeat 5 × frame 2 × Site 6 · float32 · unit count"
    assert rise.schema.point_domain.axes[0].labels() == ("1", "2"), "several labels keep the axis in axis order"
    np.testing.assert_allclose(rise.values, c[:, 1:3] - c[:, 0:1])
    lost = results["lost"]
    assert describe_schema(lost.schema) == "repeat 5 · int64 · unit 1"
    joint = judged[:, 0] & judged[:, 2]
    np.testing.assert_array_equal(lost.values[:, 0, 0], (o[:, 0] & ~o[:, 2] & joint).sum(axis=1))
    np.testing.assert_array_equal(results["empty0"].values[:, 0], ~o[:, 0])
    np.testing.assert_array_equal(results["kept_both"].valid,
                                  members["counts"].valid & np.isfinite(c) & (o[:, 0] & o[:, 2] & joint)[:, None])
    np.testing.assert_allclose(results["scaled"].values[:, 0], c[:, 0] + c[:, 1] * 2, rtol=1e-6)
    np.testing.assert_allclose(results["drift"].values[:, 0], (c[:, 0] - c[:, 1]) - c[:, 2], rtol=1e-6)
    assert answers["drift"]["grouping"] is None, "left to right needs no note"
    np.testing.assert_array_equal(results["either"].values[:, 0], ((c[:, 0] > 1) & (c[:, 1] < 2)) | o[:, 2])
    np.testing.assert_array_equal(results["dark"].valid[:, 0],
                                  members["counts"].valid[:, 1] & judged[:, 1] & ~o[:, 1])


def test_a_pick_takes_labels_of_any_axis_in_any_domain() -> None:
    """``at`` names any primary axis -- Repeat, Point or Cell, any role --
    and ``labels`` its exact labels: one drops the axis, several keep it in
    axis order whatever order they are written in; alternate coordinates
    follow their primary."""
    members = _occupancy(repeats=4)
    c = members["counts"].values
    rows = _rows(
        {"name": "corner", "terms": [{"of": "counts", "at": "Site", "labels": ["3", "1"]}]},
        {"name": "third", "terms": [{"of": "counts", "at": "repeat", "labels": ["2"]}]},
        {"name": "one", "terms": [{"of": "counts", "at": "Site", "labels": ["5"]}]},
    )
    results = _evaluated(rows, members)
    site = results["corner"].schema.cell_domain.axes[0]
    assert site.labels() == ("1", "3")
    np.testing.assert_array_equal(results["corner"].values, c[:, :, [0, 2]])
    assert results["third"].schema.repeat_domain.axes == () and results["third"].shape == (1, 3, 6)
    np.testing.assert_array_equal(results["third"].values[0], c[2])
    assert results["one"].schema.cell_domain == SCALAR_DOMAIN and results["one"].shape == (4, 3, 1)

    # A sparse scan: Point rows mapped to (a, b), time an alternate of b,
    # and a detuning axis whose labels are its floats written exactly.
    a = AxisSpec(AxisId("scan.a"), "a", SCAN_POINT, 2, (10, 20))
    b = AxisSpec(AxisId("scan.b"), "b", SCAN_POINT, 3, (1, 2, 3))
    time = AxisSpec(AxisId("scan.time"), "time", SCAN_POINT, 3, (0.1, 0.2, 0.4), "s", coordinate_of=b.axis_id)
    a_codes, b_codes = (0, 0, 1, 1, 1), (0, 2, 0, 1, 2)
    base = _schema(2, 5, 2, np.dtype(float), "count")
    values = np.arange(20.0).reshape(2, 5, 2)
    scan = Operand(replace(base, point_domain=DomainSpec((5,), (a, b, time), (a_codes, b_codes, b_codes))), values)
    detuning = AxisSpec(AxisId("rf.detuning"), "detuning", SCAN_POINT, 3, (0.1, 6834700000.0, 1e-06), "Hz")
    tuned = Operand(replace(base, point_domain=DomainSpec((3,), (detuning,), ((0, 1, 2),))), values[:, :3])
    assert detuning.labels() == ("0.1", "6834700000", "1e-06")
    members = {"scan": scan, "tuned": tuned}
    rows = _rows(
        {"name": "at_b", "terms": [{"of": "scan", "at": "b", "labels": ["2"]}]},
        {"name": "two_b", "terms": [{"of": "scan", "at": "b", "labels": ["3", "1"]}]},
        {"name": "far", "terms": [{"of": "tuned", "at": "detuning", "labels": ["6834700000"]}]},
    )
    results = _evaluated(rows, members)
    assert [axis.name for axis in results["at_b"].schema.point_domain.axes] == ["a"]
    np.testing.assert_array_equal(results["at_b"].values, values[:, [3], :])
    points = results["two_b"].schema.point_domain
    assert [axis.labels() for axis in points.axes] == [("10", "20"), ("1", "3"), ("0.1", "0.4")]
    assert points.codes(b.axis_id) is points.codes(time.axis_id)
    np.testing.assert_array_equal(results["two_b"].values, values[:, [0, 2, 1, 4], :])
    np.testing.assert_array_equal(results["far"].values, values[:, [1], :])

    # Pick problems, each at its term, typed as evaluated.
    members = _occupancy()
    other = AxisSpec(AxisId("other.frame"), "frame", SITE, 6)
    twin = replace(members["counts"].schema, cell_domain=DomainSpec((6,), (other,)), value_schema=ValueSchema(
        ValidityContract.components(other.axis_id), np.dtype(np.float32), "count"))
    labelled = replace(members["counts"].schema, point_domain=DomainSpec((3,), (replace(
        members["counts"].schema.point_domain.axes[0], coordinate_labels=("first", "rest", "rest")),), ((0, 1, 2),)))
    members.update(twin=Operand(twin, members["counts"].values), labelled=Operand(labelled, members["counts"].values),
                   scan=scan)
    for term, detail in (
        ({"of": "occupied", "at": "frmae", "labels": ["0"]}, "occupied has no axis 'frmae'; its axes: repeat, frame, Site"),
        ({"of": "occupied", "at": "Site", "labels": ["7"]}, "occupied has no Site '7'; Site labels: 1, 2, 3, 4, 5, 6"),
        ({"of": "occupied", "at": "frame", "labels": ["1", "1"]}, "frame '1' is chosen twice"),
        ({"of": "twin", "at": "frame", "labels": ["0"]}, "two axes of twin are named frame; it cannot be picked by name"),
        ({"of": "labelled", "at": "frame", "labels": ["rest"]}, "frame labels repeat; it cannot be picked by label"),
        ({"of": "scan", "at": "time", "labels": ["0.1"]}, "time is another coordinate of b; pick b"),
    ):
        rows = _rows({"name": "x", "terms": [term]})
        assert _problem(rows, members) == ("x", "term 1", detail)
        with pytest.raises(DeriveError, match=f"^x · term 1: {detail}"):
            evaluate(rows, members)
    rows = _rows({"name": "x", "terms": [{"of": "occupied", "at": "frame", "labels": ["3"]}]})
    assert _problem(rows, members) == ("x", "term 1", "occupied has no frame '3'; frame labels: 0, 1, 2")
    with pytest.raises(DeriveError, match="^x · term 1: choose frame values"):
        checked_variables(_rows({"name": "x", "terms": [{"of": "occupied", "at": "frame"}]}))


def test_a_statistic_reduces_the_axes_over_names_in_any_domain() -> None:
    """``over`` names any primary axes -- a shot axis, a frame, sites, a
    whole sparse domain by two names -- and several reduce together,
    dividing once.  Invalid and empty groups stay explicit."""
    original = _schema(2, 5, 2, np.dtype(float), "count")
    a = AxisSpec(AxisId("scan.a"), "a", READOUT_EVENT, 2, (10, 20))
    b = AxisSpec(AxisId("scan.b"), "b", READOUT_EVENT, 3, (1, 2, 3))
    time = AxisSpec(AxisId("scan.time"), "time", READOUT_EVENT, 3, (0.1, 0.2, 0.4), "s", coordinate_of=b.axis_id)
    a_codes, b_codes = (0, 0, 1, 1, 1), (0, 2, 0, 1, 2)
    values = np.arange(20.0).reshape(2, 5, 2)
    valid = np.ones(values.shape, bool)
    valid[0, 0, :] = False
    valid[1, 2:4, 1] = False
    sparse = Operand(replace(original, point_domain=DomainSpec((5,), (a, b), (a_codes, b_codes))), values, valid)
    paired = Operand(replace(original, point_domain=DomainSpec((5,), (a, b, time), (a_codes, b_codes, b_codes))),
                     values, valid)
    members = {"x": sparse, "paired": paired}

    def reduced(of, statistic, *over):
        rows = _rows({"name": "r", "terms": [{"of": of}], "statistic": statistic, "over": list(over)})
        return _evaluated(rows, members)["r"]

    for name, remaining, codes in (("a", b, b_codes), ("b", a, a_codes)):
        result = reduced("x", "mean", name)
        assert result.schema.point_domain.axes == (remaining,)
        order = tuple(dict.fromkeys(codes))
        assert tuple(result.schema.point_domain.codes(remaining.axis_id)) == order
        for row, coordinate in enumerate(order):
            selected = np.asarray(codes) == coordinate
            for shot in range(2):
                for site in range(2):
                    samples = values[shot, selected, site][valid[shot, selected, site]]
                    assert result.valid[shot, row, site] == bool(samples.size)
                    if samples.size:
                        assert result.values[shot, row, site] == pytest.approx(samples.mean())
    # Reducing b reduces its alternate time with it.
    assert reduced("paired", "mean", "b").schema == reduced("x", "mean", "b").schema
    # Shots, sparse Point groups and sites together: one division by the
    # total valid count, never a mean of means.
    whole = reduced("x", "mean", "repeat", "b", "Site")
    assert whole.shape == (1, 2, 1) and whole.schema.repeat_domain.axes == ()
    for index in (0, 1):
        selected = np.asarray(a_codes) == index
        assert whole.values[0, index, 0] == pytest.approx(values[:, selected, :][valid[:, selected, :]].mean())
    assert reduced("x", "std", "repeat", "a", "b", "Site").values.item() == pytest.approx(values[valid].std())
    constant = np.empty((100, 1, 2))
    constant[..., 0], constant[..., 1] = 0.0, 0.4328961258925487
    spread = np.empty((20, 1, 2))
    alternating = 1.0e-3 * np.tile((-1.0, 1.0), 10)
    spread[:, 0, 0], spread[:, 0, 1] = alternating, 1.0e6 + alternating
    members.update(constant=_operand(constant), spread=_operand(spread))
    np.testing.assert_array_equal(reduced("constant", "std", "repeat").values, 0.0)
    np.testing.assert_allclose(reduced("spread", "std", "repeat").values, spread.std(axis=0)[None], rtol=1e-13)

    # Validity: an all-invalid group is invalid, every statistic; booleans
    # read as a plot reads them.
    numbers = np.array([[[10., 20., 30.]], [[40., 50., 60.]]])
    flags = np.array([[[True, False, True]], [[False, True, True]]])
    members.update(n=_operand(numbers, valid=np.array([[[False] * 3], [[True, False, True]]])),
                   f=_operand(flags, "1", np.array([[[False] * 3], [[True, False, True]]])),
                   never=_operand(np.zeros(flags.shape, bool), "1"))
    for statistic, expected in (("sum", 100.), ("mean", 50.), ("min", 40.), ("max", 60.), ("std", 10.), ("count", 2)):
        result = reduced("n", statistic, "Site")
        assert not result.valid[0, 0, 0]
        assert result.valid[1, 0, 0] and result.values[1, 0, 0] == pytest.approx(expected)
    for statistic, expected, dtype in (("count", 2, np.int64), ("sum", 1, np.int64), ("mean", .5, np.float64),
                                       ("any", True, bool), ("all", False, bool)):
        result = reduced("f", statistic, "Site")
        assert not result.valid[0, 0, 0] and result.values.dtype == dtype and result.schema.value_schema.value_unit == "1"
        assert result.valid[1, 0, 0] and result.values[1, 0, 0] == expected
    rows = _rows({"name": "r", "terms": [{"of": "n"}, {"op": "where", "of": "never"}],
                  "statistic": "mean", "over": ["repeat", "Site"]})
    assert not _evaluated(rows, members)["r"].valid.any()
    for of, statistic, over, detail in (
        ("f", "min", ["Site"], "min needs numbers; for booleans use sum, mean, count, any or all"),
        ("n", "any", ["Site"], "any needs booleans"),
        ("n", "mean", ["Sitee"], "the value has no axis 'Sitee'; its axes: repeat, frame, Site"),
        ("n", "mean", ["Site", "Site"], "Site is chosen twice"),
    ):
        rows = _rows({"name": "r", "terms": [{"of": of}], "statistic": statistic, "over": over})
        assert _problem(rows, members) == ("r", "statistic", detail)


def test_problems_are_said_by_variable_and_stage_without_and_with_data() -> None:
    members = _occupancy()

    def no_data(*variables):
        with pytest.raises(ValueError) as refused:
            DERIVE_SCHEMA.project_values({"variables": variables})
        return str(refused.value)

    def typed(*variables):
        return _problem(_rows(*variables), members)

    counts, occupied = {"of": "counts"}, {"of": "occupied"}
    # Shape, read without data.
    assert no_data({"name": "x", "terms": [{"op": "+", **counts}]}) == "x · term 1: the first term takes no operator"
    assert no_data({"name": "x", "terms": [counts, counts]}) == "x · term 2: choose an operator"
    assert no_data({"name": "x", "terms": [counts, {"op": "+", "of": "counts", "number": 2}]}) == (
        "x · term 2: a term reads one thing: an input or a variable, a number, or true or false")
    assert no_data({"name": "x", "terms": [counts, {"op": ">"}]}) == "x · term 2: choose what it reads"
    assert no_data({"name": "x", "terms": [counts, {"op": "+", "boolean": "true"}]}) == (
        "x · term 2: true and false follow = or ≠ only")
    assert no_data({"name": "x", "terms": [{"of": "counts", "labels": ["1"]}]}) == (
        "x · term 1: choose the axis of these labels")
    assert no_data({"name": "x", "terms": [{"number": 2}, {"op": "*", "number": 3}]}) == (
        "x · term 1: a formula needs an input or a variable; numbers and true/false alone have no axes")
    assert no_data({"name": "x", "terms": [counts, {"op": ">", "number": 10}, {"op": "<", "number": 50}]}) == (
        "x · term 3: compare once between and, or and where; join two comparisons with and")
    assert no_data({"name": "x", "terms": [counts], "statistic": "sum"}) == (
        "x · statistic: choose the axes it reduces over")
    assert no_data({"name": "x", "terms": [counts], "over": ["Site"]}) == "x · statistic: choose the statistic"
    # References: itself, and a cycle at the term that closes it.
    assert no_data({"name": "x", "terms": [counts, {"op": "+", "of": "x"}]}) == "x · term 2: x reads itself"
    assert no_data({"name": "kept", "terms": [counts, {"op": "where", "of": "same"}]},
                   {"name": "same", "terms": [occupied, {"op": "==", "of": "kept"}]}) == (
        "same · term 2: same reads kept, which reads same")
    # Names, on every variable; at least one is published.
    assert no_data({"name": "x", "terms": [counts]}, {"name": "x", "publish": False, "terms": [counts]}) == (
        "x · name: another variable is already named x")
    assert no_data({"name": "2x", "terms": [counts]}) == (
        "2x · name: a name uses letters, digits and _, and does not start with a digit")
    assert no_data({"name": "x", "publish": False, "terms": [counts]}) == NO_PUBLISHED
    # Old drafts are refused as unknown fields, wherever they stand.
    assert no_data() == "Variables must have at least one row"
    with pytest.raises(ValueError, match=r"unknown authoring fields: \['outputs'\]"):
        DERIVE_SCHEMA.project_values({"outputs": [{"name": "x", "value": "counts"}]})
    with pytest.raises(ValueError, match=r"^Variables row 1: unknown authoring fields: \['code'\]"):
        DERIVE_SCHEMA.project_values({"variables": [{"name": "x", "code": "a.counts"}]})
    with pytest.raises(ValueError, match=r"^Variables row 1: Terms row 1: unknown authoring fields: \['with'\]"):
        DERIVE_SCHEMA.project_values({"variables": [{"name": "x", "terms": [{"of": "counts", "with": "2"}]}]})
    with pytest.raises(TypeError, match="^Variables row 1: Terms row 1: Labels must be a list of texts"):
        DERIVE_SCHEMA.project_values({"variables": [{"name": "x", "terms": [
            {"of": "counts", "at": "frame", "labels": "1"}]}]})

    # Types, axes and units, typed as they would evaluate.
    def at(name, label):
        return {"of": name, "at": "frame", "labels": [label]}

    assert typed({"name": "x", "terms": [counts, {"op": "+", "of": "smae"}]}) == ("x", "term 2", (
        "no input or variable named 'smae'; inputs: counts, occupied; variables: x"))
    assert typed({"name": "counts", "terms": [occupied]}) == ("counts", "name", "counts is an input's name; choose another")
    assert typed({"name": "x", "terms": [counts, {"op": "==", **occupied}]}) == ("x", "term 2", (
        "booleans do not mix with numbers; compare booleans with = or ≠, or count them with a statistic"))
    assert typed({"name": "x", "terms": [occupied, {"op": ">", **occupied}]}) == ("x", "term 2",
                                                                                  "booleans compare only with = and ≠")
    assert typed({"name": "x", "terms": [occupied, {"op": "and", **counts}]}) == ("x", "term 2",
                                                                                  "and needs booleans; counts is a number")
    assert typed({"name": "x", "terms": [counts, {"op": "where", **counts}]}) == ("x", "term 2",
                                                                                  "where needs a boolean mask; counts is a number")
    assert typed({"name": "x", "terms": [at("counts", "1"), {"op": "where", **occupied}]}) == ("x", "term 2", (
        "the mask has frame, which counts[frame 1] does not; pick one frame value of the mask"))
    assert typed({"name": "x", "terms": [{"of": "counts", "at": "frame", "labels": ["1", "2"]},
                                         {"op": "-", "of": "counts", "at": "frame", "labels": ["0", "1"]}]}) == (
        "x", "term 2", "frame differs between the two sides (1, 2 vs 0, 1); pick the same frame values on both "
                       "sides, or one value on one side")
    assert typed({"name": "x", "terms": [{"of": "counts", "at": "Site", "labels": ["1"]},
                                         {"op": "-", "of": "counts", "at": "frame", "labels": ["0"]}]}) == (
        "x", "term 2", "operand axes differ (repeat 5 × frame 3 vs repeat 5 × Site 6); one side's axes must all "
                       "be on the other")
    # A reader of a variable with a problem waits for it rather than repeat it.
    answers = typed_variables(_rows({"name": "bad", "terms": [counts, {"op": "and", **occupied}]},
                                    {"name": "after", "terms": [{"of": "bad"}]}),
                              {name: value.schema for name, value in members.items()})
    assert str(answers[0]["error"]) == "bad · term 2: and needs booleans; counts is a number"
    assert (answers[1]["error"], answers[1]["waits_for"], answers[1]["schema"]) == (None, "bad", None)
    assert error_stage("failed: bad · term 2: and needs booleans; counts is a number") == (
        "bad", "term 2", "and needs booleans; counts is a number")
    assert error_stage("failed: x · statistic: any needs booleans") == ("x", "statistic", "any needs booleans")


def test_an_unpublished_helper_nothing_published_reads_is_parked_and_never_blocks() -> None:
    """LIVE = the published variables and everything they read.  A parked
    helper's formula problems block nothing -- not the draft, admission,
    the build's planning or the evaluation -- but its name is checked, and
    publishing it or reading it from a published variable makes it block."""
    members = _occupancy()
    idea = {"name": "idea", "publish": False, "terms": [{"of": "counts"}, {"op": ">"}]}
    total = {"name": "total", "terms": [{"of": "counts"}], "statistic": "sum", "over": ["Site"]}
    rows = DERIVE_SCHEMA.project_values({"variables": [idea, total]})["variables"]
    answers = typed_variables(rows, {name: value.schema for name, value in members.items()})
    assert (answers[0]["live"], str(answers[0]["error"])) == (False, "idea · term 2: choose what it reads")
    assert input_members(rows, ("counts", "occupied", "frame_judged")) == ("counts",)
    assert tuple(evaluate(rows, members)) == ("total",)
    publication = SimpleNamespace(signals={
        COUNTS: SignalValue(COUNTS, _snapshot(members["counts"].values, "count"), coverage=MonitorCoverage(15, 15)),
        OCCUPIED: SignalValue(OCCUPIED, _snapshot(members["occupied"].values, "1"), coverage=MonitorCoverage(15, 15)),
    })
    assert LOGIC_NODE.admit_source({"variables": rows}, publication) == ()
    # A parked variable's member-name is still an identity problem.
    named = DERIVE_SCHEMA.project_values({"variables": [dict(idea, name="occupied"), total]})
    assert LOGIC_NODE.admit_source(named, publication) == (
        "occupied · name: occupied is an input's name; choose another",)
    with pytest.raises(DeriveError, match="occupied · name: occupied is an input's name"):
        input_members(named["variables"], ("counts", "occupied"))
    for variables in ([dict(idea, publish=True), total],
                      [idea, dict(total, terms=[{"of": "counts"}, {"op": "where", "of": "idea"}])]):
        with pytest.raises(DeriveError, match="^idea · term 2: choose what it reads"):
            DERIVE_SCHEMA.project_values({"variables": variables})
    # Live, a helper's typing problem is an admission problem; the reader waits.
    typo = [dict(idea, terms=[{"of": "counts", "at": "frame", "labels": ["7"]}]),
            dict(total, terms=[{"of": "idea"}])]
    assert LOGIC_NODE.admit_source(DERIVE_SCHEMA.project_values({"variables": typo}), publication) == (
        "idea · term 1: counts has no frame '7'; frame labels: 0, 1, 2",)
    with pytest.raises(DeriveError, match="^total · term 1: no input or variable named 'cuonts'"):
        input_members(_rows(dict(total, terms=[{"of": "cuonts"}])), ("counts", "occupied"))


def test_the_descriptor_declares_published_variables_before_data_and_absent_picks_fail_the_run() -> None:
    """Declarations come from names alone, in card order, for published
    variables only; a formula being edited keeps them, a name problem
    withdraws them.  A pick of a scan value this publication does not carry
    fails the run in the variable's words."""

    def declared(*variables):
        return [(item.name, item.contract_id) for item in LOGIC_NODE.outputs_for({"variables": list(variables)}, {})]

    helper = {"name": "helper", "publish": False, "terms": [{"of": "counts"}]}
    assert declared({"name": "b", "terms": [{"of": "helper"}]}, helper, {"name": "a", "terms": [{"of": "x"}]}) == [
        ("b", "derive.b"), ("a", "derive.a")]
    assert declared({"name": "b", "terms": [{"of": "counts"}, {"op": "+"}]}) == [("b", "derive.b")]
    assert declared({"name": "b", "terms": []}, {"name": "b", "terms": []}) == []
    assert declared({"name": "b", "code": "a.counts"}) == []
    assert LOGIC_NODE.outputs_for({"outputs": [{"name": "b", "value": "counts"}]}, {}) == ()
    assert LOGIC_NODE.input_specs[0].name == "bundle"
    assert DERIVE_SCHEMA.draft_values({})["variables"] == ()

    # A live event of a scan carries one detuning; the run's whole geometry
    # (what typing reads) carries every one.
    detuning = AxisSpec(AxisId("rf.detuning"), "detuning", SCAN_POINT, 3, (1, 2, 3), "Hz")
    whole = replace(_schema(1, 1, 2, np.dtype(float), "count"),
                    point_domain=DomainSpec((3,), (detuning,), ((0, 1, 2),)))
    event = replace(whole, point_domain=DomainSpec((1,), (replace(detuning, size=1, coordinates=(2,)),), ((0,),)))
    rows = _rows({"name": "far", "terms": [{"of": "counts", "at": "detuning", "labels": ["3"]}]})
    assert typed_variables(rows, {"counts": whole})[0]["error"] is None
    with pytest.raises(DeriveError) as absent:
        evaluate(rows, {"counts": Operand(event, np.zeros((1, 1, 2)))})
    assert str(absent.value) == "far · term 1: this publication carries no detuning '3'; detuning labels here: 2"
    assert error_stage(f"failed: {absent.value}")[:2] == ("far", "term 1")
