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
from zlc_atom.nodes._framework.frames import FRAME_AXIS_NAME
from zlc_atom.nodes.derive import DERIVE_SCHEMA, LOGIC_NODE, DeriveError, DeriveProcessor, Operand
from zlc_atom.nodes.derive.expression import (
    NO_PUBLISHED, changing_axes, checked_variables, compact_variable, describe_schema, error_stage, evaluate,
    frames_kept, input_members, pick_values, term_text, typed_variables,
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
    and ``labels`` its exact labels, or ``first`` and ``last``, its ends as
    long as it is: one drops the axis, several keep it in axis order
    whatever order they are written in, a coordinate two of them name
    once; alternate coordinates follow their primary."""
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
    # How many labels are written, not how many coordinates they reach,
    # decides whether the axis stays: on a one-long axis First and Last
    # are one coordinate, kept once.
    rows = _rows(
        {"name": "ends", "terms": [{"of": "counts", "at": "frame", "labels": ["last", "first"]}]},
        {"name": "final", "terms": [{"of": "counts", "at": "frame", "labels": ["last"]}]},
        {"name": "edge", "terms": [{"of": "counts", "at": "Site", "labels": ["last", "2"]}]},
        {"name": "zero", "terms": [{"of": "counts", "at": "frame", "labels": ["first", "0"]}]},
        {"name": "lone", "terms": [{"of": "zero", "at": "frame", "labels": ["first", "last"]}]},
    )
    results = _evaluated(rows, members)
    assert results["ends"].schema.point_domain.axes[0].labels() == ("0", "2")
    np.testing.assert_array_equal(results["ends"].values, c[:, [0, 2], :])
    assert results["final"].schema.point_domain.axes == ()
    np.testing.assert_array_equal(results["final"].values, c[:, [2], :])
    assert results["edge"].schema.cell_domain.axes[0].labels() == ("2", "6")
    np.testing.assert_array_equal(results["edge"].values, c[:, :, [1, 5]])
    for name in ("zero", "lone"):
        assert results[name].schema.point_domain.axes[0].labels() == ("0",), name
        np.testing.assert_array_equal(results[name].values, c[:, [0], :])
    assert term_text(rows[0]["terms"][0]) == "counts[frame last, first]"

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

    # A domain with two or more axes is picked by its rows, each written as
    # its axes' names and labels: one row drops the domain's axes, several
    # keep exactly those rows (in row order), axes cut to what they use.
    assert pick_values(scan.schema, "Point") == ("a 10 · b 1", "a 10 · b 3", "a 20 · b 1", "a 20 · b 2", "a 20 · b 3")
    rows = _rows(
        {"name": "row", "terms": [{"of": "scan", "at": "Point", "labels": ["a 20 · b 2"]}]},
        {"name": "rows", "terms": [{"of": "scan", "at": "Point", "labels": ["a 20 · b 3", "a 10 · b 3"]}]},
    )
    results = _evaluated(rows, {"scan": scan})
    assert results["row"].schema.point_domain.axes == () and results["row"].shape == (2, 1, 2)
    np.testing.assert_array_equal(results["row"].values, values[:, [3], :])
    points = results["rows"].schema.point_domain
    assert [axis.labels() for axis in points.axes] == [("10", "20"), ("3",), ("0.4",)]
    np.testing.assert_array_equal(results["rows"].values, values[:, [1, 4], :])
    assert term_text(rows[1]["terms"][0]) == "scan[a 20 · b 3, a 10 · b 3]"
    # A domain's ends are its first and last rows.
    results = _evaluated(_rows({"name": "ends", "terms": [{"of": "scan", "at": "Point", "labels": ["first", "last"]}]},
                               {"name": "last", "terms": [{"of": "scan", "at": "Point", "labels": ["last"]}]}),
                         {"scan": scan})
    assert [axis.labels() for axis in results["ends"].schema.point_domain.axes] == [
        ("10", "20"), ("1", "3"), ("0.1", "0.4")]
    np.testing.assert_array_equal(results["ends"].values, values[:, [0, 4], :])
    assert results["last"].schema.point_domain.axes == ()
    np.testing.assert_array_equal(results["last"].values, values[:, [4], :])
    # A dense Cell domain keeps a grid: one cell, or rows that fill one.
    y = AxisSpec(AxisId("camera.y"), "y", SITE, 3, (0, 1, 2))
    x = AxisSpec(AxisId("camera.x"), "x", SITE, 2, (0, 1))
    image = Operand(replace(base, cell_domain=DomainSpec((3, 2), (y, x)), value_schema=ValueSchema(
        ValidityContract.components(y.axis_id, x.axis_id), np.dtype(float), "count")),
        np.arange(60.0).reshape(2, 5, 3, 2))
    rows = _rows(
        {"name": "pixel", "terms": [{"of": "image", "at": "Cell", "labels": ["y 1 · x 1"]}]},
        {"name": "corner", "terms": [{"of": "image", "at": "Cell", "labels": [
            "y 2 · x 1", "y 1 · x 0", "y 1 · x 1", "y 2 · x 0"]}]},
    )
    results = _evaluated(rows, {"image": image})
    assert results["pixel"].schema.cell_domain == SCALAR_DOMAIN
    np.testing.assert_array_equal(results["pixel"].values[..., 0], image.values[:, :, 1, 1])
    assert describe_schema(results["corner"].schema) == "repeat 2 × frame 5 × y 2 × x 2 · float64 · unit count"
    np.testing.assert_array_equal(results["corner"].values, image.values[:, :, 1:3, :])

    # Pick problems, each at its term, typed as evaluated.
    members = _occupancy()
    other = AxisSpec(AxisId("other.frame"), "frame", SITE, 6)
    twin = replace(members["counts"].schema, cell_domain=DomainSpec((6,), (other,)), value_schema=ValueSchema(
        ValidityContract.components(other.axis_id), np.dtype(np.float32), "count"))
    labelled = replace(members["counts"].schema, point_domain=DomainSpec((3,), (replace(
        members["counts"].schema.point_domain.axes[0], coordinate_labels=("first", "rest", "rest")),), ((0, 1, 2),)))
    framed = replace(members["counts"].schema, point_domain=DomainSpec((3,), (replace(
        members["counts"].schema.point_domain.axes[0], name="Point"),), ((0, 1, 2),)))
    ended = replace(members["counts"].schema, point_domain=DomainSpec((3,), (replace(
        members["counts"].schema.point_domain.axes[0], coordinate_labels=("start", "first", "end")),), ((0, 1, 2),)))
    members.update(twin=Operand(twin, members["counts"].values), labelled=Operand(labelled, members["counts"].values),
                   scan=scan, image=image, framed=Operand(framed, members["counts"].values),
                   ended=Operand(ended, members["counts"].values))
    for term, detail in (
        ({"of": "occupied", "at": "frmae", "labels": ["0"]}, "occupied has no axis 'frmae'; its axes: repeat, frame, Site"),
        ({"of": "occupied", "at": "Site", "labels": ["7"]}, "occupied has no Site '7'; Site labels: 1, 2, 3, 4, 5, 6"),
        ({"of": "occupied", "at": "frame", "labels": ["1", "1"]}, "frame '1' is chosen twice"),
        ({"of": "twin", "at": "frame", "labels": ["0"]}, "two axes of twin are named frame; it cannot be picked by name"),
        ({"of": "labelled", "at": "frame", "labels": ["rest"]}, "frame labels repeat; it cannot be picked by label"),
        ({"of": "scan", "at": "time", "labels": ["0.1"]}, "time is another coordinate of b; pick b"),
        ({"of": "occupied", "at": "Cell", "labels": ["Site 1"]}, "Cell has one axis, Site; pick Site"),
        ({"of": "image", "at": "Cell", "labels": ["y 0 · x 0", "y 1 · x 1"]},
         "these Cell values do not fill a grid of y × x; pick one, or pick each axis in a variable of its own"),
        ({"of": "framed", "at": "Point", "labels": ["1"]},
         "Point names both a domain and an axis of framed; it cannot be picked by name"),
        ({"of": "ended", "at": "frame", "labels": ["end", "first"]},
         "frame has a label 'first' of its own; a pick cannot tell it from its first label"),
    ):
        rows = _rows({"name": "x", "terms": [term]})
        assert _problem(rows, members) == ("x", "term 1", detail)
        with pytest.raises(DeriveError, match=f"^x · term 1: {detail}"):
            evaluate(rows, members)
    rows = _rows({"name": "x", "terms": [{"of": "occupied", "at": "frame", "labels": ["3"]}]})
    assert _problem(rows, members) == ("x", "term 1", "occupied has no frame '3'; frame labels: 0, 1, 2")
    rows = _rows({"name": "x", "terms": [{"of": "scan", "at": "Point", "labels": ["a 10 · b 2"]}]})
    assert _problem(rows, members) == ("x", "term 1", "scan has no Point 'a 10 · b 2'; Point values: a 10 · b 1, "
                                                      "a 10 · b 3, a 20 · b 1, a 20 · b 2, a 20 · b 3")
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

    def at(name, label):
        return {"of": name, "at": "frame", "labels": [label]}

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
    # Names, on every variable, in one rule (no Python runs: "in" is a
    # name like any other); at least one variable is published.
    assert no_data({"name": "x", "terms": [counts]}, {"name": "x", "publish": False, "terms": [counts]}) == (
        "x · name: another variable is already named x")
    assert no_data({"name": "2x", "terms": [counts]}) == (
        "2x · name: a name uses letters, digits and _, and does not start with a digit")
    assert no_data({"name": "数", "terms": [counts]}) == (
        "数 · name: a name uses letters, digits and _, and does not start with a digit")
    assert no_data({"name": "", "terms": [counts]}) == "variable 1 · name: choose a name"
    assert DERIVE_SCHEMA.project_values({"variables": [{"name": "in", "terms": [counts]}]})["variables"][0]["name"] == "in"
    assert no_data({"name": "x", "publish": False, "terms": [counts]}) == NO_PUBLISHED
    # What a variable reads is the source's to say: a variable named like
    # an input reads that input, and only the inputs' names tell a
    # variable that reads itself, or a cycle, from it.  So neither is
    # refused without them; with them -- typing, admission, the build --
    # each is said once, by what it is.
    reads_itself = ({"name": "x", "terms": [counts, {"op": "+", "of": "x"}]},)
    cycle = ({"name": "kept", "terms": [counts, {"op": "where", "of": "same"}]},
             {"name": "same", "terms": [occupied, {"op": "==", "of": "kept"}]})
    named = ({"name": "counts", "terms": [counts, {"op": "*", "number": 2}]},)
    false_cycle = ({"name": "occupied", "terms": [{"of": "kept"}, {"op": ">", "number": 1}]},
                   {"name": "kept", "terms": [counts, {"op": "where", **occupied}]})
    for variables in (reads_itself, cycle, named, false_cycle):
        DERIVE_SCHEMA.project_values({"variables": variables})
    assert typed(*reads_itself) == ("x", "term 2", "x reads itself")
    assert typed(*cycle) == ("same", "term 2", "same reads kept, which reads same")
    assert typed(*named) == ("counts", "name", "counts is an input's name; choose another")
    answers = typed_variables(_rows(*false_cycle), {name: value.schema for name, value in members.items()})
    assert str(answers[0]["error"]) == "occupied · name: occupied is an input's name; choose another"
    assert answers[1]["error"] is None and answers[1]["reads"] == ()
    assert str(typed_variables(_rows(*reads_itself), {"counts": None})[0]["error"]) == "x · term 2: x reads itself"
    for variables, text in ((reads_itself, "x · term 2: x reads itself"),
                            (cycle, "same · term 2: same reads kept, which reads same")):
        with pytest.raises(DeriveError, match=f"^{text}$"):
            input_members(_rows(*variables), ("counts", "occupied"))
    # An empty name -- one being retyped -- is read by nothing: a number is
    # never taken for a reference to it.
    answers = typed_variables(_rows({"name": "", "terms": [counts]},
                                    {"name": "brighter", "terms": [counts, {"op": "+", "number": 5}]}),
                              {name: value.schema for name, value in members.items()})
    assert (answers[0]["read_by"], answers[1]["waits_for"], answers[1]["error"]) == ((), None, None)
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
    assert typed({"name": "x", "terms": [counts, {"op": "+", "of": "smae"}]}) == ("x", "term 2", (
        "no input or variable named 'smae'; inputs: counts, occupied; variables: x"))
    assert typed({"name": "counts", "terms": [occupied]}) == ("counts", "name", "counts is an input's name; choose another")
    assert typed({"name": "x", "terms": [counts, {"op": "==", **occupied}]}) == ("x", "term 2", (
        "booleans do not mix with numbers; compare booleans with = or ≠, or count them with a statistic"))
    assert typed({"name": "x", "terms": [occupied, {"op": ">", **occupied}]}) == ("x", "term 2",
                                                                                  "booleans compare only with = and ≠")
    assert typed({"name": "x", "terms": [occupied, {"op": "and", **counts}]}) == ("x", "term 2",
                                                                                  "and needs booleans; counts is a number")
    assert typed({"name": "x", "terms": [at("occupied", "0"), {"op": "+", **at("occupied", "2")}]}) == (
        "x", "term 2", "+ needs numbers; occupied[frame 0] is a boolean; count booleans with a statistic, or "
                       "compare them with = or ≠")
    # Where precedence groups the formula otherwise than its lines read, an
    # operator's problem says how it reads: where binds last.
    assert typed({"name": "same", "publish": False, "terms": [at("occupied", "0")]},
                 {"name": "kept", "terms": [counts, {"op": "where", "of": "same"}, {"op": "+", "number": 1}]}) == (
        "kept", "term 3", "+ needs numbers; same is a boolean; count booleans with a statistic, or compare them "
                          "with = or ≠; this formula reads counts where (same + 1)")
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
    assert error_stage("failed: 数 · term 1: no") is None, "a failure is read back by the names Derive allows"


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


def test_the_descriptor_declares_published_variables_and_refuses_picks_publications_do_not_share() -> None:
    """Declarations come from names alone, in card order, for published
    variables only; a formula being edited keeps them, a name problem
    withdraws that variable's alone.  A pick on an axis each publication
    carries its own part of -- a finite run's shots and scan points, a
    monitor's shots -- is refused at admission, as typing says it under its
    line; should one reach a run, the publication that lacks the value
    fails it in the variable's words."""

    def declared(*variables):
        return [(item.name, item.contract_id) for item in LOGIC_NODE.outputs_for({"variables": list(variables)}, {})]

    helper = {"name": "helper", "publish": False, "terms": [{"of": "counts"}]}
    assert declared({"name": "b", "terms": [{"of": "helper"}]}, helper, {"name": "a", "terms": [{"of": "x"}]}) == [
        ("b", "derive.b"), ("a", "derive.a")]
    assert declared({"name": "b", "terms": [{"of": "counts"}, {"op": "+"}]}) == [("b", "derive.b")]
    assert declared({"name": "b", "terms": []}, {"name": "b", "terms": []}) == [("b", "derive.b")]
    assert declared({"name": "", "terms": []}, {"name": "a", "terms": []}) == [("a", "derive.a")]
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
    variables = [{"name": "far", "terms": [{"of": "counts", "at": "detuning", "labels": ["3"]}]},
                 {"name": "early", "terms": [{"of": "counts", "at": "Site", "labels": ["1"]}]},
                 {"name": "shot", "terms": [{"of": "counts", "at": "repeat", "labels": ["0"]}]}]
    rows = _rows(*variables)
    assert {key.value: labels for key, labels in changing_axes(whole, event).items()} == {"rf.detuning": ("2",)}
    assert {key.value: labels for key, labels in changing_axes(event, event).items()} == {"camera.repeat": ("0",)}
    answers = typed_variables(rows, {"counts": whole}, changing_axes(whole, event))
    assert [str(answer["error"]) for answer in answers] == [
        "far · term 1: detuning changes with each publication (the latest carries 2); reduce over detuning with "
        "a statistic, or plot it in a panel", "None", "None"]

    def publication(canonical):
        snapshot = owned_snapshot_from_arrays(event, np.zeros((1, 1, 2)), 0, validity=np.ones((1, 1, 2), bool))
        return SimpleNamespace(signals={COUNTS: SignalValue(
            COUNTS, snapshot, coverage=MonitorCoverage(1, 1),
            canonical_schema=canonical, cell_origin=None if canonical is None else (0, 0))})

    draft = DERIVE_SCHEMA.project_values({"variables": variables})
    assert LOGIC_NODE.admit_source(draft, publication(whole)) == (str(answers[0]["error"]),)
    assert LOGIC_NODE.admit_source(draft, publication(None)) == (
        "far · term 1: counts has no detuning '3'; detuning labels: 2",
        "shot · term 1: repeat changes with each publication (the latest carries 0); reduce over repeat with a "
        "statistic, or plot it in a panel",)
    with pytest.raises(DeriveError) as absent:
        evaluate(rows[:1], {"counts": Operand(event, np.zeros((1, 1, 2)))})
    assert str(absent.value) == "far · term 1: this publication carries no detuning '3'; detuning labels here: 2"
    assert error_stage(f"failed: {absent.value}")[:2] == ("far", "term 1")


def test_the_frames_a_draft_says_are_the_frames_typing_keeps() -> None:
    """Before any data, a published variable's frames are told from the
    source's draft frames by typing's own rules on the frame axis
    (``frames_kept``, read by ``describe_output_frames``).  On real member
    schemas, for every formula typing types, they are the readout-event
    labels of the typed schema; where they cannot be told, typing refuses
    too.  One frame ("0") stands for a variable that keeps no frame axis,
    and a source of one label is read as carrying none.  What only the
    members' names decide, ``frames_kept`` is told as typing is; the
    descriptor is not, and leaves it to typing."""

    occupancy = {name: value.schema for name, value in _occupancy().items()}
    # Point carries frame x detuning (a mapped domain, detuning-major), Cell y x.
    frame = occupancy["counts"].point_domain.axes[0]
    detuning = AxisSpec(AxisId("rf.detuning"), "detuning", SCAN_POINT, 2, (1, 2), "Hz")
    y = AxisSpec(AxisId("camera.y"), "y", SITE, 2, (0, 1))
    x = AxisSpec(AxisId("camera.x"), "x", SITE, 2, (0, 1))
    point = DomainSpec((6,), (frame, detuning), ((0, 1, 2, 0, 1, 2), (0, 0, 0, 1, 1, 1)))
    scanned = {name: replace(schema, point_domain=point, cell_domain=DomainSpec((2, 2), (y, x)),
                             value_schema=ValueSchema(ValidityContract.components(y.axis_id, x.axis_id),
                                                      schema.value_schema.dtype, schema.value_schema.value_unit))
               for name, schema in occupancy.items()}
    assert pick_values(scanned["counts"], "Point")[:4] == (
        "frame 0 · detuning 1", "frame 1 · detuning 1", "frame 2 · detuning 1", "frame 0 · detuning 2")

    def at(of, axis, *labels, **extra):
        return {"of": of, "at": axis, "labels": list(labels), **extra}

    def published(name, *terms, **extra):
        return {"name": name, "terms": list(terms), **extra}

    same = {"name": "same", "publish": False, "terms": [at("occupied", "frame", "0"),
                                                        at("occupied", "frame", "2", op="==")]}
    level = {"name": "level", "publish": False, "terms": [at("counts", "frame", "1"),
                                                          at("occupied", "frame", "1", op="where")],
             "statistic": "mean", "over": ["Site"]}
    shared = [
        same,
        published("kept", {"of": "counts"}, {"op": "where", "of": "same"}),
        published("whole", {"of": "counts"}),
        published("one", at("counts", "frame", "1")),
        published("two", at("counts", "frame", "2", "0")),
        published("every", at("counts", "frame", "0", "1", "2")),
        published("ends", at("counts", "frame", "first", "last")),
        published("final", at("counts", "frame", "last")),
        published("zero", at("counts", "frame", "first", "0")),
        published("rise", at("counts", "frame", "1", "2"), at("counts", "frame", "0", op="-")),
        published("pair", at("counts", "frame", "0", "1"), at("counts", "frame", "0", "1", op="/")),
        published("steady", {"of": "counts"}, {"op": ">", "number": 30}, {"op": "and", "of": "occupied"}),
        published("window", at("counts", "frame", "1"), {"op": ">", "number": 10},
                  at("counts", "frame", "1", op="and"), {"op": "<", "number": 50},
                  at("occupied", "frame", "1", op="and")),
        published("shot", at("counts", "repeat", "2"), {"op": "-", "of": "counts"}),
        published("over_frame", {"of": "occupied"}, statistic="sum", over=["frame"]),
        published("frame_mean", {"of": "counts"}, statistic="mean", over=["repeat", "frame"]),
        published("again", {"of": "over_frame"}, {"op": "*", "number": 2}),
        # What typing refuses on the frame axis, the draft cannot say either.
        published("lost", {"of": "over_frame"}, {"op": "where", "of": "occupied"}),
        published("mixed", at("counts", "frame", "0", "1"), at("counts", "frame", "1", "2", op="+")),
        published("masked", at("counts", "frame", "1"), {"op": "where", "of": "occupied"}),
        published("absent", at("counts", "frame", "3")),
        published("twice", at("counts", "frame", "1", "1")),
        published("gone", at("over_frame", "frame", "0")),
        published("reduced", {"of": "one"}, statistic="sum", over=["frame"]),
        published("cycle_a", {"of": "cycle_b"}), published("cycle_b", {"of": "cycle_a"}),
        # A number is never a boolean, so beside and/or typing refuses it.
        published("and_number", {"of": "occupied"}, {"op": "and", "number": 30}),
        published("number_or", {"number": 1}, {"op": "or", "of": "occupied"}),
        published("not_number", {"of": "occupied"}, {"op": "and not", "number": 0}),
    ]
    by_source = (
        (occupancy, [
            level, published("norm", {"of": "counts"}, {"op": "/", "of": "level"}),
            published("sites", at("counts", "Site", "3", "1")),
            published("per_frame", {"of": "counts"}, statistic="mean", over=["Site"]),
        ]),
        (scanned, [
            published("point", at("counts", "Point", "frame 1 · detuning 2")),
            published("points", at("counts", "Point", "frame 2 · detuning 1", "frame 0 · detuning 2")),
            published("points_one", at("counts", "Point", "frame 1 · detuning 1", "frame 1 · detuning 2")),
            published("pixel", at("counts", "Cell", "y 1 · x 0")),
            published("corner", at("counts", "Cell", "y 0 · x 0", "y 0 · x 1")),
            published("tuned", at("counts", "detuning", "2")),
            published("point_last", at("counts", "Point", "last")),
            published("per_frame", {"of": "counts"}, statistic="mean", over=["detuning", "y"]),
            published("point_absent", at("counts", "Point", "frame 3 · detuning 1")),
        ]),
    )
    told = {}
    for members, own in by_source:
        draft = {"variables": [*shared, *own]}
        rows = _rows(*draft["variables"])
        answers = typed_variables(rows, members)
        kept = frames_kept(rows, FRAME_AXIS_NAME, ("0", "1", "2"), tuple(members))
        assert kept == frames_kept(rows, FRAME_AXIS_NAME, ("0", "1", "2"), None), "names alone decide nothing here"
        for row, answer, frames in zip(rows, answers, kept):
            if answer["schema"] is not None:
                axes = [axis for axis in answer["schema"].point_domain.axes if axis.role == READOUT_EVENT]
                assert frames == (axes[0].labels() if axes else ()), row["name"]
            if frames is None:
                assert answer["error"] is not None or answer["waits_for"] is not None, row["name"]
            if row["publish"]:
                answer = LOGIC_NODE.describe_output_frames(draft, row["name"], (FRAME_AXIS_NAME, ("0", "1", "2")))
                assert answer == (None if frames is None else (FRAME_AXIS_NAME, frames) if frames
                                  else (None, ("0",))), row["name"]
                told[row["name"]] = None if answer is None else answer[1]
    assert told == {
        "kept": ("0", "1", "2"), "whole": ("0", "1", "2"), "one": ("0",), "two": ("0", "2"),
        "every": ("0", "1", "2"), "ends": ("0", "2"), "final": ("0",), "zero": ("0",), "rise": ("1", "2"), "pair": ("0", "1"), "steady": ("0", "1", "2"),
        "window": ("0",), "shot": ("0", "1", "2"), "over_frame": ("0",), "frame_mean": ("0",), "again": ("0",),
        "lost": None, "mixed": None, "masked": None, "absent": None, "twice": None, "gone": None,
        "reduced": None, "cycle_a": None, "cycle_b": None, "and_number": None, "number_or": None,
        "not_number": None, "norm": ("0", "1", "2"), "sites": ("0", "1", "2"),
        "per_frame": ("0", "1", "2"), "point": ("0",), "points": ("0", "2"), "points_one": ("1",),
        "pixel": ("0", "1", "2"), "corner": ("0", "1", "2"), "tuned": ("0", "1", "2"), "point_last": ("0",),
        "point_absent": None,
    }
    # First and Last are tokens of the draft, never labels fixed when it was
    # written: the frames passed on follow the source's frame count.
    ends = {"variables": [published("ends", at("counts", "frame", "first", "last"))]}
    assert [LOGIC_NODE.describe_output_frames(ends, "ends", (FRAME_AXIS_NAME, tuple(map(str, range(count)))))[1]
            for count in (3, 4, 2)] == [("0", "2"), ("0", "3"), ("0", "1")]
    draft = {"variables": [same, published("kept", {"of": "counts"}, {"op": "where", "of": "same"})]}
    assert LOGIC_NODE.describe_output_frames(draft, "kept", None) is None, "an upstream that cannot say"
    assert LOGIC_NODE.describe_output_frames(draft, "same", (FRAME_AXIS_NAME, ("0", "1", "2"))) is None, (
        "an internal variable")
    assert LOGIC_NODE.describe_output_frames({"variables": [{"name": "kept", "code": "x"}]}, "kept",
                                             (FRAME_AXIS_NAME, ("0",))) is None

    # What the members' names decide.  frames_kept is told them as typing
    # is: a member's name is read before a variable's, a variable named like
    # a member has a name problem, a name nothing has is refused.  The
    # descriptor is told only frames, so it reads a name no variable has as
    # a member's; typing (and Start) refuses what it cannot see.
    named = _rows(
        {"name": "counts", "publish": False, "terms": [at("occupied", "frame", "1")]},
        published("shadowed", {"of": "counts"}),
        published("unknown", {"of": "same2"}),
    )
    answers = typed_variables(named, occupancy)
    assert [answer["error"] is not None for answer in answers] == [True, False, True]
    assert [axis.labels() for axis in answers[1]["schema"].point_domain.axes if axis.role == READOUT_EVENT] == [
        ("0", "1", "2")], "shadowed reads the member counts"
    assert frames_kept(named, FRAME_AXIS_NAME, ("0", "1", "2"), tuple(occupancy)) == (None, ("0", "1", "2"), None)
    assert frames_kept(named, FRAME_AXIS_NAME, ("0", "1", "2"), None) == ((), (), ("0", "1", "2"))

    # One label is one frame or no frame axis, and the source does not say
    # which: the descriptor reads it as no frame axis.  A pick or statistic
    # on the frame axis cannot be told; everything else keeps one frame.
    # Against members that carry no frame axis, typing agrees.
    reducing = _rows(published("counts_mean", {"of": "counts"}, statistic="mean", over=["frame"]),
                     published("occupied_any", {"of": "occupied"}, statistic="any", over=["frame"]))
    reduced = {name: answer["schema"] for name, answer in zip(occupancy, typed_variables(reducing, occupancy))}
    for name, value in reduced.items():
        assert not [axis for axis in value.point_domain.axes if axis.role == READOUT_EVENT], name
    single = [published("whole", {"of": "counts"}), published("picked", at("counts", "frame", "0")),
              published("summed", {"of": "counts"}, statistic="sum", over=["frame"]),
              published("sites", {"of": "counts"}, statistic="mean", over=["Site"]),
              published("kept", {"of": "occupied"}, {"op": "where", "of": "occupied"})]
    told = {row["name"]: LOGIC_NODE.describe_output_frames({"variables": single}, row["name"], (FRAME_AXIS_NAME, ("0",)))
            for row in _rows(*single)}
    assert told == {"whole": (None, ("0",)), "picked": None, "summed": None, "sites": (None, ("0",)),
                    "kept": (None, ("0",))}
    for row, answer in zip(_rows(*single), typed_variables(_rows(*single), reduced)):
        assert (answer["error"] is None) == (told[row["name"]] is not None), row["name"]

    # Over Frame Survival the readout-event axis is "pair": the source says
    # its name with its labels, and a pick or statistic naming it is told
    # as typing types it.
    pair = AxisSpec(AxisId("survival.pair"), "pair", READOUT_EVENT, 3, (0, 1, 2),
                    coordinate_labels=("0→1", "0→2", "1→2"))
    survival = replace(occupancy["occupied"], point_domain=DomainSpec((3,), (pair,), ((0, 1, 2),)))
    over_pairs = [published("ends", at("survival", "pair", "first", "last")),
                  published("one", at("survival", "pair", "0→2")),
                  published("rate", {"of": "survival"}, statistic="mean", over=["pair"]),
                  published("sites", at("survival", "Site", "1", "2"))]
    rows = _rows(*over_pairs)
    told = {row["name"]: LOGIC_NODE.describe_output_frames({"variables": over_pairs}, row["name"],
                                                           ("pair", pair.labels())) for row in rows}
    assert told == {"ends": ("pair", ("0→1", "1→2")), "one": (None, ("0",)), "rate": (None, ("0",)),
                    "sites": ("pair", ("0→1", "0→2", "1→2"))}
    for row, answer in zip(rows, typed_variables(rows, {"survival": survival})):
        axes = [axis for axis in answer["schema"].point_domain.axes if axis.role == READOUT_EVENT]
        assert told[row["name"]] == (("pair", axes[0].labels()) if axes else (None, ("0",))), row["name"]


def _editor_session(rows=(), bundle=None):
    """The editor as the Edit page hosts it, shown offscreen, with helpers
    that drive its real widgets: clicks on controls and on a list row's
    words or box."""
    from PyQt5 import QtCore, QtTest

    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["zlc-derive-editor-test"])
    editor = LOGIC_NODE.ui_contributions[0]()
    drafts = []
    editor.draft_changed.connect(lambda patch: drafts.append(patch["values"]["variables"]))
    if bundle is None:
        bundle = (("counts", _schema(1, 3, 6, np.float32, "count"), None),
                  ("occupied", _schema(1, 3, 6, bool, "1"), None))

    def settle():
        for _ in range(3):
            app.processEvents()

    def project(shown, inputs=bundle, error=""):
        editor.update_projection({"form_values": {"variables": shown}, "input_bundle": inputs,
                                  "node_id": "derive", "error": error})
        settle()
        return editor._cards

    def click(widget, pos=None, modifier=QtCore.Qt.NoModifier):
        QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, modifier,
                                widget.rect().center() if pos is None else pos)
        settle()

    def row_at(combo, row, *, box=False, modifier=QtCore.Qt.NoModifier):
        """Click row ``row`` of an open list: on its words, or on its box."""
        view = combo.view()
        assert view.window().isVisible(), "the list is open"
        rect = view.visualRect(combo.model().index(row, 0))
        click(view.viewport(), QtCore.QPoint(rect.left() + 4 if box else rect.center().x(), rect.center().y()),
              modifier)

    def key(widget, which):
        QtTest.QTest.keyClick(widget, which)
        settle()
        settle()

    def pick(combo, data):
        click(combo)
        row_at(combo, combo.findData(data))

    def texts(combo):
        return [combo.itemText(row) for row in range(combo.count())]

    editor.resize(1200, 800)
    editor.set_label_width(editor.label_width_floor())
    editor.show()
    project(rows)
    return SimpleNamespace(app=app, editor=editor, drafts=drafts, settle=settle, project=project, click=click,
                           row_at=row_at, pick=pick, texts=texts, key=key)


def test_the_editor_builds_the_users_example_by_clicks() -> None:
    """Add variable, a name (written when finished), an operand, an axis --
    its labels list opens at once, and the pick is written once a label is:
    a click on a label's box ticks it, Shift with it a run, on its words
    picks only it; Escape leaves the pick as it was, and the axis a line
    already picks opens its labels again -- and Publish off; + Comparison
    pins the same axis and asks for its label, and + Where masks the next
    variable with it: the draft is the user's example.  Renaming a
    variable renames what reads it.  Each operator kind is its own strip,
    withholding what the typed values cannot take; "over" ticks axes by
    their words, a domain by its header; × removes a line or the variable."""
    from PyQt5 import QtCore, QtTest, QtWidgets

    s = _editor_session()
    editor, drafts = s.editor, s.drafts

    def faces(strip):
        return [button.text() for button in strip.findChildren(QtWidgets.QPushButton)]

    try:
        s.click(editor.add_variable)
        assert drafts[-1] == ({"name": "variable_1", "terms": ({"of": "counts"},)},)
        (same,) = editor._cards
        head = same.head
        assert head["name"].selectedText() == "variable_1", "the name is selected to type over"
        written = len(drafts)
        QtTest.QTest.keyClicks(head["name"], "same")
        assert len(drafts) == written, "a name is not written per keystroke"
        s.key(head["name"], QtCore.Qt.Key_Return)
        assert drafts[-1][0]["name"] == "same"
        assert s.texts(head["operand"]) == ["Inputs", "    counts", "    occupied", "Constants", "    Number"]
        s.pick(head["operand"], "occupied")
        assert s.texts(head["axis"]) == ["All", "Repeat", "    repeat", "Point", "    frame", "Cell", "    Site"]
        written, preview = len(drafts), same.preview.text()
        s.pick(head["axis"], "frame")
        labels = head["labels"]
        assert s.texts(labels) == ["All", "First", "Last", "frame", "0", "1", "2"], "the ends, then the labels"
        assert labels.view().window().isVisible(), "its labels list opens"
        assert head["error"].isHidden(), "no problem while its labels are being picked"
        assert (len(drafts), same.preview.text()) == (written, preview), "nothing changes until a label is picked"
        s.key(labels.view(), QtCore.Qt.Key_Escape)
        assert head["axis"].currentText() == "All" and len(drafts) == written, "Escape leaves the pick as it was"
        s.pick(head["axis"], "frame")
        s.row_at(labels, 5, box=True)
        s.row_at(labels, 6, box=True)
        assert labels.view().window().isVisible() and labels.checked_data() == ("1", "2"), "a box ticks"
        s.row_at(labels, 4, box=True)
        s.row_at(labels, 6, box=True, modifier=QtCore.Qt.ShiftModifier)
        assert labels.checked_data() == ("0", "1", "2"), "Shift ticks the run from the last box clicked"
        s.row_at(labels, 5, box=True)
        s.row_at(labels, 4, box=True, modifier=QtCore.Qt.ShiftModifier)
        assert labels.checked_data() == ("2",), "... or clears it, as that box now is"
        s.row_at(labels, 4)
        assert not labels.view().window().isVisible() and labels.checked_data() == ("0",), "words: only this"
        assert drafts[-1][0]["terms"][0] == {"of": "occupied", "at": "frame", "labels": ("0",)}
        s.pick(head["axis"], "frame")
        assert labels.view().window().isVisible(), "the axis already picked opens its labels again"
        # First and Last are the frame axis' ends whatever its length: the
        # draft keeps them as written, and the box reads them so.
        s.row_at(labels, 1, box=True)
        s.row_at(labels, 2, box=True)
        s.row_at(labels, 4, box=True)
        s.key(labels.view(), QtCore.Qt.Key_Escape)
        assert drafts[-1][0]["terms"][0] == {"of": "occupied", "at": "frame", "labels": ("first", "last")}
        assert labels.toolTip() == "first, last" and same.preview.text().startswith("→ repeat 1 × frame 2 × Site 6")
        # All is no pick, as the axis list's All is.
        s.pick(head["axis"], "frame")
        s.row_at(labels, 0)
        assert drafts[-1][0]["terms"][0] == {"of": "occupied"}
        assert head["axis"].currentText() == "All" and labels.isHidden()
        s.pick(head["axis"], "frame")
        s.row_at(labels, 4)
        s.click(head["publish"])
        assert drafts[-1] == ({"name": "same", "publish": False, "terms": (
            {"of": "occupied", "at": "frame", "labels": ("0",)},)},)

        s.click(same.add["comparison"])
        line = same.terms_ui[1]
        assert faces(line["strip"]) == ["=", "≠", ">", "<", "≥", "≤"]
        assert line["labels"].view().window().isVisible(), "the pick asks for its label at once"
        s.row_at(line["labels"], 6)
        assert drafts[-1][0]["terms"][1] == {"op": "==", "of": "occupied", "at": "frame", "labels": ("2",)}
        assert [line["strip"].button(op).isEnabled() for op in ("==", "!=", ">", "<", ">=", "<=")] == [
            True, True, False, False, False, False], "booleans compare only with = and ≠"
        assert same.preview.text() == (
            "→ repeat 1 × Site 6 · bool · unit 1 · internal · read by nothing, so not evaluated")

        s.click(editor.add_variable)
        kept = editor._cards[1]
        QtTest.QTest.keyClicks(kept.head["name"], "kept")
        s.click(kept.add["where"])
        assert faces(kept.terms_ui[1]["strip"]) == ["where", "where not"]
        example = (
            {"name": "same", "publish": False, "terms": (
                {"of": "occupied", "at": "frame", "labels": ("0",)},
                {"op": "==", "of": "occupied", "at": "frame", "labels": ("2",)})},
            {"name": "kept", "terms": ({"of": "counts"}, {"op": "where", "of": "same"})},
        )
        assert drafts[-1] == example, "the user's example, as the run record keeps it"
        # A rename follows into what reads the variable, when the new name is free.
        same.head["name"].selectAll()
        QtTest.QTest.keyClicks(same.head["name"], "agree")
        s.key(same.head["name"], QtCore.Qt.Key_Return)
        assert drafts[-1][1]["terms"][1] == {"op": "where", "of": "agree"}
        same.head["name"].selectAll()
        QtTest.QTest.keyClicks(same.head["name"], "same")
        s.key(same.head["name"], QtCore.Qt.Key_Return)
        assert drafts[-1] == example
        assert same.preview.text().endswith("· internal · read by kept")
        assert kept.preview.text() == "→ repeat 1 × frame 3 × Site 6 · float32 · unit count · valid where same"
        assert "    same" in s.texts(kept.head["operand"])
        assert "    kept" not in s.texts(same.head["operand"]), "same may not read what reads it: no cycle"

        # The other strips, the statistic and every ×.
        for kind, shown in (("arithmetic", ["+", "−", "×", "÷"]), ("logic", ["and", "or", "and not"])):
            s.click(kept.add[kind])
            assert faces(kept.terms_ui[-1]["strip"]) == shown
            s.click(kept.terms_ui[-1]["remove"])
            assert drafts[-1] == example
        s.click(kept.add["statistic"])
        statistic = kept.statistic
        assert len(faces(statistic["strip"])) == 8 and drafts[-1] == example, "opened, nothing chosen yet"
        assert kept.add["statistic"].isHidden()
        assert [statistic["strip"].button(name).isEnabled() for name in ("min", "any")] == [True, False]
        s.click(statistic["strip"].button("mean"))
        assert (drafts[-1][1]["statistic"], drafts[-1][1]["over"]) == ("mean", ("Site",)), "over every Cell axis"
        over = statistic["over"]
        s.click(over)
        assert s.texts(over) == ["Repeat", "repeat", "Point", "frame", "Cell", "Site"]
        s.row_at(over, 1)
        assert over.view().window().isVisible() and over.checked_data() == ("repeat", "Site"), "words tick"
        s.row_at(over, 2)
        assert drafts[-1][1]["over"] == ("frame",), "a domain's header picks its axes"
        s.click(statistic["remove"])
        assert drafts[-1] == example and not kept.add["statistic"].isHidden()
        s.click(editor.add_variable)
        s.click(editor._cards[-1].head["remove"])
        assert drafts[-1] == example and len(editor._cards) == 2
        editor.set_mutation_enabled(False)
        assert not editor.add_variable.isEnabled() and not any(card.isEnabled() for card in editor._cards)
    finally:
        editor.close()
        editor.deleteLater()
        s.app.processEvents()


def test_the_editor_offers_the_frame_pick_before_data_only_where_the_frames_are_known() -> None:
    """Before the source publishes, an operand offers the readout-event
    axis the source's draft says every member carries, by the name the
    source gives it, with the labels that operand keeps: a member's are the
    source's, a variable's are told by ``frames_kept`` with the members'
    names, as typing reads them -- a variable reading a name nothing has
    keeps none, one reading a member shadowed by a variable's name keeps
    the member's.  One frame is no pick (it may as well be no frame axis);
    nothing is offered while the source cannot say."""
    s = _editor_session()
    frame, pair = ["All", "Point", "    frame"], ["All", "Point", "    pair"]
    rows = [
        {"name": "kept", "terms": [{"of": "counts"}]},
        {"name": "v", "terms": [{"of": "kept"}]},
        {"name": "x", "terms": [{"of": "ghost"}]},
        {"name": "w", "terms": [{"of": "x"}]},
        {"name": "counts", "publish": False, "terms": [{"of": "occupied"}], "statistic": "any", "over": ["frame"]},
        {"name": "shadow", "terms": [{"of": "counts"}]},
        {"name": "u", "terms": [{"of": "shadow"}]},
    ]

    def offered(source_frames, axis="frame"):
        s.editor.update_projection({
            "form_values": {"variables": rows}, "node_id": "derive", "error": "",
            "input_bundle": (("counts", None, None), ("occupied", None, None)), "source_frames": source_frames,
            "source_frame_axis": axis})
        s.settle()
        return [s.texts(card.head["axis"]) if card.head["axis"].isVisibleTo(s.editor) else None
                for card in s.editor._cards]

    try:
        assert offered(("0", "1", "2")) == [frame, frame, None, None, frame, frame, frame]
        assert offered(("0→1", "0→2", "1→2"), "pair") == [pair, pair, None, None, pair, pair, pair]
        assert offered(("0",)) == [None] * 7
        assert offered(None) == [None] * 7
    finally:
        s.editor.close()
        s.editor.deleteLater()
        s.app.processEvents()


def test_the_editor_offers_what_each_publication_shares_and_domains_by_their_header() -> None:
    """The axis list leaves out an axis each publication carries its own
    part of (a monitor's shots) and offers a domain of two or more axes by
    its header, whose labels are its rows; Add variable brings its card
    into view on a long page."""
    from PyQt5 import QtCore, QtWidgets

    from zlc_ui.fluent import FluentScrollArea

    base = _schema(1, 3, 6, np.float32, "count")
    y = AxisSpec(AxisId("camera.y"), "y", SITE, 3, (0, 1, 2))
    x = AxisSpec(AxisId("camera.x"), "x", SITE, 2, (0, 1))
    image = replace(base, cell_domain=DomainSpec((3, 2), (y, x)), value_schema=ValueSchema(
        ValidityContract.components(y.axis_id, x.axis_id), np.dtype(np.float32), "count"))
    # A monitor: what typing reads is its latest publication.
    latest = SimpleNamespace(block=SimpleNamespace(schema=base))
    bundle = (("counts", base, latest), ("frames", image, None))
    s = _editor_session([{"name": "px", "terms": [{"of": "frames"}]}], bundle)
    editor, drafts = s.editor, s.drafts
    try:
        (card,) = editor._cards
        axis = card.head["axis"]
        assert s.texts(axis) == ["All", "Point", "    frame", "Cell", "    y", "    x"]
        assert "changes with each publication" in editor._input_rows[0][0].toolTip()
        s.pick(axis, "Cell")
        labels = card.head["labels"]
        assert s.texts(labels) == ["All", "First", "Last", "Cell", "y 0 · x 0", "y 0 · x 1", "y 1 · x 0", "y 1 · x 1",
                                   "y 2 · x 0", "y 2 · x 1"]
        s.row_at(labels, 7)
        assert drafts[-1] == ({"name": "px", "terms": ({"of": "frames", "at": "Cell", "labels": ("y 1 · x 1",)},)},)
        assert axis.currentText() == "Cell" and card.preview.text() == "→ repeat 1 × frame 3 · float32 · unit count"
        # A pick on the shots, written by an agent, is said under its line.
        cards = s.project([{"name": "shot", "terms": [{"of": "counts", "at": "repeat", "labels": ["0"]}]}])
        assert cards[0].head["error"].text() == (
            "repeat changes with each publication (the latest carries 0); reduce over repeat with a statistic, "
            "or plot it in a panel.")
        assert s.texts(cards[0].head["axis"])[-1] == "repeat"

        # On a page that scrolls, a new card comes into view, its name ready.
        page = FluentScrollArea()
        page.set_width_bounded_widget(editor)
        page.resize(1200, 300)
        page.show()
        s.project([{"name": f"v{index}", "terms": [{"of": "counts"}]} for index in range(6)])
        s.click(editor.add_variable)
        s.settle()
        name = editor._cards[-1].head["name"]
        top = name.mapTo(page.viewport(), QtCore.QPoint()).y()
        assert 0 <= top and top + name.height() <= page.viewport().height()
        assert QtWidgets.QApplication.focusWidget() is name
        page.close()
    finally:
        editor.close()
        editor.deleteLater()
        s.app.processEvents()


def test_the_editor_reads_down_one_control_column_and_says_problems_under_their_lines() -> None:
    """The Inputs' names, every operand, strip of statistics, add command,
    result and problem start on one x; operator strips end on the name
    field's right edge; every ``at``, axis, labels, Publish and × has one x,
    and opening or removing a line, or a long name, moves none of them.  The
    label column the editor asks for is the same with or without cards, and
    a wider one widens the names.  On a narrow page the add commands and a
    statistic's "over" run on to a second row, the word and × beside them
    staying on the first, and nothing passes the card's edge.  A problem
    stands under its line where the result stood -- the card keeps its
    height -- and a parked helper's is muted and blocks nothing.  Before
    the source publishes the lists offer only what is stored, an empty
    operand says "choose", and Tab walks the controls as they read."""
    from PyQt5 import QtCore, QtTest, QtWidgets

    from zlc_ui.fluent import MUTED_INK, field_text_inset, row_gap

    def at(of, *labels, **extra):
        return {"of": of, "at": "frame", "labels": list(labels), **extra}

    rows = [
        {"name": "same", "publish": False, "terms": [at("occupied", "0"), at("occupied", "2", op="==")]},
        {"name": "kept", "terms": [{"of": "counts"}, {"op": "where", "of": "same"}]},
        {"name": "rise", "terms": [at("counts", "2"), at("counts", "0", op="-"), {"op": "*", "number": 2}],
         "statistic": "mean", "over": ["Site"]},
        {"name": "bright", "terms": [at("counts", "1"), at("occupied", "1", op="where")]},
    ]
    s = _editor_session()
    editor = s.editor

    def x(widget):
        return widget.mapTo(editor, QtCore.QPoint()).x()

    def text_x(label):
        return x(label) + label.contentsMargins().left()

    def lines():
        return [ui for card in editor._cards for ui in card.terms_ui]

    def columns():
        return {key: {x(ui[key]) for ui in lines() if ui[key].isVisible()}
                for key in ("operand", "at", "axis", "labels", "remove")}

    try:
        floor = editor.label_width_floor()
        s.project(rows)
        assert editor.label_width_floor() == floor, "the column stays put as cards come"
        names = []
        for width in (floor, floor + 40):
            editor.set_label_width(width)
            cards = s.project(rows)
            control = width + row_gap()
            assert x(editor.inputs_title) + editor.inputs_title.width() + row_gap() == control
            assert x(editor._input_rows[0][0]) == control + field_text_inset()
            shown = columns()
            assert shown["operand"] == {control} and all(len(xs) == 1 for xs in shown.values()), shown
            for card in cards:
                name = card.head["name"]
                for ui in card.terms_ui[1:]:
                    assert x(ui["strip"]) + ui["strip"].width() == x(name) + name.width()
                assert text_x(card.preview) == control + field_text_inset()
                assert x(card.add["arithmetic"]) == control
            rise = cards[2]
            assert x(rise.statistic["strip"]) == control
            assert {x(card.head["publish"]) for card in cards} == {x(cards[0].head["publish"])}
            assert {x(card.head["remove"]) for card in cards} == shown["remove"]
            names.append(cards[0].head["name"].width())
        assert names[1] - names[0] == 40

        # Opening or removing a line, or a long name, moves no column.
        before = columns()
        s.click(cards[1].add["arithmetic"])
        s.click(cards[1].terms_ui[-1]["remove"])
        assert columns() == before
        cards[0].head["name"].selectAll()
        QtTest.QTest.keyClicks(cards[0].head["name"], "frames_zero_and_two_agree_at_every_site")
        s.key(cards[0].head["name"], QtCore.Qt.Key_Return)
        assert columns() == before and cards[1].terms_ui[1]["operand"].toolTip() == (
            "frames_zero_and_two_agree_at_every_site")
        cards = s.project(rows)

        # Narrow: what does not fit runs on, nothing passes the card's edge.
        wide = editor.width()
        editor.resize(560, editor.height())
        s.settle()
        s.settle()
        rise = cards[2]
        statistic = rise.statistic
        assert len(cards[0]._add_box._flow_rows) == 2 and len(rise._add_box._flow_rows) == 1
        assert statistic["box"]._flow_rows == (1, 1)
        assert x(statistic["reduce"]) == x(statistic["strip"]) == control

        def middle(widget):
            return widget.mapTo(editor, QtCore.QPoint()).y() + widget.height() / 2

        assert abs(middle(statistic["remove"]) - middle(statistic["strip"])) <= 1, "the × stands on the strip's row"
        assert middle(statistic["over"]) > middle(statistic["strip"]) + statistic["strip"].height()
        for card in cards:
            edge = x(card) + card.width()
            for widget in card.findChildren(QtWidgets.QWidget):
                if widget.isVisible() and widget.window() is editor.window():
                    assert x(widget) + widget.width() <= edge, widget
        editor.resize(wide, editor.height())
        s.settle()
        s.settle()

        # A problem under its line, where the result stood.
        bright = cards[3]
        result_y, height = bright.preview.mapTo(bright, QtCore.QPoint()).y(), bright.height()
        rows[3] = {"name": "bright", "terms": [at("counts", "1"), {"op": "where", "of": "occupied"}]}
        s.project(rows)
        error = bright.terms_ui[1]["error"]
        assert error.text() == "the mask has frame, which counts[frame 1] does not; pick one frame value of the mask."
        assert error.isVisible() and not bright.preview.isVisible()
        assert (error.mapTo(bright, QtCore.QPoint()).y(), bright.height()) == (result_y, height)
        assert text_x(error) == control + field_text_inset()
        assert editor.shown_problems() == {
            "bright · term 2: the mask has frame, which counts[frame 1] does not; pick one frame value of the mask":
                "Variables"}
        # A run's failure stands under the line it names.
        s.project(rows[:3], error="failed: kept · term 2: this publication carries no Site '7'")
        assert cards[1].terms_ui[1]["error"].text() == "this publication carries no Site '7'."
        # A parked helper's problem is muted and blocks nothing.
        idea = {"name": "idea", "publish": False, "terms": [{"of": "counts"}, {"op": ">"}]}
        cards = s.project([*rows[:3], idea])
        parked = cards[3].terms_ui[1]["error"]
        assert parked.text() == "choose what it reads." and parked.property("ink") == MUTED_INK
        assert cards[3].preview.text() == "→ internal · read by nothing, so not evaluated"
        assert editor.shown_problems() == {}
        assert not cards[3].statistic["block"].isVisible() and cards[2].statistic["block"].isVisible()

        # Before the source publishes: every name, and only what is stored.
        cards = s.project(rows[:3], inputs=(("counts", None, None), ("occupied", None, None)))
        head = cards[2].head
        assert s.texts(head["axis"]) == ["All", "frame"]
        assert s.texts(head["labels"]) == ["All", "First", "Last", "frame", "2"]
        assert s.texts(cards[2].statistic["over"]) == ["Site"]
        assert s.texts(cards[2].terms_ui[1]["operand"]) == [
            "Inputs", "    counts", "    occupied", "Variables", "    same", "    kept", "Constants", "    Number"]
        assert cards[2].preview.text() == "→ checked when the source publishes"
        assert editor._input_rows[0][1].text() == "waiting for its first publication"
        cards = s.project([{"name": "idea", "terms": [{"of": "counts"}, {"op": "and"}]}])
        operand = cards[0].terms_ui[1]["operand"]
        assert operand.currentIndex() < 0 and operand._display_text() == "choose", "an empty operand says so"

        # Tab walks a card as it reads, a line built later in its place.
        cards = s.project(rows[:3])
        s.click(cards[0].add["arithmetic"])

        def walk(count):
            order, widget = [], editor.add_variable
            for _ in range(count):
                widget = widget.nextInFocusChain()
                order.append(widget)
            return order

        head, later = cards[0].head, cards[0].terms_ui[-1]
        chain = [widget.focusProxy() or widget for widget in (
            head["name"], head["operand"], head["axis"], head["labels"], head["publish"], head["remove"],
            later["operand"], later["remove"], cards[0].add["arithmetic"], cards[1].head["name"])]
        order = walk(400)
        assert [item for item in order if item in chain] == chain
        s.click(later["remove"])
    finally:
        editor.close()
        editor.deleteLater()
        s.app.processEvents()
