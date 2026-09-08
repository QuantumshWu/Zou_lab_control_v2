"""What both scan nodes stand on: the plan, and the ports a bench offers it.

Nothing here knows which node runs the plan.  A plan is a document, a port is
a projection of a knob somebody already owns, and binding is where the two
meet -- before any device is touched, and identically for the board-advanced
and the host-advanced engine.
"""

from __future__ import annotations

import json

import pytest

from types import SimpleNamespace

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.install import create_installation, tunable_devices
from tests.pulse_fixture import pulse_document, pulse_sequence
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanAxis,
    ScanPlan,
    bind_plan,
    load_stepped_template,
    manual_axis,
    scan_dataset_schema,
    scan_ports_for,
    scan_ports_for_devices,
)
from zlc_atom.nodes.scan.plan import scan_axis_ids
from zlc_atom.nodes.seamless_scan import LOGIC_NODE as SEAMLESS_NODE
from test_scan_repeat_domain import _source_schema


BIAS_PORTS = tuple(
    PULSE_PARAM_FAMILY + name
    for name in ("da_bias_x", "da_bias_y", "da_bias_z")
)


def _template_sequence():
    return pulse_sequence("mot_field_template.json")


def test_the_mot_template_offers_the_three_bias_ports() -> None:
    """Ports come from the pulse's own declarations, nothing invented."""

    ports = scan_ports_for(_template_sequence())
    assert tuple(port.port for port in ports) == BIAS_PORTS
    for port in ports:
        # NOT dimensionless: a DAC code is a count of codes, and calling it
        # nothing is what an empty string says.  The registry has the unit
        # now, so the axis can carry it without a label reaching the plot.
        assert port.unit == "code"
        assert port.lo < 0 < port.hi, "the signed range brackets zero"


def test_scan_accepts_the_complete_document_saved_by_the_pulse_editor(
    tmp_path,
) -> None:
    tree = json.loads(pulse_document("mot_field_template.json").decode("utf-8"))
    tree["editor"] = {
        "visible_ports": None,
        "scan_source": "",
        "scan_rows": [],
        "scan_source_dirty": False,
        "scan_repeats": 0,
    }
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(tree), encoding="utf-8")

    assert load_stepped_template(path).name == tree["name"]


def test_plan_rows_nest_outer_first_and_round_trip() -> None:
    plan = ScanPlan(
        (
            ScanAxis(BIAS_PORTS[0], (1.0, 2.0)),
            ScanAxis(BIAS_PORTS[1], (10.0, 20.0, 30.0)),
        )
    )
    assert plan.shape == (2, 3)
    assert plan.point_count == 6
    rows = plan.rows()
    # The declared order is the nesting order: the LAST axis advances fastest.
    assert rows == (
        (1.0, 10.0), (1.0, 20.0), (1.0, 30.0),
        (2.0, 10.0), (2.0, 20.0), (2.0, 30.0),
    )
    assert ScanPlan.from_tree(plan.to_tree()) == plan


def test_binding_refuses_unknown_ports_and_out_of_range_values() -> None:
    from zlc_atom.nodes.scan.plan import ScanPort
    from zlc_data.units import DEFAULT_UNITS

    ports = scan_ports_for(_template_sequence())

    with pytest.raises(ValueError, match="offers no scan port named"):
        bind_plan(ScanPlan((ScanAxis("pulse:param:nonsense", (1.0,)),)), ports)

    with pytest.raises(ValueError, match="outside the port's range"):
        bind_plan(ScanPlan((ScanAxis(BIAS_PORTS[0], (1e9,)),)), ports)

    bound = bind_plan(
        ScanPlan((ScanAxis(BIAS_PORTS[2], (-256.0, 0.0, 256.0)),)), ports
    )
    assert bound[0].label == "da_bias_z"

    power = ScanPort("device:rf:ch1_power_dbm", "rf.ch1_power_dbm", "dBm", -30.0, 10.0)
    authored = ScanAxis(power.port, (135.0, 247.0), "mVpp")
    plan = ScanPlan((authored,))
    assert bind_plan(plan, (power,)) == (power,)
    assert plan.axes[0].values == (135.0, 247.0)
    assert ScanPlan.from_tree(plan.to_tree()) == plan
    assert authored.native_value(power, 135.0) == float(DEFAULT_UNITS.convert(135.0, "mVpp", "dBm"))
    with pytest.raises(ValueError, match="outside the port's range"):
        bind_plan(ScanPlan((ScanAxis(power.port, (1e6,), "mVpp"),)), (power,))
    with pytest.raises(ValueError, match="scan axis fields"):
        ScanPlan.from_tree({"axes": [{"port": power.port, "values": [-10.0], "display_unit": "mVpp"}]})


def test_tunable_devices_project_device_ports() -> None:
    """A device volunteers its runtime knobs as ``device:<key>:<field>`` ports.

    The aggregation is duck-typed off the installation, and only fields with
    BOTH bounds declared become ports -- a plan must be refusable against a
    finite range before anything touches hardware.
    """

    installation = create_installation("virtual")
    try:
        tunables = tunable_devices(installation)
        assert "mot_camera" in tunables and "camera" in tunables
        (exposure,) = tunables["mot_camera"].tunable_fields()
        assert exposure.metadata.name == "exposure_seconds"
        assert exposure.current == tunables["mot_camera"].tunable_values()[
            "exposure_seconds"
        ]
        assert exposure.live_write is True
        assert exposure.dependency_group == ("exposure_seconds",)
        ports = scan_ports_for_devices(tunables)
        by_name = {port.port: port for port in ports}
        key = DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds"
        assert key in by_name
        port = by_name[key]
        assert port.label == "mot_camera.exposure_seconds"
        assert 0 < port.lo < port.hi
    finally:
        installation.close()


def _field(name: str, low: float, high: float) -> TunableField:
    return TunableField(
        AuthoringField(name, "float", name, None, minimum=low, maximum=high),
        low,
        True,
        (name,),
    )


def test_a_field_pinned_to_one_value_is_a_control_not_an_axis() -> None:
    """One knob with no interval must not take the device's other knobs down.

    A live, bounded field whose minimum equals its maximum -- a knob fenced
    to one value by policy -- satisfied every port filter and then failed
    ScanPort's own "no usable initial sweep", so the whole projection
    raised and no knob of any device was offered.  Such a field is a
    control; the sweepable ones beside it are still axes.
    """

    device = SimpleNamespace(
        tunable_fields=lambda: (_field("fixed", 1.0, 1.0), _field("level", 0.0, 3.0))
    )
    ports = scan_ports_for_devices({"device": device})
    assert [port.port for port in ports] == [DEVICE_PARAM_FAMILY + "device:level"]
    from zlc_atom.authoring import read_tunable_in_unit, tune_in_unit

    delay = TunableField(
        AuthoringField("delay", "int", "Delay", 500, minimum=0, maximum=10000, unit="ns"),
        500, True, ("delay",),
    )
    written = []
    device = SimpleNamespace(tunable_fields=lambda: (delay,),
                             tune=lambda name, value: (written.append(value), value)[1])
    projected = read_tunable_in_unit(device, "delay", "us")
    assert projected.current == 0.5 and projected.metadata.value_type == "float"
    assert tune_in_unit(device, "delay", 2.0, "us") == 2.0
    assert written == [2000] and type(written[0]) is int


def test_a_region_lands_on_the_axis_the_picture_drew_when_two_ports_share_a_name() -> None:
    """The dataset names a plan's axes from the plan alone, and the selection
    names them back the same way.

    A manual ``bias`` and a pulse parameter ``bias`` are two legal ports
    with one human name.  The schema disambiguated them as ``scan.bias``
    and ``scan.bias.2``; the way back guessed ``scan.bias`` for BOTH, so a
    box drawn on the picture wrote the x range into the y axis's port --
    the wrong numbers in the wrong unit, as the next scan.
    """

    from zlc_runtime import SelectionRange, SelectionState

    plan = ScanPlan(
        (manual_axis("bias", (1.0, 2.0)), ScanAxis(PULSE_PARAM_FAMILY + "bias", (10.0, 20.0)))
    )
    labels = ("bias", "bias")
    assert scan_axis_ids(labels) == ("scan.bias", "scan.bias.2")
    schema = scan_dataset_schema(
        _source_schema(shots=1), plan.rows(), (("bias", "1"), ("bias", "code"))
    )
    assert [axis.axis_id.value for axis in schema.point_domain.axes[-2:]] == [
        "scan.bias",
        "scan.bias.2",
    ]

    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("scan.bias", 1.25, 1.75, domain="point"),
            SelectionRange("scan.bias.2", 12.0, 18.0, domain="point"),
        ),
    )
    patched = SEAMLESS_NODE.selection_patch(
        selection, draft={"plan": json.dumps(plan.to_tree())},
        context={"axis_units": {"scan.bias": "1", "scan.bias.2": "code"}},
    )
    narrowed = ScanPlan.from_tree(json.loads(patched["plan"]))
    assert narrowed.axes[0].values == (1.25, 1.75)
    assert narrowed.axes[1].values == (12.0, 18.0)

    import numpy as np
    from zlc_data import owned_snapshot_from_arrays
    from zlc_data.units import DEFAULT_UNITS
    from zlc_plot import DEFAULTS, AxisRef, CurvePlot
    from zlc_plot._fit_projection import FitProjection, ProjectionContext
    from zlc_plot.selectors import NumericRange, SelectorKind, SelectorSnapshot, SelectorState
    from zlc_plot.specs import parameter_schema_for
    from zlc_plot.state import DisplayStateStore
    from zlc_workbench.selection import panel_selection_from_plot

    power = ScanAxis("device:rf:ch1_power_dbm", tuple(np.linspace(135.0, 247.0, 10)), "mVpp")
    power_id = scan_axis_ids(("rf.ch1_power_dbm",))[0]
    schema = scan_dataset_schema(_source_schema(shots=1), ScanPlan((power,)).rows(), (("rf.ch1_power_dbm", "mVpp"),))
    snapshot = owned_snapshot_from_arrays(schema, np.zeros(schema.physical_shape), 0)
    spec = CurvePlot(AxisRef.point(power_id))
    for shown_unit, shown_bounds in (("mVpp", (150.0, 220.0)), ("Vpp", (0.15, 0.22))):
        assert DEFAULT_UNITS.compatible("mVpp", shown_unit)
        display = DisplayStateStore(parameter_schema_for(spec, style=DEFAULTS.style), {"x_display_unit": shown_unit}).state
        projection = FitProjection(data=snapshot, revision=0, spec=spec,
            context=ProjectionContext(display, SelectorSnapshot(())), unit_registry=None,
            defaults=DEFAULTS, histogram_projection=None)
        projection._build_view_and_payload()
        # Use the actual Plot conversion and Workbench event translator: their
        # canonical coordinates are the Dataset's mVpp, not the registry's W.
        bounds = projection._display_range_to_canonical(NumericRange(*shown_bounds), spec.x)
        selected = panel_selection_from_plot(
            SelectorState(SelectorKind.X_RANGE, bounds),
            projection.view.selection_subject(spec, projection.payload),
        )
        assert (selected.ranges[0].lower, selected.ranges[0].upper) == (150.0, 220.0)
        context = {"axis_units": {power_id: "mVpp"}}
        patch = SEAMLESS_NODE.selection_patch(
            selected, draft={"plan": json.dumps(ScanPlan((power,)).to_tree())}, context=context,
        )
        authored = ScanPlan.from_tree(json.loads(patch["plan"])).axes[0]
        assert authored.unit == "mVpp"
        assert authored.values == tuple(np.linspace(150.0, 220.0, 10))
        # The draft can have changed unit since this exact Dataset was shown.
        changed = ScanAxis(power.port, tuple(DEFAULT_UNITS.convert(power.values, "mVpp", "dBm")), "dBm")
        patch = SEAMLESS_NODE.selection_patch(
            selected, draft={"plan": json.dumps(ScanPlan((changed,)).to_tree())}, context=context,
        )
        authored = ScanPlan.from_tree(json.loads(patch["plan"])).axes[0]
        converted = DEFAULT_UNITS.convert((150.0, 220.0), "mVpp", "dBm")
        assert authored.unit == "dBm"
        assert authored.values == tuple(np.linspace(*converted, 10))
