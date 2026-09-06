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
    ports = scan_ports_for(_template_sequence())

    with pytest.raises(ValueError, match="offers no scan port named"):
        bind_plan(ScanPlan((ScanAxis("pulse:param:nonsense", (1.0,)),)), ports)

    with pytest.raises(ValueError, match="outside the port's range"):
        bind_plan(ScanPlan((ScanAxis(BIAS_PORTS[0], (1e9,)),)), ports)

    bound = bind_plan(
        ScanPlan((ScanAxis(BIAS_PORTS[2], (-256.0, 0.0, 256.0)),)), ports
    )
    assert bound[0].label == "da_bias_z"


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
        selection, draft={"plan": json.dumps(plan.to_tree())}, context={}
    )
    narrowed = ScanPlan.from_tree(json.loads(patched["plan"]))
    assert narrowed.axes[0].values == (1.25, 1.75)
    assert narrowed.axes[1].values == (12.0, 18.0)
