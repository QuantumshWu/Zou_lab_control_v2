"""What both scan nodes stand on: the plan, and the ports a bench offers it.

Nothing here knows which node runs the plan.  A plan is a document, a port is
a projection of a knob somebody already owns, and binding is where the two
meet -- before any device is touched, and identically for the board-advanced
and the host-advanced engine.
"""

from __future__ import annotations

import json

import pytest
from zlc_pulse import sequence_from_tree

from zlc_atom.install import create_installation, tunable_devices
from zlc_atom.nodes import scan_pulse_template_bytes
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanAxis,
    ScanPlan,
    bind_plan,
    load_scan_template,
    scan_ports_for,
    scan_ports_for_devices,
)


BIAS_PORTS = tuple(
    PULSE_PARAM_FAMILY + name
    for name in ("da_bias_x", "da_bias_y", "da_bias_z")
)


def _template_sequence():
    return sequence_from_tree(json.loads(scan_pulse_template_bytes().decode("utf-8")))


def test_the_mot_template_offers_the_three_bias_ports() -> None:
    """Ports come from the pulse's own declarations, nothing invented."""

    ports = scan_ports_for(_template_sequence())
    assert tuple(port.port for port in ports) == BIAS_PORTS
    for port in ports:
        assert port.unit == "", "a DAC code is dimensionless; the unit is empty"
        assert port.lo < 0 < port.hi, "the signed range brackets zero"


def test_scan_accepts_the_complete_document_saved_by_the_pulse_editor(
    tmp_path,
) -> None:
    tree = json.loads(scan_pulse_template_bytes().decode("utf-8"))
    tree["editor"] = {
        "visible_ports": None,
        "scan_source": "",
        "scan_rows": [],
        "scan_source_dirty": False,
        "scan_repeats": 0,
    }
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(tree), encoding="utf-8")

    assert load_scan_template(path).name == tree["name"]


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
        ports = scan_ports_for_devices(tunables)
        by_name = {port.port: port for port in ports}
        key = DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds"
        assert key in by_name
        port = by_name[key]
        assert port.label == "mot_camera.exposure_seconds"
        assert 0 < port.lo < port.hi
    finally:
        installation.close()
