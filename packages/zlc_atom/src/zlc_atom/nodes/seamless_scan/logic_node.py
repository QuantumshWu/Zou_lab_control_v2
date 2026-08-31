"""The seamless scan Measurement: a plan the BOARD plays from its scan table.

One load, one fire, every point back to back.  What the operator authors is
the template, the plan, how many whole sweeps and how many in-place shots per
point, and how long the pulse stays stopped before the table starts.  There
is nothing to say about how a fresh value is taken: the fired cycle drives
the source, so its publications ARE the played rows, in order.

The loop this node offers is ``scan.SeamlessScanMeasurement``: it moved into
the library the day a second consumer appeared (the temperature Task), and
what stays here is what only this node knows -- its form, and that the frames
it takes are published as the scan itself.
"""

from __future__ import annotations

from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    ResolvedWorkspaceResource,
)
from zlc_atom.nodes.scan import (
    watched_signal_source,
    SCAN_PLAN_SELECTIONS,
    MANUAL_PARAM_FAMILY,
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    SEAMLESS_PULSE_RESOURCE,
    SeamlessScanMeasurement,
    ScanPlan,
    bind_plan,
    hardware_scan_ports_for,
    plan_from_authored,
    scan_ports_for_devices,
    split_outer_axes,
)


#: How long the pulse stays stopped before the table plays.  A tenth of a
#: second is what the bench needs to reach the state the first point starts
#: from; it is authored because only the operator knows their apparatus.
DEFAULT_SETTLE_SECONDS = 0.1


SEAMLESS_SCAN_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "",
            required=True,
        ),
        AuthoringField(
            "plan",
            "text",
            "Scan plan",
            "",
            required=True,
        ),
        # Repeats and shots both land on the dataset's repeat axis (size
        # repeats x shots): shots play inside one point as the pulse's
        # outermost repeat bracket; repeats are whole-table sweeps.
        # Physically different, structurally the same fact -- the same
        # conditions, again.
        AuthoringField(
            "repeats",
            "int",
            "Repeats (full sweeps)",
            1,
            minimum=1,
        ),
        AuthoringField(
            "shots_per_point",
            "int",
            "Shots per point",
            1,
            minimum=1,
        ),
        AuthoringField(
            "settle_seconds",
            "float",
            "Settle time (s)",
            DEFAULT_SETTLE_SECONDS,
            minimum=0.0,
        ),
    )
)


def _build(
    *,
    sequencer: object,
    sequencer_key: str = "sequencer",
    signal_plane: object,
    source_signal: str,
    pulse_resource: ResolvedWorkspaceResource,
    plan: object,
    tunable_devices: object = None,
    repeats: int = 1,
    shots_per_point: int = 1,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
) -> SeamlessScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != SCAN_PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    sequence = pulse_resource.value
    parsed = plan_from_authored(plan)
    # An operator's axis binds to no port at all: the run stops and asks
    # for it.  A device axis binds to an installed knob the HOST moves
    # between fires.  Everything left under them is the table the board
    # plays -- split_outer_axes owns the ordering law and its refusals.
    split_outer_axes(parsed)
    bindable = tuple(
        axis
        for axis in parsed.axes
        if not axis.port.startswith(MANUAL_PARAM_FAMILY)
    )
    ports = bind_plan(
        ScanPlan(bindable),
        (
            *hardware_scan_ports_for(sequence),
            *scan_ports_for_devices(tunable_devices),
        ),
    ) if bindable else ()

    return SeamlessScanMeasurement(
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        source=watched_signal_source(signal_plane, source_signal),
        sequence=sequence,
        plan=parsed,
        ports=ports,
        tunables=tunable_devices,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        settle_seconds=float(settle_seconds),
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    # The board axes are the template's own hardware slots: the board plays
    # exactly what the template scans.  Manual AND device axes are offered
    # because this node can stop between fires -- for a hand on a
    # thumbscrew or a tune() call on an installed device alike.
    return scan_plan_editor_factory(
        parent, device_ports=True, hardware_slots=True, manual_axes=True
    )


LOGIC_NODE = LogicNodeDescriptor(
    "seamless_scan",
    NodeKind.MEASUREMENT,
    SEAMLESS_SCAN_SCHEMA,
    input_specs=(DatasetInputSpec("signal", None, "exact"),),
    outputs=(SCAN_OUTPUT,),
    # A scan is one measurement per point, so its plot is one cell per
    # point.  A plan that leaves nothing to face -- one axis over one
    # number per point -- opens as the curve it is.
    node_previews=(NodePreviewSpec(SCAN_OUTPUT, "facet_grid"),),
    device_requirements=(
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
    ),
    build=_build,
    # A region drawn on this scan's own plot is a statement about
    # what to sweep next, in the axes the picture is drawn in.
    selection_mappings=SCAN_PLAN_SELECTIONS,
    ui_contributions=(_editor_factory,),
    workspace_resources=(SEAMLESS_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_SETTLE_SECONDS", "LOGIC_NODE", "SEAMLESS_SCAN_SCHEMA"]
