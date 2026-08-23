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
    SCAN_PLAN_SELECTIONS,
    DEVICE_PARAM_FAMILY,
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    SCAN_PULSE_RESOURCE,
    PublishedSignalSource,
    SeamlessScanMeasurement,
    bind_plan,
    plan_from_authored,
    scan_ports_for,
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
            "mot_field_template.json",
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
        # repeats x shots): in-place looks while the board holds a point
        # versus whole-table sweeps.  Physically different, structurally the
        # same fact -- the same conditions, again.
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
    signal_plane: object,
    source_signal: str,
    pulse_resource: ResolvedWorkspaceResource,
    plan: object,
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
    # A device knob is moved by a host call between cycles, which is exactly
    # what a single fired table does not have.  Refused here, where the plan
    # meets the bench, and named to the node that can run it.
    for axis in parsed.axes:
        if axis.port.startswith(DEVICE_PARAM_FAMILY):
            raise ValueError(
                f"the board cannot advance {axis.port!r} between the cycles of "
                "one fired scan table; a device port is moved by the host, so "
                "this plan belongs to the stepped_scan node"
            )
    ports = bind_plan(parsed, scan_ports_for(sequence))
    publication = signal_plane.latest_publication(source_signal)
    if publication is None or not signal_plane.is_generation_live(source_signal):
        raise ValueError(
            "the scan watches a LIVE signal for each point's value, and "
            f"{source_signal!r} is not live on this bench"
        )
    return SeamlessScanMeasurement(
        sequencer=sequencer,
        source=PublishedSignalSource(
            signal_plane, str(source_signal), publication.event_ref.generation
        ),
        sequence=sequence,
        plan=parsed,
        ports=ports,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        settle_seconds=float(settle_seconds),
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    # No device ports: this node would refuse one, so its form never offers it.
    return scan_plan_editor_factory(parent, device_ports=False)


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
    workspace_resources=(SCAN_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_SETTLE_SECONDS", "LOGIC_NODE", "SEAMLESS_SCAN_SCHEMA"]
