"""The stepped scan Measurement: a plan the HOST advances, point by point.

What the operator authors is the template, the plan, how many whole sweeps
and how many shots per point, how long the pulse stays stopped while the
bench settles -- and the one thing no machine can answer for them: what gates
a publication on THIS bench, the fired cycle or the source's own clock.
"""

from __future__ import annotations

from collections.abc import Mapping

from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    OutputSpec,
    ResolvedWorkspaceResource,
)
from zlc_atom.nodes.scan import (
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    SCAN_PULSE_RESOURCE,
    bind_plan,
    plan_from_authored,
    scan_ports_for,
    scan_ports_for_devices,
)

from .measurement import SteppedScanMeasurement


#: How long the pulse stays stopped before each point is applied.  A tenth of
#: a second is what the bench needs to reach the state a point starts from; it
#: is authored because only the operator knows their apparatus.
DEFAULT_SETTLE_SECONDS = 0.1


STEPPED_SCAN_SCHEMA = AuthoringSchema(
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
        # repeats x shots): consecutive looks while the point stays applied
        # versus whole-plan sweeps.  Physically different, structurally the
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
        # What gates a publication on this bench.  The operator's declaration
        # about their SOURCE, never a probe of the board: a free-running
        # monitor needs the straddler skipped, a pulse-driven camera does not.
        AuthoringField(
            "gating",
            "choice",
            "Gating",
            "sw_gated",
            choices=(
                AuthoringChoice(
                    "sw_gated", "Free-running source (skip the straddling shot)"
                ),
                AuthoringChoice(
                    "pulse_gated", "Pulse-driven source (one fired cycle per shot)"
                ),
            ),
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
    gating: str = "sw_gated",
    tunable_devices: Mapping | None = None,
) -> SteppedScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != SCAN_PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    sequence = pulse_resource.value
    parsed = plan_from_authored(plan)
    # One vocabulary, two projectors: the pulse offers its API parameters,
    # the bench's tunable devices offer their runtime knobs.  Binding sees
    # the union, so an axis is refused by name against everything offered.
    ports = bind_plan(
        parsed,
        scan_ports_for(sequence) + scan_ports_for_devices(tunable_devices),
    )
    publication = signal_plane.latest_publication(source_signal)
    if publication is None or not signal_plane.is_generation_live(source_signal):
        raise ValueError(
            "the scan watches a LIVE signal for each point's value, and "
            f"{source_signal!r} is not live on this bench"
        )
    return SteppedScanMeasurement(
        sequencer=sequencer,
        signal_plane=signal_plane,
        signal_name=str(source_signal),
        source_generation=publication.event_ref.generation,
        sequence=sequence,
        plan=parsed,
        ports=ports,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        settle_seconds=float(settle_seconds),
        gating=str(gating),
        tunables=dict(tunable_devices or {}),
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    return scan_plan_editor_factory(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "stepped_scan",
    NodeKind.MEASUREMENT,
    STEPPED_SCAN_SCHEMA,
    input_specs=(DatasetInputSpec("signal", None),),
    outputs=(OutputSpec(SCAN_OUTPUT.name, SCAN_OUTPUT.contract_id),),
    node_previews=(NodePreviewSpec(SCAN_OUTPUT.name),),
    device_requirements=(
        DeviceRequirement(
            "sequencer.streamer",
            "sequencer",
            DeviceAccess.EXCLUSIVE,
        ),
    ),
    build=_build,
    ui_contributions=(_editor_factory,),
    workspace_resources=(SCAN_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_SETTLE_SECONDS", "LOGIC_NODE", "STEPPED_SCAN_SCHEMA"]
