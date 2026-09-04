"""The stepped scan Measurement: a plan the HOST advances, point by point.

What the operator authors is the template, the plan, how many whole sweeps
and how many shots per point, how long the pulse stays stopped while the
bench settles, how long a free-running source waits after On -- and the one
thing no machine can answer for them: what gates a publication on THIS bench,
the fired cycle or the source's own clock.
"""

from __future__ import annotations

from collections.abc import Mapping

from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
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
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    STEPPED_PULSE_RESOURCE,
    bind_plan,
    api_overrides_from_authored,
    apply_api_overrides,
    plan_from_authored,
    scan_ports_for,
    scan_ports_for_devices,
)

from .measurement import SteppedScanMeasurement


#: How long the pulse stays stopped before each point is applied.  A tenth of
#: a second is what the bench needs to reach the state a point starts from; it
#: is authored because only the operator knows their apparatus.
DEFAULT_SETTLE_SECONDS = 0.1
DEFAULT_FREE_RUN_DELAY_SECONDS = 0.0


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
        # The pulse's API slots, set once for this run.  Only what differs
        # from the pulse is written here, so a recalibration that lands in
        # the workspace's current set is not outranked by a stale copy.
        AuthoringField(
            "api_values",
            "text",
            "API values",
            "",
        ),
        # Repeats and shots both land on the dataset's repeat axis (size
        # repeats x shots): shots override Run repeats for one host-applied
        # point; repeats walk the whole host-owned plan again. The independent
        # PulseBracket stays inside every shot.
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
                    "pulse_gated", "Pulse-driven source (one pulse repeat per shot)"
                ),
            ),
        ),
        AuthoringField(
            "free_run_delay_seconds",
            "float",
            "Free-run delay (s)",
            DEFAULT_FREE_RUN_DELAY_SECONDS,
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
    api_values: object = "",
    repeats: int = 1,
    shots_per_point: int = 1,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    gating: str = "sw_gated",
    free_run_delay_seconds: float = DEFAULT_FREE_RUN_DELAY_SECONDS,
    tunable_devices: Mapping | None = None,
) -> SteppedScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != SCAN_PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    # This run's own API values over whatever the pulse carries; the plan
    # still overrides the ones it sweeps.
    sequence = apply_api_overrides(
        pulse_resource.value, api_overrides_from_authored(api_values)
    )
    parsed = plan_from_authored(plan)
    # One vocabulary, two projectors: the pulse offers its API parameters,
    # the bench's tunable devices offer their runtime knobs.  Binding sees
    # the union, so an axis is refused by name against everything offered.
    ports = bind_plan(
        parsed,
        scan_ports_for(sequence) + scan_ports_for_devices(tunable_devices),
    )

    return SteppedScanMeasurement(
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        source=watched_signal_source(signal_plane, source_signal),
        sequence=sequence,
        plan=parsed,
        ports=ports,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        settle_seconds=float(settle_seconds),
        gating=str(gating),
        free_run_delay_seconds=float(free_run_delay_seconds),
        tunables=dict(tunable_devices or {}),
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    return scan_plan_editor_factory(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "stepped_scan",
    NodeKind.MEASUREMENT,
    STEPPED_SCAN_SCHEMA,
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
    workspace_resources=(STEPPED_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_SETTLE_SECONDS", "LOGIC_NODE", "STEPPED_SCAN_SCHEMA"]
