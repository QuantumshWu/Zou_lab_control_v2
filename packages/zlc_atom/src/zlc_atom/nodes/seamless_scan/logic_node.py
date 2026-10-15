"""The seamless scan Measurement: a plan the BOARD plays from its scan table.

The board's axes play from one load, every point back to back; a manual,
device or API axis is moved by the host between fires, one fire per host
point, and a point that moves an API value is compiled and loaded again
first.  What the operator authors is the template, the plan, how many
whole sweeps and how many in-place shots per point. There is nothing to say about how a fresh value is taken: the fired cycle drives
the source, so its publications ARE the played rows, in order.

The loop this node offers is ``scan.SeamlessScanMeasurement``; this module
declares its form and published output.
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
    API_PARAM_FAMILY,
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    PublishedSignalSource,
    SCAN_PLAN_SELECTIONS,
    MANUAL_PARAM_FAMILY,
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    SEAMLESS_PULSE_RESOURCE,
    SeamlessScanMeasurement,
    ScanPlan,
    estimate_scan_time,
    api_scan_ports_for,
    bind_plan,
    hardware_scan_ports_for,
    api_overrides_from_authored,
    apply_api_overrides,
    plan_from_authored,
    scan_ports_for_devices,
    unscannable_api_ports,
)
from zlc_atom.nodes.scan.plan import device_port_parts, plan_input_rows


def _validate(values) -> None:
    """The plan and the API values read as the build reads them, while the
    draft is still being edited: an empty Values list, an axis on one port
    twice or every axis parked is said on the Edit page, not at Start."""

    plan_from_authored(values["plan"])
    api_overrides_from_authored(values.get("api_values"))


SEAMLESS_SCAN_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "",
            required=True,
            description="Pulse filename relative to workspace/pulses (for example imaging.json), or an absolute path.",
        ),
        AuthoringField(
            "plan",
            "text",
            "Scan plan",
            "",
            required=True,
            description=(
                'JSON text: {"axes":[{"port":"device:rf:frequency","values":[500,501,502],"unit":"MHz"}]}. '
                'Axes are outermost first; values is the explicit ordered numeric point list, not [start,stop,count]. '
                'Ports: device:<device_id>:<field>, pulse:param:<scan field_id>, api:<API field_id>, manual:<name>. '
                'Use device Control metadata or the Pulse field/binding catalog for IDs and units. '
                'Only authored axes are scanned; omitted Pulse scan slots retain their authored value. '
                'The optional Values input bank uses mode="values" and value_text="500,501,502" '
                'while values retains the independent Range bank. Manual axes require the human operator. '
                'An axis with "enabled": false is parked: kept in the plan and the run record, never '
                'bound, played or counted in the shape, its knob left as it is, whatever port it names. '
                'Park an axis the operator wrote instead of deleting it; at least one axis must be enabled.'
            ),
        ),
        AuthoringField(
            "acquisition_logic",
            "str",
            "Acquisition logic",
            "",
            description="Optional acquisition Logic to restart once and wait until ready before the scan.",
        ),
        # The pulse's API slots, set once for this run.  Only what differs
        # from the pulse is written here, so a recalibration that lands in
        # the workspace's current set is not outranked by a stale copy.
        AuthoringField(
            "api_values",
            "text",
            "API values",
            "",
            description=(
                "Optional fixed API overrides, one <field_id> = <number> per line in that field's "
                "authored unit; empty keeps the Pulse defaults. Keep lines you are not using: a "
                "value for a field the plan sweeps is overridden point by point and comes back "
                "when the axis is removed, and a name this Pulse does not declare is applied to "
                "nothing and listed in the run record as unapplied_api_values. A label two of the "
                "Pulse's fields share is refused at the draft; write the field id instead."
            ),
        ),
        # Repeats and shots both land on the dataset's repeat axis (size
        # repeats x shots): shots override the Pulse's Run repeats while the
        # row is held; repeats override complete scan-table sweeps. The
        # independent PulseBracket stays inside every shot.
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
    ),
    validator=_validate,
)


def _admit_resources(values, resources, installation) -> tuple[str, ...]:
    """The API values and every enabled axis against the knobs this draft
    can reach, so an API value the pulse cannot take (a label two of its
    fields answer to), an axis on a port its pulse does not offer, or one on
    a device not installed, is a draft issue the Edit page says and Start
    waits on.  A parked axis is not asked: it binds nothing; nor is a kept
    API value the pulse does not declare: it is applied to nothing.  A
    device field's range needs the device read, on its worker, so that one
    is still checked by the build."""

    from zlc_atom.install import tunable_devices

    plan = plan_from_authored(values["plan"])
    issues = []
    resource = resources.get("pulse_template")
    sequence = None
    if resource is not None:
        try:
            sequence, _unapplied = apply_api_overrides(
                resource.value, api_overrides_from_authored(values.get("api_values"))
            )
        except ValueError as error:
            # The build refuses these values before it binds anything, and
            # so does the draft: the axes are asked once they can be applied.
            issues.append(str(error))
    pulse_axes = tuple(
        axis for axis in plan.axes if axis.port.startswith((PULSE_PARAM_FAMILY, API_PARAM_FAMILY))
    )
    if sequence is not None and pulse_axes:
        try:
            bind_plan(ScanPlan(pulse_axes), (
                *hardware_scan_ports_for(sequence), *api_scan_ports_for(sequence),
            ))
        except ValueError as error:
            issues.append(f"{error}; switch that axis off to keep it without scanning it")
    installed = tunable_devices(installation)
    for axis in plan.axes:
        if axis.port.startswith(DEVICE_PARAM_FAMILY):
            key, _field = device_port_parts(axis.port)
            if key not in installed:
                issues.append(
                    f"device axis {axis.port!r} has no installed device {key!r} behind it; "
                    "switch that axis off to keep it without scanning it"
                )
    return tuple(issues)


def _build(
    *,
    sequencer: object,
    sequencer_key: str = "sequencer",
    signal_plane: object,
    source_signal: str,
    pulse_resource: ResolvedWorkspaceResource,
    plan: object,
    api_values: object = "",
    tunable_devices: object = None,
    device_labels: object = None,
    repeats: int = 1,
    shots_per_point: int = 1,
    acquisition_logic: str = "",
    restart_logic: object = None,
) -> SeamlessScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != SCAN_PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    # This run's own API values over whatever the pulse carries, before
    # anything reads what its slots hold.  The plan still overrides the
    # ones it sweeps; a value for a name this pulse does not declare is
    # kept in the draft and named in the run record as not applied.
    sequence, unapplied = apply_api_overrides(
        pulse_resource.value, api_overrides_from_authored(api_values)
    )
    parsed = plan_from_authored(plan)
    # A parked axis is the draft's, not the run's: named in the run record,
    # bound and played by nothing.
    parked = tuple(row for row in plan_input_rows(plan) if not row.get("enabled", True))
    # An operator's axis binds to no port at all: the run stops and asks
    # for it.  A device axis binds to an installed knob the HOST moves
    # between fires, and an API axis to a parameter of the pulse the host
    # writes and loads again between fires.  Everything left under them is
    # the table the board plays, if there are board axes.  A host-only plan
    # plays a fixed Pulse.
    bindable = tuple(
        axis
        for axis in parsed.axes
        if not axis.port.startswith(MANUAL_PARAM_FAMILY)
    )
    ports = bind_plan(
        ScanPlan(bindable),
        (
            *hardware_scan_ports_for(sequence),
            *api_scan_ports_for(sequence),
            *scan_ports_for_devices(
                tunable_devices, units={axis.port: axis.unit for axis in parsed.axes},
                device_labels=device_labels,
            ),
        ),
        withheld=unscannable_api_ports(sequence),
    ) if bindable else ()

    return SeamlessScanMeasurement(
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        source=PublishedSignalSource(signal_plane, source_signal),
        sequence=sequence,
        pulse_path=pulse_resource.path,
        plan=parsed,
        ports=ports,
        tunables=tunable_devices,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        acquisition_logic=acquisition_logic,
        restart_logic=restart_logic,
        unapplied_api_values=unapplied,
        parked_axes=parked,
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import ScanPlanEditor

    # The board axes are the template's own hardware slots: the board plays
    # the slots selected by the plan; omitted slots keep their Pulse values.
    # Manual, device AND API axes are offered because this node can stop
    # between fires -- for a hand on a thumbscrew, a tune() call on an
    # installed device or a new value in the pulse alike.
    return ScanPlanEditor(parent)


def _estimate_run(values, devices, resources):
    """Preview this Scan's current saved inputs on a worker, never Load/Fire."""

    resource = resources.get("pulse_template")
    if resource is None:
        return {"status": "unknown", "reason": "Choose a resolvable Pulse template first."}
    # Estimation is an explicit file boundary, like Start/Refresh, not a poll.
    # Reuse the same resource resolver, including its Subpulse resolution.
    current = SEAMLESS_PULSE_RESOURCE.resolve(resource.path)
    plan = plan_from_authored(values.get("plan", ""))
    sequencer = devices.get("sequencer")
    board = None if sequencer is None else sequencer.describe()
    config = (
        sequencer.preview_config_values()
        if sequencer is not None and current.value.config_enabled else {}
    )
    result = estimate_scan_time(
        current.value, plan,
        geometry=None if board is None else board.geometry,
        clock_hz=None if board is None else board.clock_hz,
        config_values=config,
        api_values=values.get("api_values", ""),
        repeats=values.get("repeats", 1),
        shots_per_point=values.get("shots_per_point", 1),
    )
    result["pulse_path"] = str(current.path)
    result["config_source"] = "" if sequencer is None else sequencer.config_source
    return result


LOGIC_NODE = LogicNodeDescriptor(
    "seamless_scan",
    NodeKind.MEASUREMENT,
    SEAMLESS_SCAN_SCHEMA,
    description=(
        "Collect a selected signal, including processor or fit-parameter outputs, over "
        "Pulse scan slots, device parameters and/or manual axes. Publishes scan with "
        "the original cell data and explicit scan/repeat coordinates for native "
        "Curve/Image/Facet plots and fitting. This measurement starts the selected "
        "acquisition logic, waits for ready and owns Pulse Fire throughout the scan."
    ),
    acquisition_input="acquisition_logic",
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
    admit_resources=_admit_resources,
    estimate_run=_estimate_run,
    # A region drawn on this scan's own plot is a statement about
    # what to sweep next, in the axes the picture is drawn in.
    selection_mappings=SCAN_PLAN_SELECTIONS,
    ui_contributions=(_editor_factory,),
    workspace_resources=(SEAMLESS_PULSE_RESOURCE,),
)


__all__ = ["LOGIC_NODE", "SEAMLESS_SCAN_SCHEMA"]
