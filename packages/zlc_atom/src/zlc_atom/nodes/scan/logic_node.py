"""The general scan Measurement: a plan over ports, watched on a live signal.

This supersedes the Pulse-API-scan plugin outright.  What that node hard-wired
-- API parameters only, a ScanTab program as the table's source -- is here one
port family and one way of writing a plan down.  The plan is pure data
(``ScanPlan``), so a notebook, the editor and a saved board all author the
same document, and the dataset that comes back carries the plan's axes by
name and unit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from zlc_pulse import PulseSequence, sequence_from_tree

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
    ResolvedWorkspaceResource,
    WorkspaceResourceSpec,
)

from .measurement import SCAN_OUTPUT, ScanMeasurement
from .plan import ScanPlan, bind_plan, scan_ports_for, scan_ports_for_devices


_PULSE_CONTRACT = "zlc.pulse/scan-template"


def _load_pulse_template(path: str | Path) -> PulseSequence:
    source = Path(path).expanduser().resolve()
    sequence = sequence_from_tree(json.loads(source.read_text(encoding="utf-8")))
    if not sequence.api_parameters:
        raise ValueError(
            "a scan template declares API parameters; this pulse declares none, "
            "so it offers nothing to scan"
        )
    if sequence.slots:
        raise ValueError(
            "a scan template cannot carry hardware scan slots; the plan is the "
            "only thing that says what varies"
        )
    return sequence


_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    _PULSE_CONTRACT,
    "pulses",
    (".json",),
    _load_pulse_template,
    argument_name="pulse_resource",
)


SCAN_SCHEMA = AuthoringSchema(
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
        # Repeats and samples both land on the dataset's repeat axis (size
        # repeats x samples): consecutive looks while the point stays applied
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
            "samples_per_point",
            "int",
            "Samples per point",
            1,
            minimum=1,
        ),
        # How a fresh value is taken after a point is applied.  This is the
        # operator's declaration about the SOURCE, not something the scan
        # guesses: a free-running monitor needs the straddler skipped, a
        # pulse-driven camera pays nothing.
        AuthoringField(
            "capture",
            "choice",
            "Capture",
            "skip_one",
            choices=(
                AuthoringChoice(
                    "skip_one", "Skip one, keep the next (free-running source)"
                ),
                AuthoringChoice(
                    "direct", "Keep every publication (pulse-driven source)"
                ),
            ),
        ),
        # Who moves the plan: the host, one compiled pulse per point, or the
        # BOARD, playing the whole plan from its hardware scan table --
        # seamless, frame order equal to point order by construction.
        AuthoringField(
            "advance",
            "choice",
            "Advance",
            "stepped",
            choices=(
                AuthoringChoice("stepped", "Stepped (host reloads per point)"),
                AuthoringChoice(
                    "streamed", "Streamed (board scan table, seamless)"
                ),
            ),
        ),
    )
)


def _parse_plan(payload: object) -> ScanPlan:
    if isinstance(payload, ScanPlan):
        return payload
    if isinstance(payload, Mapping):
        return ScanPlan.from_tree(payload)
    text = str(payload or "").strip()
    if not text:
        raise ValueError(
            'the scan plan is empty; it reads like '
            '{"axes": [{"port": "pulse:param:da_bias_x", "values": [-256, 0, 256]}]}'
        )
    return ScanPlan.from_tree(json.loads(text))


def _build(
    *,
    sequencer: object,
    signal_plane: object,
    source_signal: str,
    pulse_resource: ResolvedWorkspaceResource,
    plan: object,
    repeats: int = 1,
    samples_per_point: int = 1,
    capture: str = "skip_one",
    advance: str = "stepped",
    tunable_devices: Mapping | None = None,
) -> ScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != _PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    sequence = pulse_resource.value
    parsed = _parse_plan(plan)
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
    return ScanMeasurement(
        sequencer=sequencer,
        signal_plane=signal_plane,
        signal_name=str(source_signal),
        source_generation=publication.event_ref.generation,
        sequence=sequence,
        plan=parsed,
        ports=ports,
        repeats=int(repeats),
        samples_per_point=int(samples_per_point),
        capture=str(capture),
        advance=str(advance),
        tunables=dict(tunable_devices or {}),
    )


def _editor_factory(parent=None):
    from .editor import scan_plan_editor_factory

    return scan_plan_editor_factory(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "scan",
    NodeKind.MEASUREMENT,
    SCAN_SCHEMA,
    input_specs=(DatasetInputSpec("signal", None),),
    outputs=(OutputSpec(SCAN_OUTPUT.name, SCAN_OUTPUT.contract_id),),
    device_requirements=(
        DeviceRequirement(
            "sequencer.streamer",
            "sequencer",
            DeviceAccess.EXCLUSIVE,
        ),
    ),
    build=_build,
    ui_contributions=(_editor_factory,),
    workspace_resources=(_PULSE_RESOURCE,),
)


__all__ = ["LOGIC_NODE", "SCAN_SCHEMA"]
