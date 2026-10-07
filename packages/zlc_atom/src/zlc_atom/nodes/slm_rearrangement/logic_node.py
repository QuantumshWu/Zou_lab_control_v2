"""One imaging Pulse and an explicitly authored SLM rearrangement Task."""

from __future__ import annotations

from zlc_plot import AxisRef
from zlc_plot.semantics import fate_field_name
from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.slm.solver import (
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    load_science_context,
)
from zlc_atom.nodes._framework.descriptor import (
    ArtifactCodec,
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    ResolvedArtifact,
    ResolvedWorkspaceResource,
    WorkspaceResourceSpec,
)
from zlc_atom.nodes.calibration import CALIBRATION_ARTIFACT_CODEC, TrapCalibration
from zlc_atom.nodes.calibration.pulse import load_calibration_pulse_template

from .task import (
    AFTER_FRAME_OUTPUT,
    AFTER_OCCUPIED_OUTPUT,
    BEFORE_FRAME_OUTPUT,
    BEFORE_OCCUPIED_OUTPUT,
    PHASE_OUTPUT,
    QUALITY_OUTPUT,
    REARRANGEMENT_ARTIFACT_CONTRACT,
    TRAJECTORY_OUTPUT,
    SlmRearrangementTask,
)


_SCIENCE_CONTEXT_CODEC = ArtifactCodec(
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    "SLM Science Contexts (*.npz)",
    (".npz",),
    load_science_context,
)
_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse/slm-rearrangement",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)


def _validate_rearrangement(values):
    if values["before_period"] == values["after_period"]:
        raise ValueError("Before and after imaging must select different Periods")


SLM_REARRANGEMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField("pulse_template", "resource", "Imaging pulse", "", required=True),
        AuthoringField("target_rows", "int", "Target rows", 3, minimum=1),
        AuthoringField("target_columns", "int", "Target columns", 3, minimum=1),
        AuthoringField("before_period", "text", "Before imaging Period", "", required=True),
        AuthoringField("after_period", "text", "After imaging Period", "", required=True),
        AuthoringField("exposure_seconds", "float", "Camera exposure seconds", .005, minimum=1e-9),
        AuthoringField("motion_frames", "int", "Motion frames", 16, minimum=1, maximum=256),
        AuthoringField("frame_rate_hz", "float", "SLM frame rate Hz", 60., minimum=1e-9),
        AuthoringField("ramp_frames", "int", "Removal ramp frames", 2, minimum=1, maximum=16),
        AuthoringField(
            "nominal_playback_seconds", "float", "Nominal SLM playback seconds", None,
            derived=True,
            description=(
                "Removal and motion frame count divided by the SLM frame rate. "
                "GPU computation, upload, preparation and any additional optical settling are excluded."
            ),
        ),
        AuthoringField("matching_radius", "int", "Matching radius Fourier bins", 80, minimum=0),
        AuthoringField("minimum_separation", "float", "Minimum separation Fourier bins", 4.5, minimum=0.),
        AuthoringField("intensity_tolerance", "float", "Maximum weighted intensity ratio", 1.01, minimum=1.),
        AuthoringField("dark_tolerance", "float", "Maximum dark / bright intensity ratio", .01, minimum=0.),
        AuthoringField("save_phase_sequence", "bool", "Save phase sequence", True),
    ),
    validator=_validate_rearrangement,
)


def _resolve_defaults(values, resources):
    del resources
    try:
        frames = int(values.get("motion_frames", 16)) + int(values.get("ramp_frames", 2))
        rate = float(values.get("frame_rate_hz", 60.))
    except (TypeError, ValueError):
        return {}
    if rate <= 0:
        return {}
    return {"nominal_playback_seconds": frames / rate}


def _build(
    *,
    camera: object,
    camera_key: str,
    sequencer: object,
    sequencer_key: str,
    slm: object,
    slm_key: str,
    signal_plane: object,
    calibration: ResolvedArtifact,
    science_context: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
    save_figure_artifact: object = None,
    **values: object,
) -> SlmRearrangementTask:
    authored = SLM_REARRANGEMENT_SCHEMA.project_values(values)
    if not isinstance(calibration, ResolvedArtifact) or not isinstance(calibration.value, TrapCalibration):
        raise TypeError("calibration must be a resolved calibration artifact")
    if not isinstance(science_context, ResolvedArtifact):
        raise TypeError("science_context must be a resolved Science Context artifact")
    if not isinstance(pulse_resource, ResolvedWorkspaceResource) or not isinstance(pulse_resource.value, PulseSequence):
        raise TypeError("pulse_resource must be a resolved imaging pulse")
    period_ids = tuple(period.period_id for period in pulse_resource.value.periods)
    for key in ("before_period", "after_period"):
        if authored[key] not in period_ids:
            raise ValueError(f"{key} names a Period absent from the selected imaging pulse")
    if period_ids.index(authored["before_period"]) >= period_ids.index(authored["after_period"]):
        raise ValueError("Before imaging Period must precede after imaging Period")
    return SlmRearrangementTask(
        camera=camera,
        camera_key=camera_key,
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        slm=slm,
        slm_key=slm_key,
        signal_plane=signal_plane,
        calibration=calibration.value,
        calibration_path=calibration.path,
        science_context=science_context.value,
        science_context_path=science_context.path,
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        save_figure_artifact=save_figure_artifact,
        **{key: value for key, value in authored.items()
           if key not in {"pulse_template", "nominal_playback_seconds"}},
    )


def _rearrangement_editor_factory(parent=None):
    """Project the selected Pulse's Periods into the leaf's Fluent form."""
    from collections.abc import Mapping
    from dataclasses import replace
    from PyQt5 import QtCore
    from zlc_ui.form import FluentParameterForm, FormChoice, FormSpec

    class RearrangementForm(FluentParameterForm):
        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = tuple(
            field.name for field in SLM_REARRANGEMENT_SCHEMA.fields
            if field.value_type != "resource"
        )

        def __init__(self, parent=None):
            super().__init__(FormSpec(()), parent=parent)
            self.changed.connect(self._value_changed)
            self.value_normalized.connect(lambda key: self._value_changed(key, normalized=True))

        def _value_changed(self, key, *, normalized=False):
            try:
                value = self.read_value(key)
            except (TypeError, ValueError):
                return
            patch = {"values": {key: value}}
            if normalized:
                patch["normalized"] = True
            self.draft_changed.emit(patch)

        def update_projection(self, projection):
            resources = projection.get("workspace_resources") or {}
            resource = resources.get("pulse_template") if isinstance(resources, Mapping) else None
            sequence = getattr(resource, "value", None)
            choices = (FormChoice("Select Period", ""),)
            if isinstance(sequence, PulseSequence):
                choices += tuple(
                    FormChoice(period.name or period.period_id, period.period_id)
                    for period in sequence.periods
                )
            values = projection.get("form_values") or {}
            fields = []
            for field in projection["form_spec"].fields:
                if field.key not in self.managed_fields:
                    continue
                if field.key in {"before_period", "after_period"}:
                    selected = str(values.get(field.key) or "")
                    offered = choices
                    if selected and not any(choice.value == selected for choice in choices):
                        offered += (FormChoice(f"Unavailable: {selected}", selected),)
                    field = replace(field, kind="choice", default=selected, choices=offered)
                fields.append(field)
            self.reconcile(FormSpec(tuple(fields)), {field.key: values[field.key] for field in fields})

        def set_mutation_enabled(self, enabled):
            self.setEnabled(enabled)

    return RearrangementForm(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "slm_rearrangement",
    NodeKind.TASK,
    SLM_REARRANGEMENT_SCHEMA,
    input_specs=(
        ArtifactInputSpec("calibration_path", "Calibration", CALIBRATION_ARTIFACT_CODEC,
                          argument_name="calibration"),
        ArtifactInputSpec("science_context_path", "SLM Science Context", _SCIENCE_CONTEXT_CODEC,
                          argument_name="science_context"),
    ),
    outputs=(BEFORE_FRAME_OUTPUT, BEFORE_OCCUPIED_OUTPUT, AFTER_FRAME_OUTPUT,
             AFTER_OCCUPIED_OUTPUT, PHASE_OUTPUT, TRAJECTORY_OUTPUT, QUALITY_OUTPUT),
    node_previews=(
        NodePreviewSpec(BEFORE_FRAME_OUTPUT, "image", overlay=BEFORE_OCCUPIED_OUTPUT),
        NodePreviewSpec(AFTER_FRAME_OUTPUT, "image", overlay=AFTER_OCCUPIED_OUTPUT),
        NodePreviewSpec(PHASE_OUTPUT, "image"),
        NodePreviewSpec(
            TRAJECTORY_OUTPUT, "curve",
            semantic={
                fate_field_name(AxisRef.point("slm_rearrangement.frame")): "x",
                fate_field_name(AxisRef.cell_data("slm_rearrangement.target_site")): "group",
            },
        ),
    ),
    artifact_outputs=(ArtifactOutputSpec("artifact_path", REARRANGEMENT_ARTIFACT_CONTRACT),),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
        DeviceRequirement("slm.phase", "slm", (
            "phase", "wavelength_nm", "display_name", "width", "height",
            "correction_path", "flip_x", "flip_y",
        )),
    ),
    build=_build,
    workspace_resources=(_PULSE_RESOURCE,),
    ui_contributions=(_rearrangement_editor_factory,),
    resolve_defaults=_resolve_defaults,
)


__all__ = ["LOGIC_NODE", "SLM_REARRANGEMENT_SCHEMA"]
