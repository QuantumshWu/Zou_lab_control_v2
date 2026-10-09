"""One imaging Pulse and an explicitly authored SLM rearrangement Task."""

from __future__ import annotations

from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.slm.solver import (
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    load_science_context,
    load_target,
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
    REARRANGEMENT_ARTIFACT_CONTRACT,
    SlmRearrangementTask,
    rearrangement_outputs,
)


_SCIENCE_CONTEXT_CODEC = ArtifactCodec(
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    "SLM Science Contexts (*.npz)",
    (".npz",),
    load_science_context,
)
_TARGET_CODEC = ArtifactCodec(
    "zlc.slm.target", "SLM Targets (*.json)", (".json",), load_target,
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
        AuthoringField("exposure_seconds", "float", "Camera exposure", .005, minimum=1e-9, unit="s",
                       description="Authored camera integration; the Task does not compare it with Pulse Period lengths."),
        AuthoringField("frame_mode", "choice", "Frame count", "fixed",
                       choices=(AuthoringChoice("fixed", "Fixed frames"),
                                AuthoringChoice("camera_step", "Maximum camera step"))),
        AuthoringField("motion_frames", "int", "Total frames", 16, minimum=1, maximum=256,
                       enabled_when=("frame_mode", ("fixed",)),
                       description="Total displayed maps, including one initial map removing unused traps (when needed) and the destination; excludes the already displayed source. Movement follows without an additional hold."),
        AuthoringField("max_camera_step", "float", "Maximum camera step", 1., minimum=1e-9, unit="pixel",
                       enabled_when=("frame_mode", ("camera_step",)),
                       description="Maximum two-dimensional Euclidean movement per frame in original camera sensor pixels. The actual frame count is determined after occupancy and path planning."),
        AuthoringField("frame_rate_hz", "float", "Display frame rate", 60., minimum=1e-9, unit="Hz",
                       description="Requested display cadence, not a measured liquid-crystal response."),
        AuthoringField(
            "nominal_playback_seconds", "float", "Display duration", None,
            derived=True, unit="s",
            description=(
                "Total frame count divided by the SLM frame rate. Automatic frame count remains pending until occupancy and path planning. "
                "GPU computation, upload, preparation and any additional optical settling are excluded."
            ),
        ),
        AuthoringField("phase_method", "choice", "Phase method", "lpi",
                       choices=(AuthoringChoice("iterative", "Iterative"), AuthoringChoice("lpi", "LPI (phase interpolation)")),
                       description="LPI interpolates site phases, holds each frame's phases fixed during amplitude balancing, and records the iteration count. Both methods enforce the intensity tolerance."),
        AuthoringField("minimum_separation", "float", "Minimum trap distance (Fourier)", 15., minimum=0., unit="pixel",
                       description="Hard continuous separation of moving and stationary traps, including surplus atoms while they fade. Uses the same units as SLM Target grid spacing (for example, 25 gap); not camera pixels."),
        AuthoringField("intensity_error_percent", "float", "Intensity tolerance (%)", 1., minimum=0.,
                       description="Maximum/minimum intensity after division by requested site weights, minus one. This is not a bound on absolute trap-depth change or atom loss."),
        AuthoringField("save_phase_sequence", "bool", "Save phase sequence", True),
    ),
    validator=_validate_rearrangement,
)


def _resolve_defaults(values, resources):
    del resources
    if values.get("frame_mode", "fixed") == "camera_step":
        return {"nominal_playback_seconds": None}
    try:
        frames = int(values.get("motion_frames", 16))
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
    end_target: ResolvedArtifact | None = None,
    save_figure_artifact: object = None,
    recording_frames: int | None = None,
    schema: AuthoringSchema = SLM_REARRANGEMENT_SCHEMA,
    **values: object,
) -> SlmRearrangementTask:
    if recording_frames is not None:
        values["recording_frames"] = recording_frames
    authored = schema.project_values(values)
    recording = "recording_frames" in authored
    if not isinstance(calibration, ResolvedArtifact) or not isinstance(calibration.value, TrapCalibration):
        raise TypeError("calibration must be a resolved calibration artifact")
    if not isinstance(science_context, ResolvedArtifact):
        raise TypeError("science_context must be a resolved Science Context artifact")
    if not isinstance(pulse_resource, ResolvedWorkspaceResource) or not isinstance(pulse_resource.value, PulseSequence):
        raise TypeError("pulse_resource must be a resolved imaging pulse")
    period_ids = tuple(period.period_id for period in pulse_resource.value.periods)
    for key in (("before_period",) if recording else ("before_period", "after_period")):
        if authored[key] not in period_ids:
            raise ValueError(f"{key} names a Period absent from the selected imaging pulse")
    if not recording and period_ids.index(authored["before_period"]) >= period_ids.index(authored["after_period"]):
        raise ValueError("Before imaging Period must precede after imaging Period")
    authored.setdefault("after_period", "")
    destination = None
    if end_target is not None:
        destination, kind = end_target.value
        if kind != "spots":
            raise ValueError("The end Target must be a spots Target")
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
        target_intensity=destination,
        target_path=None if end_target is None else end_target.path,
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        save_figure_artifact=save_figure_artifact,
        intensity_tolerance=1.0 + authored["intensity_error_percent"] / 100.0,
        **{key: value for key, value in authored.items()
           if key not in {"pulse_template", "nominal_playback_seconds", "intensity_error_percent"}},
    )


def _rearrangement_editor_factory(parent=None, *, schema=SLM_REARRANGEMENT_SCHEMA):
    """Project the selected Pulse's Periods into the leaf's Fluent form."""
    from collections.abc import Mapping
    from dataclasses import replace
    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.form import FluentParameterForm, FormChoice, FormSpec
    from zlc_ui.fluent import FluentSectionLabel, scaled_px

    class RearrangementForm(QtWidgets.QWidget):
        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = tuple(
            field.name for field in schema.fields
            if field.value_type != "resource"
        )

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(scaled_px(6,minimum=4))
            self._groups = (
                ("Target grid", ("target_rows","target_columns")),
                ("Imaging", ("before_period","after_period","recording_frames","exposure_seconds")),
                ("Movement", ("frame_mode","motion_frames","max_camera_step","frame_rate_hz","nominal_playback_seconds")),
                ("Quality and output", ("phase_method","minimum_separation","intensity_error_percent","save_phase_sequence")),
            )
            self._forms = {}
            for title, keys in self._groups:
                layout.addWidget(FluentSectionLabel(title))
                form = FluentParameterForm(FormSpec(()), parent=self)
                form.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
                layout.addWidget(form)
                self._forms[title] = form
                form.changed.connect(self._value_changed)
                form.value_normalized.connect(lambda key: self._value_changed(key, normalized=True))

        @property
        def spec(self):
            return FormSpec(tuple(field for form in self._forms.values() for field in form.spec.fields))

        def _form_for(self,key):
            return next(self._forms[title] for title,keys in self._groups if key in keys)

        def read_value(self,key):
            return self._form_for(key).read_value(key)

        def widget_for(self,key):
            return self._form_for(key).widget_for(key)

        def set_label_width(self,width):
            for form in self._forms.values():
                form.set_label_width(width)

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
                    FormChoice(sequence.period_label(period.period_id), period.period_id)
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
            for title, keys in self._groups:
                selected = tuple(field for field in fields if field.key in keys)
                self._forms[title].reconcile(FormSpec(selected), {field.key:values[field.key] for field in selected})
            self.widget_for("nominal_playback_seconds").setPlaceholderText("Pending occupancy")
            use_grid = not str((projection.get("artifact_values") or {}).get("end_target_path", "")).strip()
            for key in ("target_rows", "target_columns"):
                self.widget_for(key).setEnabled(use_grid)
                self.widget_for(key).setToolTip("Used when End Target is blank; otherwise the selected Target supplies the destinations.")

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
        ArtifactInputSpec("end_target_path", "End Target (blank = grid)", _TARGET_CODEC,
                          required=False, argument_name="end_target"),
    ),
    declare_outputs=lambda values, devices: rearrangement_outputs(str(values.get("frame_mode", "fixed"))),
    node_previews=(),
    declare_previews=lambda values, devices: (
        NodePreviewSpec(BEFORE_FRAME_OUTPUT, "image", overlay=BEFORE_OCCUPIED_OUTPUT),
        NodePreviewSpec(AFTER_FRAME_OUTPUT, "image", overlay=AFTER_OCCUPIED_OUTPUT),
        NodePreviewSpec(PHASE_OUTPUT, "image"),
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
