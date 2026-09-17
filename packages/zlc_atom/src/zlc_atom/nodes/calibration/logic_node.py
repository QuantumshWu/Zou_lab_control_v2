"""Discoverable calibration task with one live preview and saved results."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.camera.photoelectrons import (
    PHOTOELECTRONS,
    photoelectron_switch,
    resolve_photoelectron_availability,
)
from zlc_atom.nodes._framework.descriptor import (
    ArtifactOutputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    ResolvedWorkspaceResource,
    NodePreviewSpec,
    WorkspaceResourceSpec,
)
from zlc_pulse import PulseSequence

from .artifact import CALIBRATION_ARTIFACT_CODEC
from .calibration import ReadoutModelKind
from .outputs import CAPTURE_PREVIEW_DECLARATION, SITE_REVIEW_DECLARATION
from .pulse import load_calibration_pulse_template
from .task import (
    FRAMES_FROM_CAMERA,
    FRAMES_FROM_FOLDER,
    CalibrationRequest,
    CalibrationTask,
)


_CALIBRATION_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse/calibration",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)
def _validate_calibration(values: dict[str, object]) -> None:
    # The same relation the request enforces, so the form never accepts a
    # draft the task will refuse: the readout frame is recognised in the
    # compiled program by being the SHORT window, so equal is not enough.
    if float(values["readout_exposure_seconds"]) >= float(
        values["reference_exposure_seconds"]
    ):
        raise ValueError("readout exposure must be shorter than the reference exposure")


CALIBRATION_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Calibration pulse template",
            # No default file name: pulses live in the workspace's pulse
            # folder and the operator names the one this node runs.  A
            # name shipped inside the logic layer was a pulse document in
            # a place pulse documents do not live, and it went stale the
            # moment anybody edited theirs.
            "",
            required=True,
        ),
        AuthoringField("repeats", "int", "Samples", 200, minimum=1),
        AuthoringField(
            "reference_exposure_seconds",
            "float",
            "Reference exposure seconds",
            0.02,
            minimum=1e-9,
        ),
        AuthoringField(
            "readout_exposure_seconds",
            "float",
            "Readout exposure seconds",
            0.005,
            minimum=1e-9,
        ),
        AuthoringField(
            "reference_before_field", "text", "Reference before API field",
            "", required=True,
        ),
        AuthoringField(
            "readout_field", "text", "Readout API field",
            "", required=True,
        ),
        AuthoringField(
            "reference_after_field", "text", "Reference after API field",
            "", required=True,
        ),
        AuthoringField(
            "default_model_kind",
            "choice",
            "Default readout model",
            # The matched filter, per site: it weights each pixel by how much
            # signal that pixel actually carries, which is what the readout
            # is for.  Box stays offered -- it is the one to fall back to
            # when the PSF fit is in doubt -- but it is not the default.
            ReadoutModelKind.PER_SITE_PSF.value,
            choices=(
                AuthoringChoice(ReadoutModelKind.BOX.value, "Box"),
                AuthoringChoice(
                    ReadoutModelKind.PER_SITE_PSF.value,
                    "Per-site PSF",
                ),
                AuthoringChoice(
                    ReadoutModelKind.UNIFORM_PSF.value,
                    "Uniform PSF",
                ),
            ),
        ),
        AuthoringField(
            "threshold_method",
            "choice",
            "Threshold method",
            "gaussian",
            choices=(
                AuthoringChoice("gaussian", "Gaussian"),
                AuthoringChoice("empirical", "Empirical"),
            ),
        ),
        AuthoringField(
            "box_half_width",
            "int",
            "Box half-width",
            1,
            minimum=0,
        ),
        AuthoringField(
            "psf_half_width",
            "int",
            "PSF half-width",
            3,
            minimum=0,
        ),
        AuthoringField(
            "psf_padding",
            "int",
            "PSF background padding",
            3,
            minimum=1,
        ),
        AuthoringField(
            "detection_spot_sigma",
            "float",
            "Detection spot sigma",
            1.0,
            minimum=1e-9,
        ),
        AuthoringField(
            "detection_sigma",
            "float",
            "Detection threshold sigma",
            6.0,
            minimum=1e-9,
        ),
        # Where the frames come from, and whether they are kept.  Detection
        # settings are the part of a calibration an operator retunes most, and
        # retuning them used to mean holding the bench for another few hundred
        # samples; a folder of saved ones is calibrated again in seconds.
        AuthoringField(
            "frame_source",
            "choice",
            "Frames",
            FRAMES_FROM_CAMERA,
            choices=(
                AuthoringChoice(FRAMES_FROM_CAMERA, "Acquire new frames"),
                AuthoringChoice(FRAMES_FROM_FOLDER, "Calibrate saved frames"),
            ),
        ),
        AuthoringField(
            "saved_frames_path",
            "folder",
            "Saved frames folder",
            # Where runs are filed: saved samples live under the day folder of
            # the run that took them, so this is the place to start looking.
            "data",
            enabled_when=("frame_source", (FRAMES_FROM_FOLDER,)),
        ),
        AuthoringField(
            "save_frames",
            "bool",
            "Save every sample",
            False,
            enabled_when=("frame_source", (FRAMES_FROM_CAMERA,)),
        ),
        AuthoringField(
            "review_detected_sites",
            "bool",
            "Review detected sites",
            False,
        ),
        photoelectron_switch(enabled_when=("frame_source", (FRAMES_FROM_CAMERA,))),
    ),
    validator=_validate_calibration,
)


def _build(
    *,
    camera: object,
    camera_key: str,
    sequencer: object,
    sequencer_key: str,
    pulse_resource: ResolvedWorkspaceResource,
    signal_plane: object,
    build_figure_host: object = None,
    save_figure_artifact: object = None,
    **values: object,
) -> CalibrationTask:
    authored = CALIBRATION_SCHEMA.project_values(values)
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id
        != _CALIBRATION_PULSE_RESOURCE.contract_id
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved calibration pulse")
    return CalibrationTask(
        camera=camera,  # type: ignore[arg-type]
        sequencer=sequencer,
        request=CalibrationRequest(
            camera_key=camera_key,
            sequencer_key=sequencer_key,
            pulse_template=pulse_resource.path.name,
            repeats=int(authored["repeats"]),
            reference_exposure_seconds=float(
                authored["reference_exposure_seconds"]
            ),
            readout_exposure_seconds=float(authored["readout_exposure_seconds"]),
            reference_before_field=str(authored["reference_before_field"]),
            readout_field=str(authored["readout_field"]),
            reference_after_field=str(authored["reference_after_field"]),
            default_model_kind=ReadoutModelKind(authored["default_model_kind"]),
            threshold_method=str(authored["threshold_method"]),
            box_half_width=int(authored["box_half_width"]),
            psf_half_width=int(authored["psf_half_width"]),
            psf_padding=int(authored["psf_padding"]),
            detection_spot_sigma=float(authored["detection_spot_sigma"]),
            detection_sigma=float(authored["detection_sigma"]),
            save_frames=bool(authored["save_frames"]),
            review_detected_sites=bool(authored["review_detected_sites"]),
            photoelectrons=bool(authored[PHOTOELECTRONS]),
            frame_source=str(authored["frame_source"]),
            saved_frames_path=str(authored["saved_frames_path"]),
        ),
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        signal_plane=signal_plane,
        build_figure_host=build_figure_host,
        save_figure_artifact=save_figure_artifact,
    )


def _calibration_editor_factory(parent=None):
    """Keep exposure and API choices in the same projected Fluent form."""
    from collections.abc import Mapping
    from dataclasses import replace
    from PyQt5 import QtCore
    from zlc_pulse import field_label
    from zlc_ui.form import FluentParameterForm, FormChoice, FormSpec

    class CalibrationForm(FluentParameterForm):
        draft_changed = QtCore.pyqtSignal(object)
        # Resource pickers keep their existing generic Refresh action.
        managed_fields = tuple(
            field.name for field in CALIBRATION_SCHEMA.fields if field.value_type != "resource"
        )

        def __init__(self, parent=None):
            super().__init__(FormSpec(()), parent=parent)
            self.changed.connect(
                lambda key: self.draft_changed.emit({"values": {key: self.read_value(key)}})
            )
            self.value_normalized.connect(
                lambda key: self.draft_changed.emit({"values": {key: self.read_value(key)}})
            )

        def update_projection(self, projection):
            resources = projection.get("workspace_resources") or {}
            resource = resources.get("pulse_template") if isinstance(resources, Mapping) else None
            sequence = getattr(resource, "value", None)
            choices = (FormChoice("Select API field", ""),)
            if isinstance(sequence, PulseSequence):
                choices += tuple(
                    FormChoice(field_label(sequence, binding.field_ref), binding.field_id)
                    for binding in sequence.api_bindings
                )
            values = projection.get("form_values") or {}
            fields = []
            for field in projection["form_spec"].fields:
                key = field.key
                if key not in self.managed_fields:
                    continue
                if key not in {"reference_before_field", "readout_field", "reference_after_field"}:
                    fields.append(field)
                    continue
                selected = str(values.get(key) or "")
                offered = choices
                if selected and not any(choice.value == selected for choice in choices):
                    offered += (FormChoice(f"Unavailable: {selected}", selected),)
                fields.append(replace(
                    field, kind="choice", default=selected, choices=offered,
                ))
            self.reconcile(FormSpec(tuple(fields)), {
                field.key: values[field.key] for field in fields
            })

        def set_mutation_enabled(self, enabled):
            self.setEnabled(enabled)

    return CalibrationForm(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    outputs=(CAPTURE_PREVIEW_DECLARATION,),
    # Long reference, short readout, long reference: three frames whose whole
    # point is to be compared with each other, so they are faceted rather than
    # reduced.  Pinned to "image" they were averaged into one picture; left
    # unpinned they were averaged too, because a plotting package reads shape
    # and this is physics.
    node_previews=(
        NodePreviewSpec(CAPTURE_PREVIEW_DECLARATION, "facet_grid"),
        NodePreviewSpec(SITE_REVIEW_DECLARATION, "image", producer="review"),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", CALIBRATION_ARTIFACT_CODEC.contract_id),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
    ),
    build=_build,
    workspace_resources=(_CALIBRATION_PULSE_RESOURCE,),
    ui_contributions=(_calibration_editor_factory,),
    # Whether it can read that way is the camera's answer: calibration keeps
    # no conversion of its own, so a bench that has not configured one cannot
    # switch this on here either.
    resolve_field_availability=resolve_photoelectron_availability,
)


__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
