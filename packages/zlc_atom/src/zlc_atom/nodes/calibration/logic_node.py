"""Discoverable calibration task with one live preview and saved results."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.nodes.calibration.task import camera_exposure_seconds
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
from zlc_pulse import PulseSequence, api_bindings_in_period_order

from .artifact import CALIBRATION_ARTIFACT_CODEC
from .calibration import READOUT_MODEL_CHOICES, ReadoutModelKind
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
#: What only taking NEW frames uses: the pulse, the samples, the windows and
#: the API fields that drive them, saving them, reading them in
#: photoelectrons -- and the camera and sequencer below.  Calibrating saved
#: frames again reads the crop, exposure and pulse facts recorded beside
#: them, so these stay in the draft as they are, unchecked and unused.
_NEW_FRAMES = ("frame_source", (FRAMES_FROM_CAMERA,))


def _validate_calibration(values: dict[str, object]) -> None:
    # The same relation the request enforces, so the form never accepts a
    # draft the task will refuse: the readout frame is recognised in the
    # compiled program by being the SHORT window, so equal is not enough.
    if values["frame_source"] != FRAMES_FROM_CAMERA:
        return
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
            enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "repeats", "int", "Samples", 200, minimum=1, enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "reference_exposure_seconds",
            "float",
            "Reference exposure seconds",
            0.02,
            minimum=1e-9,
            enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "readout_exposure_seconds",
            "float",
            "Readout exposure seconds",
            0.005,
            minimum=1e-9,
            enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "camera_exposure_seconds",
            "float",
            "Camera exposure seconds",
            None,
            derived=True,
            description=(
                "Set by the calibration: the camera is armed with the longest "
                "of the three windows, so every frame integrates its whole window."
            ),
            # Calibrating saved frames arms no camera, so there is none.
            blank_label="no camera",
        ),
        AuthoringField(
            "reference_before_field", "text", "Reference before API field",
            "", required=True, enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "readout_field", "text", "Readout API field",
            "", required=True, enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "reference_after_field", "text", "Reference after API field",
            "", required=True, enabled_when=_NEW_FRAMES,
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
            # Every model by name; a calibration has no default of its own
            # to defer to while it is the one being made.
            choices=READOUT_MODEL_CHOICES[1:],
            segmented=True,
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
            description=(
                "Gaussian: fit Empty/Occupied populations separately using agreeing "
                "reference frames, then use their weighted crossing. Empirical: "
                "maximize correct classifications on those labelled samples."
            ),
            segmented=True,
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
            # Short enough to sit side by side: the row already says
            # "Frames", and the folder row below names where saved ones are.
            choices=(
                AuthoringChoice(FRAMES_FROM_CAMERA, "New frames"),
                AuthoringChoice(FRAMES_FROM_FOLDER, "Saved frames"),
            ),
            segmented=True,
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
            enabled_when=_NEW_FRAMES,
        ),
        AuthoringField(
            "review_detected_sites",
            "bool",
            "Review detected sites",
            False,
        ),
        photoelectron_switch(enabled_when=_NEW_FRAMES),
    ),
    validator=_validate_calibration,
)


def _build(
    *,
    signal_plane: object,
    camera: object = None,
    camera_key: str = "",
    sequencer: object = None,
    sequencer_key: str = "",
    pulse_resource: ResolvedWorkspaceResource | None = None,
    build_figure_host: object = None,
    save_figure_artifact: object = None,
    **values: object,
) -> CalibrationTask:
    authored = CALIBRATION_SCHEMA.project_values(values)
    analysis = dict(
        default_model_kind=ReadoutModelKind(authored["default_model_kind"]),
        threshold_method=str(authored["threshold_method"]),
        box_half_width=int(authored["box_half_width"]),
        psf_half_width=int(authored["psf_half_width"]),
        psf_padding=int(authored["psf_padding"]),
        detection_spot_sigma=float(authored["detection_spot_sigma"]),
        detection_sigma=float(authored["detection_sigma"]),
        review_detected_sites=bool(authored["review_detected_sites"]),
        frame_source=str(authored["frame_source"]),
        saved_frames_path=str(authored["saved_frames_path"]),
    )
    if authored["frame_source"] != FRAMES_FROM_CAMERA:
        # Saved frames: no bench, no pulse -- the folder says what they were.
        return CalibrationTask(
            request=CalibrationRequest(**analysis),
            signal_plane=signal_plane,
            build_figure_host=build_figure_host,
            save_figure_artifact=save_figure_artifact,
        )
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
            **analysis,
            camera_key=camera_key,
            sequencer_key=sequencer_key,
            pulse_template=pulse_resource.path.name,
            repeats=int(authored["repeats"]),
            reference_exposure_seconds=float(
                authored["reference_exposure_seconds"]
            ),
            readout_exposure_seconds=float(authored["readout_exposure_seconds"]),
            # By the rule, not from the form: the derived field is what the
            # form SHOWS of this rule, and a build reached without the form
            # (a notebook, a test) arms the camera the same way.
            camera_exposure_seconds=camera_exposure_seconds(
                float(authored["reference_exposure_seconds"]),
                float(authored["readout_exposure_seconds"]),
            ),
            reference_before_field=str(authored["reference_before_field"]),
            readout_field=str(authored["readout_field"]),
            reference_after_field=str(authored["reference_after_field"]),
            save_frames=bool(authored["save_frames"]),
            photoelectrons=bool(authored[PHOTOELECTRONS]),
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
        # Every field, the pulse picker included: what Frames chooses greys
        # the pulse with the rest of what only new frames use, and a form
        # follows only the fields it holds.  The picker's Refresh reaches
        # the editor through this form's own refresh_requested.
        managed_fields = CALIBRATION_SCHEMA.field_names

        def __init__(self, parent=None):
            super().__init__(FormSpec(()), parent=parent)
            self.changed.connect(self._value_changed)
            self.value_normalized.connect(
                lambda key: self._value_changed(key, normalized=True)
            )

        def _value_changed(self, key, *, normalized=False):
            # A Qt slot: a box holding a number it cannot show, or a required
            # box left empty, is no value yet -- and an exception raised here
            # aborts the whole console.  The draft keeps its last value.
            try:
                value = self.read_value(key)
            except (TypeError, ValueError):
                # An emptied pulse picker is the vacancy it shows, as on the
                # editor's own form: Start names the missing file.
                field = next(field for field in self.spec.fields if field.key == key)
                if field.kind != "path" or not self.is_empty(key):
                    return
                value = ""
            patch = {"values": {key: value}}
            if normalized:
                # The box's own display normalization, not an operator edit.
                patch["normalized"] = True
            self.draft_changed.emit(patch)

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
                    # Calibrating saved frames reads no pulse, so a choice
                    # kept for new frames is named as it stands, not called
                    # unavailable.
                    offered += (FormChoice(
                        f"Unavailable: {selected}"
                        if values.get("frame_source") == FRAMES_FROM_CAMERA
                        else selected,
                        selected,
                    ),)
                fields.append(replace(
                    field, kind="choice", default=selected, choices=offered,
                ))
            self.reconcile(FormSpec(tuple(fields)), {
                field.key: values[field.key] for field in fields
            })

        def set_mutation_enabled(self, enabled):
            self.setEnabled(enabled)

    return CalibrationForm(parent)


#: The three windows a calibration compares, in the order the pulse plays
#: them: the reference before, the readout, the reference after.
_API_FIELD_NAMES = ("reference_before_field", "readout_field", "reference_after_field")


def _default_api_fields(values, resources):
    """The pulse's first three API durations, in the order they play.

    A calibration pulse is written with exactly these three windows as its
    API parameters, so on a template chosen for the purpose the defaults are
    the answer and the operator has nothing to pick.  Only empty fields take
    them (the framework's rule): a field the operator set stands, and one
    naming a parameter the chosen pulse lacks is shown as unavailable rather
    than silently replaced.
    """

    del values
    resource = resources.get("pulse_template")
    sequence = getattr(resource, "value", None)
    if not isinstance(sequence, PulseSequence):
        return {}
    windows = (
        binding for binding in api_bindings_in_period_order(sequence)
        if binding.kind == "duration"
    )
    return {
        name: binding.field_id
        for name, binding in zip(_API_FIELD_NAMES, windows)
    }


def _camera_exposure(values):
    """The derived camera exposure, while the two windows are numbers and
    the frames are new: calibrating saved ones arms no camera."""

    if values.get("frame_source", FRAMES_FROM_CAMERA) != FRAMES_FROM_CAMERA:
        return {}
    try:
        reference = float(values.get("reference_exposure_seconds"))
        readout = float(values.get("readout_exposure_seconds"))
    except (TypeError, ValueError):
        return {}
    return {"camera_exposure_seconds": camera_exposure_seconds(reference, readout)}


def _resolve_defaults(values, resources):
    """What the node fills itself: the API fields from its pulse, the camera exposure from its windows."""

    return {**_default_api_fields(values, resources), **_camera_exposure(values)}


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    description=(
        "Acquire long-reference/readout/long-reference camera cycles using the selected "
        "Pulse, detect and optionally review sites, then calibrate per-site readout "
        "classification. Saves a Calibration artifact for Occupancy plus native "
        "diagnostic Figures; this Task owns camera preparation and Pulse Fire. With "
        "Frames = Saved frames it calibrates the samples a run saved instead, using "
        "no camera, sequencer or Pulse and claiming no device; the Pulse, Samples, "
        "exposure and API-field settings then stay in the draft unused."
    ),
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
        DeviceRequirement(
            "camera.adapter", "camera", CAMERA_PROTECTED_FIELDS, enabled_when=_NEW_FRAMES,
        ),
        DeviceRequirement(
            "sequencer.streamer", "sequencer", ("program",), enabled_when=_NEW_FRAMES,
        ),
    ),
    build=_build,
    workspace_resources=(_CALIBRATION_PULSE_RESOURCE,),
    ui_contributions=(_calibration_editor_factory,),
    # Whether it can read that way is the camera's answer: calibration keeps
    # no conversion of its own, so a bench that has not configured one cannot
    # switch this on here either.
    resolve_field_availability=resolve_photoelectron_availability,
    resolve_defaults=_resolve_defaults,
)


__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
