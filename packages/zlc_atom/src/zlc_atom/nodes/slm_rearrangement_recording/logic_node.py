"""Record an authored imaging sequence around camera-guided rearrangement."""

from dataclasses import replace
from functools import partial

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes.slm_rearrangement.logic_node import (
    LOGIC_NODE as _REARRANGEMENT_NODE,
    SLM_REARRANGEMENT_SCHEMA,
    _build,
    _rearrangement_editor_factory,
)
from zlc_atom.nodes.slm_rearrangement.task import rearrangement_outputs


SLM_REARRANGEMENT_RECORDING_SCHEMA = AuthoringSchema(tuple(
    AuthoringField("recording_frames", "int", "Recording frames", 50, minimum=2,
                   description="Number of camera frames to retain, starting with the first imaging Period. The authored Pulse supplies the triggers.")
    if field.name == "after_period" else
    replace(field, label="First imaging Period") if field.name == "before_period" else field
    for field in SLM_REARRANGEMENT_SCHEMA.fields
))

LOGIC_NODE = replace(
    _REARRANGEMENT_NODE,
    api_name="slm_rearrangement_recording",
    authoring_schema=SLM_REARRANGEMENT_RECORDING_SCHEMA,
    build=partial(_build, schema=SLM_REARRANGEMENT_RECORDING_SCHEMA),
    ui_contributions=(partial(_rearrangement_editor_factory, schema=SLM_REARRANGEMENT_RECORDING_SCHEMA),),
    declare_outputs=lambda values, devices: rearrangement_outputs("camera_step"),
)

__all__ = ["LOGIC_NODE", "SLM_REARRANGEMENT_RECORDING_SCHEMA"]
