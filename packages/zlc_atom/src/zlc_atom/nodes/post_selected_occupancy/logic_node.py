"""Post-selected occupancy: Occupancy, kept only where two frames say so.

It IS Occupancy -- its inputs, fields, editor, build, processor, outputs and
frames -- plus one step Occupancy's processor takes before it publishes
(``post_select``): counts and occupied stay valid only where a condition on
two frames of the cycle holds for that shot and site, on every frame.  The
condition is Derive rows (:func:`_rows`) evaluated by Derive's engine, so the
defaults are exactly Occupancy -> Derive ``same = occupied[frame first] ==
occupied[frame last]``, ``kept = occupied where same``.  frame_judged is
Occupancy's evidence frames as they are: its cells are pixels, and Derive
refuses a site mask on them.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

from zlc_data import owned_snapshot_from_arrays

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import LogicNodeDescriptor, NodeKind
from zlc_atom.nodes._framework.frames import FIRST, LAST, REST, absent_notes, frame_token, resolve_frame
from zlc_atom.nodes.derive.expression import CATEGORIES, OPERATORS, Operand, evaluate
from zlc_atom.nodes.derive.logic_node import DERIVE_SCHEMA
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY, OCCUPANCY_SCHEMA
from zlc_atom.nodes.occupancy.processor import OccupancyResult

#: A slot's verdict in Values mode, spelled as Derive's boolean term is;
#: "any" asks nothing of the slot, so its frame need not be judged.
ANY = "any"
#: The members the condition keeps; frame_judged has no site to keep.
_MASKED = ("counts", "occupied")

FRAME_A = AuthoringField(
    "frame_a", "text", "Frame A", FIRST, required=True,
    description="'first', 'last' or a middle frame number between them, as Occupancy's frame claims "
                "name frames: first and last follow the cycle's ends")
FRAME_B = AuthoringField(
    "frame_b", "text", "Frame B", LAST, required=True, description=FRAME_A.description)
MODE = AuthoringField(
    "mode", "choice", "Mode", "relation", required=True, segmented=True,
    choices=(AuthoringChoice("relation", "Relation"), AuthoringChoice("values", "Values")),
    description="relation: kept where frame A's verdict relates to frame B's by relation; values: kept "
                "where each frame's verdict is its value. The other mode's fields stay in the draft, unused")
# Derive's comparison spellings and labels: the condition is a Derive term.
RELATION = AuthoringField(
    "relation", "choice", "Relation", "==", segmented=True, enabled_when=("mode", ("relation",)),
    choices=tuple(AuthoringChoice(op, OPERATORS[op][2]) for op in dict(CATEGORIES)["comparison"]),
    description="Derive's comparisons on booleans, false < true: '>' keeps loaded at A and empty at B; "
                "both frames must be judged")
_VALUES = (AuthoringChoice(ANY, "Any"), AuthoringChoice("true", "1"), AuthoringChoice("false", "0"))
VALUE_A = AuthoringField(
    "value_a", "choice", "Frame A value", "true", segmented=True, choices=_VALUES,
    enabled_when=("mode", ("values",)),
    description="'true' (1, loaded), 'false' (0, empty) or 'any' (not asked; its frame need not be judged)")
VALUE_B = AuthoringField(
    "value_b", "choice", "Frame B value", ANY, segmented=True, choices=_VALUES,
    enabled_when=("mode", ("values",)), description=VALUE_A.description)
CONDITION_FIELDS = (FRAME_A, FRAME_B, MODE, RELATION, VALUE_A, VALUE_B)


def _frame(field: AuthoringField, text: object) -> int | str:
    """One slot's frame claim; frame 0 is written "first", as a claim is."""

    try:
        token = frame_token(text)
    except ValueError:
        token = REST
    if token == REST:
        raise ValueError(f"{field.label} must be first, last or a middle frame number, not {text!r}")
    return FIRST if token == 0 else token


def _frame_words(token: int | str) -> str:
    return f"the {token} frame" if token in (FIRST, LAST) else f"frame {token}"


def condition(values: Mapping[str, object]) -> dict[str, object]:
    """What the draft keeps, checked without the source -- the run record's
    ``post_selection`` -- or the one sentence refusing it.  Only the chosen
    mode's fields are read."""

    frames = (_frame(FRAME_A, values[FRAME_A.name]), _frame(FRAME_B, values[FRAME_B.name]))
    if values[MODE.name] == "relation":
        if frames[0] == frames[1]:
            raise ValueError(f"Frame A and Frame B are both {_frame_words(frames[0])}: "
                             "a relation needs two different frames")
        return {"frames": frames, "mode": "relation", "relation": values[RELATION.name]}
    wanted = (values[VALUE_A.name], values[VALUE_B.name])
    if wanted == (ANY, ANY):
        raise ValueError("Frame A and Frame B are both Any: give one of them 1 or 0")
    return {"frames": frames, "mode": "values", "values": wanted}


def _used(kept: Mapping[str, object]) -> tuple[bool, bool]:
    """Which slots the condition reads: both for a relation, the ones not Any for values."""

    return (True, True) if kept["mode"] == "relation" else tuple(value != ANY for value in kept["values"])


def source_problem(kept: Mapping[str, object], count: int) -> str:
    """Why a ``count``-frame cycle cannot be post-selected as ``kept`` asks, or ''."""

    if count < 2:
        return f"post-selection needs at least two frames per cycle and this source carries {count}"
    used = [(field.label, token) for field, token, read in zip((FRAME_A, FRAME_B), kept["frames"], _used(kept))
            if read]
    return "; ".join(absent_notes(tuple((token,) for _name, token in used), count,
                                  tuple(name for name, _token in used)))


@functools.lru_cache(maxsize=32)
def _rows(axis: str, picks: tuple[str | None, str | None], mode: str, how) -> tuple[Mapping[str, object], ...]:
    """The condition as Derive variables over Occupancy's members: ``kept``
    (internal) and each masked member ``where kept``.  ``picks`` are the two
    frames' labels on ``axis``, None for a slot the condition does not read;
    ``how`` is the relation, or the two values."""

    def pick(label: str) -> dict[str, object]:
        return {"of": "occupied", "at": axis, "labels": (label,)}

    if mode == "relation":
        terms = (pick(picks[0]), {"op": how, **pick(picks[1])})
    else:
        terms = ()
        for label, value in zip(picks, how):
            if label is not None:
                terms += ({**pick(label), **({"op": "and"} if terms else {})}, {"op": "==", "boolean": value})
    variables = (
        {"name": "kept", "publish": False, "terms": terms},
        *({"name": f"{member}_kept", "terms": ({"of": member}, {"op": "where", "of": "kept"})}
          for member in _MASKED),
    )
    return DERIVE_SCHEMA.project_values({"variables": variables})["variables"]


def _post_selected(result: OccupancyResult, axis, *, kept: Mapping[str, object]) -> OccupancyResult:
    """Occupancy's ``result`` with counts and occupied valid only where ``kept`` holds."""

    count = 1 if axis is None else int(axis.size)
    problem = source_problem(kept, count)
    if problem:
        raise ValueError(problem)
    labels = axis.labels()
    picks = tuple(labels[resolve_frame(token, count)] if read else None
                  for token, read in zip(kept["frames"], _used(kept)))
    how = kept["relation"] if kept["mode"] == "relation" else kept["values"]
    snapshots = {member: result.artifacts[member].materialize() for member in _MASKED}
    masked = evaluate(_rows(axis.name, picks, kept["mode"], how), {
        member: Operand(snapshot.block.schema, snapshot.block.values, snapshot.expanded_validity())
        for member, snapshot in snapshots.items()
    })
    return OccupancyResult({**result.artifacts, **{
        member: owned_snapshot_from_arrays(
            snapshot.block.schema, snapshot.block.values, snapshot.block.revision,
            validity=masked[f"{member}_kept"].valid, stream_generation=snapshot.ref.stream_generation)
        for member, snapshot in snapshots.items()
    }})


def _validate(values: Mapping[str, object]) -> None:
    condition(values)


POST_SELECTED_OCCUPANCY_SCHEMA = AuthoringSchema((*OCCUPANCY_SCHEMA.fields, *CONDITION_FIELDS), validator=_validate)


def _build(*, calibrations, source_signal: str, **values: object):
    """Occupancy's own build, given the post-selection step."""

    authored = POST_SELECTED_OCCUPANCY_SCHEMA.project_values(values)
    kept = condition(authored)
    return OCCUPANCY.build(
        calibrations=calibrations, source_signal=source_signal,
        post_select=functools.partial(_post_selected, kept=kept), post_select_record=kept,
        **{name: authored[name] for name in OCCUPANCY_SCHEMA.field_names},
    )


def _editor_factory(parent=None):
    from .editor import PostSelectionEditor

    return PostSelectionEditor(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "post_selected_occupancy",
    NodeKind.PROCESSOR,
    POST_SELECTED_OCCUPANCY_SCHEMA,
    description=(
        "Post-selected occupancy: Occupancy with the post-selection done for you. It reads the camera frames "
        "exactly as Occupancy does -- the same Calibrations rows claiming frames, readout model, parked rows "
        "and refusals -- and publishes Occupancy's counts, occupied and frame_judged, with counts and "
        "occupied valid only where a condition on two frames of the cycle holds for that shot and site (on "
        "every frame). With its defaults it equals Occupancy followed by Derive same = occupied[frame "
        "first] == occupied[frame last] (internal), kept = occupied where same. frame_a and frame_b are "
        "'first' (default for A), 'last' (default for B) or a middle frame number, as Occupancy's frame "
        "claims name frames: first and last follow the cycle's ends. mode is either 'relation' (default): "
        "kept where occupied at frame_a <relation> occupied at frame_b, relation one of Derive's '==' "
        "(default), '!=', '>', '<', '>=', '<=' on booleans false < true ('>' keeps loaded at A and empty at "
        "B), both frames judged and different; or 'values': kept where each frame's verdict is value_a / "
        "value_b, 'true' (1), 'false' (0) or 'any' (not asked; its frame need not be judged), at least one "
        "not 'any' -- e.g. value_a 'true', value_b 'false' keeps loaded at A and empty at B, value_a 'true' "
        "alone keeps loaded at A. The other mode's fields stay in the draft unused. The condition is "
        "evaluated by the Derive engine. frame_judged is Occupancy's evidence frames unchanged (pixels have "
        "no site). A cycle of fewer than two frames, or a chosen middle frame the cycle does not have, is "
        "refused and the processor waits. Feed occupied to Frame Survival for survival of the kept sites."
    ),
    input_specs=OCCUPANCY.input_specs,
    ui_contributions=(*OCCUPANCY.ui_contributions, _editor_factory),
    outputs=OCCUPANCY.outputs,
    describe_output_frames=OCCUPANCY.describe_output_frames,
    build=_build,
)

__all__ = ["LOGIC_NODE", "POST_SELECTED_OCCUPANCY_SCHEMA"]
