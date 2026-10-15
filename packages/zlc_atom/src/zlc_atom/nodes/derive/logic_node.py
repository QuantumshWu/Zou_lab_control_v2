"""Discoverable derive processor descriptor."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)
from zlc_runtime import DatasetOutputDeclaration, split_signal_key

from .expression import (
    VARIABLE_COLUMNS, changing_axes, checked_variables, frames_kept, input_members, published, typed_variables,
)
from .processor import DeriveProcessor, declared_outputs


def _readable(values: Mapping[str, object]) -> None:
    """A complete draft is one whose variables read without data or the
    source: every name, at least one published variable, and the formula
    and statistic of every variable a published one reads.  What each one
    reads is checked where the inputs' names are known: admission, build.

    Refused here, by what was written, while the draft is being edited --
    not at Start, where the same words would arrive as a failure to run.
    """

    checked_variables(values["variables"])


DERIVE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "variables",
            "rows",
            "Variables",
            (),
            required=True,
            columns=VARIABLE_COLUMNS,
            description="one row per variable; only variables with publish on become signals",
        ),
    ),
    validator=_readable,
)
#: The same rows without the validator: a raw draft's rows projected only
#: for what they declare.
_ROWS = AuthoringSchema(DERIVE_SCHEMA.fields)


def _draft_outputs(
    values: Mapping[str, object],
    devices: Mapping[str, object],
) -> tuple[DatasetOutputDeclaration, ...]:
    """What one draft publishes: its published variables, in card order.

    Each one whose name reads is declared -- a formula being edited does
    not withdraw the signal a panel shows, and a variable being renamed
    withdraws only its own -- and nothing while the rows do not project;
    the schema's validator says why, to the editor, in the draft's own
    words.
    """

    del devices
    try:
        rows = _ROWS.draft_values({"variables": values.get("variables", ())})["variables"]
        return declared_outputs(rows)
    except (TypeError, ValueError):
        return ()


def _describe_output_frames(values: Mapping[str, object], output_name: str,
                            source: tuple[str | None, tuple[str, ...]] | None,
                            ) -> tuple[str | None, tuple[str, ...]] | None:
    """A published variable's readout-event axis before any data: typing's
    own rules on the axis the source names (:func:`frames_kept`), every
    member carrying the source's labels -- the camera's ``frame``, Frame
    Survival's ``pair``, whatever it is called; no axis, one frame, when
    the variable keeps none; None when the source or the draft cannot say.

    The console answers one axis for the whole bundle, and only when every
    member carries the same.  A single label is one frame or no axis at
    all, so the members are read as carrying none: a pick or statistic on
    that axis cannot be told, and every other formula keeps one frame.

    A name no variable has is read as a member's, so a name nothing has, or
    a variable named like a member, is refused by typing once the members
    publish and by Start, not here.
    """

    if source is None:
        return None
    try:
        rows = _ROWS.draft_values({"variables": values.get("variables", ())})["variables"]
    except (TypeError, ValueError):
        return None
    if output_name not in published(rows):
        return None
    name, labels = source
    carried = tuple(labels) if len(labels) > 1 else ()
    kept = frames_kept(rows, name, carried, None)
    frames = next(kept[position] for position, row in enumerate(rows) if row["name"] == output_name)
    return None if frames is None else (name, frames) if frames else (None, ("0",))


def _admit_source(values: Mapping[str, object], publication: object) -> tuple[str, ...]:
    """What the variables cannot compute from what the producer publishes.

    Every variable typed against the schemas an evaluation reads -- the
    run's whole geometry for a finite source, the latest event for a
    monitor -- by the evaluation's own rules, the typing the editor shows
    under each card: what each one reads, its picks (never on an axis each
    publication carries its own part of, :func:`changing_axes`), operators
    and statistic.  Each problem of a LIVE variable, and any variable's
    name problem, in its ``<variable> · <stage>: …`` words.  Asked with the
    rest of the draft's admission, so the Edit page says it as a draft and
    Start waits; a parked helper's problems never hold Start.
    """

    inputs, changing = {}, {}
    for key, value in publication.signals.items():
        parts = split_signal_key(key)
        if parts is None:
            continue
        event = value.snapshot.block.schema
        inputs[parts[1]] = schema = value.canonical_schema or event
        changing.update(changing_axes(schema, event))
    return tuple(
        str(answer["error"]) for answer in typed_variables(values["variables"], inputs, changing)
        if answer["error"] is not None and (answer["live"] or answer["error"].stage == "name")
    )


def _build(*, signal_plane: object, source_signal: str, **values: object) -> DeriveProcessor:
    """The processor for the variables over the bound producer's bundle.

    The plane says which outputs the bound producer has, so a name a live
    variable reads that the producer never publishes is refused here, by
    variable and stage, before anything runs -- and the bound signal is
    recognised among them by the plane's own resolution, never by taking
    its name apart.  What the variables cannot compute from the producer's
    latest publication was admitted with the draft (``_admit_source``).
    """

    rows = DERIVE_SCHEMA.project_values(values)["variables"]
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    described = {item.name: item for item in signal_plane.describe_signals()}
    source = described.get(selected_source)
    if source is None:
        raise ValueError("select an available input bundle")
    # Output names are read by the signal-key grammar's one reader.
    selected = split_signal_key(selected_source)
    if selected is None:
        raise ValueError("select an available input bundle")
    primary = selected[1]
    members = tuple(
        parts[1] for item in described.values()
        if item.owner_id == source.owner_id and (parts := split_signal_key(item.name)) is not None
    )
    names = tuple(dict.fromkeys((primary, *input_members(rows, members))))
    signal_plane.resolve_sibling_signals(selected_source, names)
    return DeriveProcessor(variables=rows, primary_output=primary, input_outputs=names)


def _editor_factory(parent=None):
    from .editor import DeriveEditor
    return DeriveEditor(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "derive",
    NodeKind.PROCESSOR,
    DERIVE_SCHEMA,
    description=(
        "Within-evaluation algebra on one producer's synchronized bundle: live, each publication; at "
        "the end of a finite run, the run's Dataset. values.variables has one row per variable: name; "
        "publish (default true; false keeps a helper internal: other variables read it, nothing "
        "publishes it); terms, the formula, one row per line; optional statistic + over. A term names "
        "of (an input member or any other variable, no cycles), or gives number, or boolean "
        "\"true\"/\"false\" (only after a comparison). at with labels (a list) picks coordinates of of: "
        "at is an axis name (any domain) and labels are that axis' labels from the catalog, or at is a domain "
        "with two or more axes (Repeat, Point, Cell) and labels are its rows written \"detuning 2 · power "
        "1\"; \"first\" and \"last\" are its first and last coordinate (row) whatever its length, resolved "
        "each evaluation, never a fixed label. One label drops the axis (the domain's axes), several keep "
        "those coordinates (one two of them name, once). A pick never "
        "names an axis each publication carries its own part of (a finite run's shots and scan points, a "
        "monitor's shots): reduce those, or plot them. Every term after the first has op: arithmetic + - "
        "* /, comparison == != > < >= <=, logic and, or, \"and not\", mask where, \"where not\". "
        "Precedence: * / then + - then one comparison then and/\"and not\" then or then where; left to "
        "right within a level; no parentheses (put a group in an internal variable). statistic sum mean "
        "count min max std any all reduces the over axis names (a list, any domains). Booleans: sum is "
        "the number true, mean the fraction true, count the valid samples; booleans never mix with "
        "numbers, and two booleans take every comparison, false < true (occupied at frame 0 > occupied "
        "at frame 2: loaded at 0 and empty at 2). An operand whose axes are a subset of the other's "
        "repeats along the rest. Only published variables become signals; an unpublished variable no "
        "published one reads is not evaluated and never blocks Start. Example: "
        "[{\"name\":\"same\",\"publish\":false,\"terms\":[{\"of\":\"occupied\",\"at\":\"frame\","
        "\"labels\":[\"0\"]},{\"op\":\"==\",\"of\":\"occupied\",\"at\":\"frame\",\"labels\":[\"2\"]}]},"
        "{\"name\":\"kept\",\"terms\":[{\"of\":\"counts\"},{\"op\":\"where\",\"of\":\"same\"}]}]. "
        "Shot averages with error bars, per-site probabilities and fits belong in a plot panel."
    ),
    input_specs=(DatasetInputSpec("bundle", None, "exact", select_bundle=True),),
    declare_outputs=_draft_outputs,
    describe_output_frames=_describe_output_frames,
    admit_source=_admit_source,
    ui_contributions=(_editor_factory,),
    build=_build,
)


__all__ = ["DERIVE_SCHEMA", "LOGIC_NODE"]
