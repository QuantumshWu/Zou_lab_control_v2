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

from .expression import OUTPUT_COLUMNS, DeriveError, admit, input_members, output_rows
from .processor import DeriveProcessor, declared_outputs


def _readable(values: Mapping[str, object]) -> None:
    """A complete draft is one whose outputs can be read without data.

    Refused here, by what was written, while the draft is being edited --
    not at Start, where the same words would arrive as a failure to run.
    """

    output_rows(values["outputs"])


DERIVE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "outputs",
            "rows",
            "Outputs",
            (),
            required=True,
            columns=OUTPUT_COLUMNS,
            description="one row per published signal: structured columns, or code",
        ),
    ),
    validator=_readable,
)


def _draft_outputs(
    values: Mapping[str, object],
    devices: Mapping[str, object],
) -> tuple[DatasetOutputDeclaration, ...]:
    """What one draft publishes: the names of its outputs.

    A draft that cannot be read publishes nothing yet; the schema's
    validator is what says why, to the editor, in the draft's own words.
    """

    del devices
    try:
        return declared_outputs(values.get("outputs") or ())
    except DeriveError:
        return ()


def _build(*, signal_plane: object, source_signal: str, **values: object) -> DeriveProcessor:
    """The processor for the outputs over the bound producer's bundle.

    The plane says which outputs the bound producer has, so a name an
    output reads that the producer never publishes is refused here, by
    output and stage, before anything runs -- and the bound signal is
    recognised among them by the plane's own resolution, never by taking
    its name apart.  Once the producer has published, every structured
    output is typed against the schemas the next evaluation reads, by the
    evaluation's own rules: an output that cannot be computed refuses Start.
    """

    rows = DERIVE_SCHEMA.project_values(values)["outputs"]
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
    publication = signal_plane.latest_publication(selected_source)
    if publication is not None:
        admit(rows, {
            parts[1]: value.schema for key, value in publication.signals.items()
            if (parts := split_signal_key(key)) is not None
        })
    signal_plane.resolve_sibling_signals(selected_source, names)
    return DeriveProcessor(outputs=rows, primary_output=primary, input_outputs=names)


def _editor_factory(parent=None):
    from .editor import DeriveEditor
    return DeriveEditor(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "derive",
    NodeKind.PROCESSOR,
    DERIVE_SCHEMA,
    description=(
        "Within-shot algebra on one producer's synchronized bundle; each output row publishes a named "
        "signal for plots, scans and processors. A structured row takes value (a member or an earlier "
        "output), pins it with at (a readout-event label such as \"1\"; empty keeps every event), then "
        "optionally masks it (where, where_at, where_is), combines it (op + - * / == != > < >= <= & | "
        "with a member, an earlier output or a number in the value's unit; with_at) and reduces it "
        "(statistic sum mean count min max std any all over comma-separated within-shot axes), in "
        "that order; chain rows for more. On booleans sum is the number true, mean the fraction true, "
        "count the valid samples; booleans do not mix with numbers. Operands whose axes are a subset "
        "of the value's repeat along the rest. A code row has only name and code: Python over "
        "a.<member>, assigning result. Example: {\"name\": \"loaded\", \"value\": \"occupied\", "
        "\"at\": \"0\", \"statistic\": \"sum\", \"over\": \"Site\"}. Averages over shots, per-site "
        "probabilities and fits belong in a plot panel."
    ),
    input_specs=(DatasetInputSpec("a", None, "exact", select_bundle=True),),
    declare_outputs=_draft_outputs,
    ui_contributions=(_editor_factory,),
    build=_build,
)


__all__ = ["DERIVE_SCHEMA", "LOGIC_NODE"]
