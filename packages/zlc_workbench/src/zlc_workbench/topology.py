"""What is being produced right now, in the words a window can render.

The console had no answer to "what can I put on a panel?".  Its Add Panel
button was connected to nothing, and every panel in the application was named
by hand in the composition root -- so a signal that appeared while the
experiment ran was invisible unless someone had written its name in advance.

The division that matters:

* the PLANE owns the facts (which signals exist, who owns them, whether their
  generation is still live, what each was cut from) and answers in copies;
* THIS owns the projection: what to call each one for a person, which order to
  offer them in, and which are worth offering at all;
* the WINDOW owns pixels, and receives only strings, bools and ints.

No plane object, publication, snapshot or node crosses out of here.  That is
the rule that keeps a view from reading live state at whatever moment it
happens to paint, and showing half of one instant beside half of another.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_plot.semantics import schema_structure
from zlc_runtime import split_signal_key


__all__ = [
    "SignalRow", "format_signal_shape", "project_signals", "signal_label",
    "signal_output_name",
]


@dataclass(frozen=True, slots=True)
class SignalRow:
    """One offerable signal, as a window should show it."""

    #: The plane's name, which is what the caller passes back to add a panel.
    name: str
    #: What to show a person.  Signal names are qualified and repetitive; the
    #: producer is already the group, so the label need not repeat it.
    label: str
    #: The producer's group heading.
    producer: str
    #: "waiting" before the first publication, "live" while more data can
    #: arrive, "finished" once it cannot, and "failed" on producer failure.
    state: str
    #: The signal this one was cut from, or "" when it was acquired.
    derived_from: str


def project_signals(
    descriptions: object,
) -> tuple[SignalRow, ...]:
    """Project one already-read signal directory, live producers first.

    Ordering is a decision, not an accident: what is still arriving is what an
    operator is most likely to want on screen, and within a producer the plane's
    own alphabetical order is stable enough to click through without things
    moving underneath.
    """

    rows = [
        SignalRow(
            name=description.name,
            label=signal_label(description.name, description.schema),
            producer=_producer(description.name, description.owner_id),
            state=_state(description),
            derived_from=description.source_name or "",
        )
        for description in descriptions
    ]
    order = {"live": 0, "waiting": 1, "finished": 2, "failed": 3}
    return tuple(
        sorted(rows, key=lambda row: (order[row.state], row.producer, row.name))
    )


def signal_label(name: str, schema: object) -> str:
    """What to call one signal for a person: its readable name and its shape."""

    return f"{signal_output_name(name)}  [{format_signal_shape(schema)}]"


def format_signal_shape(schema: object) -> str:
    """Format the shared three-domain sizes, without adding axis names."""

    if schema is None:
        return "—"
    return " × ".join(
        "(" + (" × ".join(str(size) for _name, size in group) or "1") + ")"
        for group in schema_structure(schema)
    )


def _state(description: object) -> str:
    if getattr(description, "failure", None):
        return "failed"
    if getattr(description, "shape", None) is None:
        return "waiting"
    return "live" if description.live else "finished"


def signal_output_name(name: str) -> str:
    """The output a signal key names, read by the grammar's one reader.

    A qualified name the grammar does not spell -- a Figure Viewer's
    ``@figure/<serial>/<name>`` -- is called by its readable tail, as the
    person reading the list knows it.
    """

    parts = split_signal_key(name)
    return (name.rsplit("/", 1)[-1] or name) if parts is None else parts[1]


def _producer(name: str, owner_id: str) -> str:
    """The group a signal belongs under.

    Signal names carry their producer already (``@logic/<producer>/<name>``);
    falling back to the owner id keeps a producer that names its signals some
    other way from landing in a group called nothing.
    """

    parts = split_signal_key(name)
    return parts[0] if parts is not None else str(owner_id)
