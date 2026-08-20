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


__all__ = ["SignalRow", "format_signal_shape", "project_signals"]


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
    #: Whether a panel is already showing it.
    shown: bool


def project_signals(
    plane: object,
    *,
    shown: object = (),
) -> tuple[SignalRow, ...]:
    """Project the plane's signals into rows, live producers first.

    Ordering is a decision, not an accident: what is still arriving is what an
    operator is most likely to want on screen, and within a producer the plane's
    own alphabetical order is stable enough to click through without things
    moving underneath.
    """

    displayed = {str(name) for name in shown}
    rows = [
        SignalRow(
            name=description.name,
            label=(
                f"{_label(description.name)}  "
                f"[{format_signal_shape(description.shape)}]"
            ),
            producer=_producer(description.name, description.owner_id),
            state=_state(description),
            derived_from=description.source_name or "",
            shown=description.name in displayed,
        )
        for description in plane.describe_signals()
    ]
    order = {"live": 0, "waiting": 1, "finished": 2, "failed": 3}
    return tuple(
        sorted(rows, key=lambda row: (order[row.state], row.producer, row.name))
    )


def format_signal_shape(shape: object) -> str:
    """Spell one published physical ``R × P × cell`` tensor."""

    if shape is None:
        return "—"
    dimensions = tuple(int(value) for value in shape)  # type: ignore[arg-type]
    if len(dimensions) < 3:
        return " × ".join(str(value) for value in dimensions)
    cell = "×".join(str(value) for value in dimensions[2:])
    return f"{dimensions[0]} × {dimensions[1]} × ({cell})"


def _state(description: object) -> str:
    if getattr(description, "failure", None):
        return "failed"
    if getattr(description, "shape", None) is None:
        return "waiting"
    return "live" if description.live else "finished"


def _label(name: str) -> str:
    """The readable tail of a qualified signal name."""

    return name.rsplit("/", 1)[-1] or name


def _producer(name: str, owner_id: str) -> str:
    """The group a signal belongs under.

    Signal names carry their producer already (``@logic/<producer>/<name>``);
    falling back to the owner id keeps a producer that names its signals some
    other way from landing in a group called nothing.
    """

    parts = [part for part in name.split("/") if part and not part.startswith("@")]
    return parts[0] if len(parts) > 1 else str(owner_id)
