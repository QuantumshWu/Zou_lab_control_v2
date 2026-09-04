"""Bring pulse files written by an older build up to today's document schema.

A pulse is experiment content: it lives in the operator's workspace, not in
the package, and it outlives the build that wrote it.  The reader, by design,
is a strict whitelist -- a field it does not know is an error, not something
to ignore -- so every time the schema is renamed, every pulse already sitting
on a disk stops opening, with a message naming a field the operator never
typed:

    cannot open imaging_template.json: unknown pulse field(s): repeat

The product does not carry readers for old shapes; that is a deliberate
choice and this tool is its other half.  It is a ONE-SHOT: run it once on
each machine that holds pulses, then delete it -- module, launcher and test
in one commit -- exactly as the previous migration was removed in ``5d889a7``.

What it will not do is guess.  Each step below restores one specific field
the product itself once wrote, and nothing is written until the migrated
document has been read back through the very reader the Pulse Editor uses.
A document that still will not read is left exactly as it was found, and
reported in the reader's own words.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
import sys
from typing import Any, Callable

from zlc_pulse import PulseSequence
from zlc_pulse.codec import PULSE_TREE_FORMAT, parse_pulse_tree_json

from ..pulse_state import read_pulse, state_from_tree, write_pulse
from ..session import Workspace

#: What a pulse document called itself before ``de9d5ee`` dropped version
#: suffixes from every artifact this project writes.  Listed rather than
#: pattern-matched: a tag this tool has not been taught belongs to a document
#: nobody has checked, and guessing at one is how a migration eats content.
SUPERSEDED_FORMATS = (f"{PULSE_TREE_FORMAT}.v1",)

#: Slot ids the pre-split editor gave to API parameters, back when they were
#: slots and owned scan-table columns of their own.
_API_SLOT_PREFIX = "api_"

#: The original, kept beside the file it came from.  Deliberately not a
#: ``.json`` name, so the Open dialog does not offer it as a pulse and a
#: second run of this tool does not try to migrate it.
BACKUP_SUFFIX = ".pre-migration"


def _default_run_repeats() -> int:
    """The value the model itself uses when nobody authored one.

    Read off ``PulseSequence`` rather than written down here, because a
    document that predates the field did not choose 0 -- it had no opinion,
    and the honest reconstruction is whatever "no opinion" means today.
    """

    return inspect.signature(PulseSequence.__init__).parameters["run_repeats"].default


def _editor(tree: dict[str, Any]) -> dict[str, Any]:
    section = tree.get("editor")
    return dict(section) if isinstance(section, Mapping) else {}


def _had_api_parameter_columns(before: Mapping[str, Any]) -> bool:
    """Whether this document is older than the API-parameter split.

    Before it, an API parameter was a slot named ``api_*`` and owned a column
    in the scan table; after it, API parameters are their own list and the
    scan table stopped carrying them.
    """

    if "api_parameters" in before:
        return False
    return any(
        str(slot.get("slot_id", "")).startswith(_API_SLOT_PREFIX)
        for slot in before.get("slots", ())
        if isinstance(slot, Mapping)
    )


def _adopt_the_current_format_tag(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    if before.get("format") not in SUPERSEDED_FORMATS:
        return None
    tree["format"] = PULSE_TREE_FORMAT
    return f"format {before['format']!r} -> {PULSE_TREE_FORMAT!r}"


def _rename_repeat_to_bracket(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    """``663de9b``: one word, three meanings, split into three fields.

    The old ``repeat`` was the period bracket and only ever that; the run and
    scan repeats it was confused with were never in the file at all.
    """

    if "repeat" not in before or "bracket" in before:
        return None
    tree["bracket"] = tree.pop("repeat")
    return "repeat -> bracket"


def _add_the_missing_run_repeats(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    if "run_repeats" in before:
        return None
    tree["run_repeats"] = _default_run_repeats()
    return f"run_repeats = {tree['run_repeats']}"


def _add_the_missing_api_parameters(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    """The API-parameter split, as the removed migrator performed it.

    Columns belonging to ``api_*`` slots leave the scan table with them; a
    document that never had such slots simply gains an empty list.
    """

    if "api_parameters" in before:
        return None
    tree["api_parameters"] = []
    if not _had_api_parameter_columns(before):
        return "api_parameters = []"
    slots = tuple(before.get("slots", ()))
    kept = tuple(
        index
        for index, slot in enumerate(slots)
        if not str(slot.get("slot_id", "")).startswith(_API_SLOT_PREFIX)
    )
    section = _editor(tree)
    rows = tuple(section.get("scan_rows") or ())
    section["scan_rows"] = [
        [row[index] for index in kept if index < len(row)] for row in rows
    ]
    tree["editor"] = section
    return (
        f"api_parameters = [] and {len(slots) - len(kept)} api_ scan column(s) dropped"
    )


def _retire_scan_use_loaded(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    """``6642b29``/``5d889a7``: a hint that only ever seeded one boolean.

    ``scan_use_loaded`` said the rows came from a loaded table rather than
    from running the source, which is one of the ways the source can be out
    of step with its rows -- so it fed ``scan_source_dirty`` and nothing
    else.  Its replacement is stated outright, and this fills it in with the
    formula the migrator used before it was deleted.
    """

    section = _editor(tree)
    if "scan_use_loaded" not in section:
        return None
    used = section.pop("scan_use_loaded")
    if section.get("scan_source_dirty") is None:
        section["scan_source_dirty"] = bool(section.get("scan_source", "")) and (
            _had_api_parameter_columns(before)
            or used is True
            or not bool(section.get("scan_rows") or ())
        )
    tree["editor"] = section
    return (
        "editor.scan_use_loaded -> scan_source_dirty = "
        f"{section['scan_source_dirty']}"
    )


#: In order.  Each reads the document AS FOUND for its condition, so a later
#: step cannot be misled by an earlier one having already filled a field in.
STEPS: tuple[Callable[[dict[str, Any], Mapping[str, Any]], "str | None"], ...] = (
    _adopt_the_current_format_tag,
    _rename_repeat_to_bracket,
    _add_the_missing_run_repeats,
    _add_the_missing_api_parameters,
    _retire_scan_use_loaded,
)


def migrate_tree(before: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Today's document, and what had to change, for one parsed pulse tree."""

    tree = dict(before)
    notes = tuple(note for step in STEPS if (note := step(tree, before)) is not None)
    return tree, notes


class Outcome:
    """One file's result, in the words the operator needs to read."""

    __slots__ = ("path", "verdict", "detail", "backup")

    def __init__(
        self,
        path: Path,
        verdict: str,
        detail: str = "",
        backup: "Path | None" = None,
    ) -> None:
        self.path = path
        self.verdict = verdict
        self.detail = detail
        self.backup = backup


def migrate_file(path: Path) -> Outcome:
    """Migrate one pulse file in place, or explain why it was left alone."""

    if path.suffix.lower() != ".json":
        # The same rule ``read_pulse`` applies.  It matters here because the
        # backups this tool writes hold a pulse the reader would refuse: a
        # second run must pass them by rather than migrate its own copies.
        return Outcome(path, "not a pulse", "not a .json file")
    try:
        # BYTES, not text.  The backup is only a backup if it is what was
        # there: reading through the text layer and writing back through it
        # rewrites every line ending on the way past.
        raw = path.read_bytes()
        before = parse_pulse_tree_json(raw)
    except Exception as error:  # noqa: BLE001 - every reason is worth reporting
        return Outcome(path, "not a pulse", f"{type(error).__name__}: {error}")
    declared = before.get("format") if isinstance(before, Mapping) else None
    if declared != PULSE_TREE_FORMAT and declared not in SUPERSEDED_FORMATS:
        return Outcome(path, "not a pulse", f"format is {declared!r}")
    try:
        read_pulse(path)
        return Outcome(path, "already current")
    except Exception:  # noqa: BLE001 - that it does not read is the whole reason
        pass

    tree, notes = migrate_tree(before)
    try:
        migrated = state_from_tree(tree)
    except Exception as error:  # noqa: BLE001 - reported verbatim, file untouched
        return Outcome(path, "REFUSED", f"{type(error).__name__}: {error}")

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        return Outcome(
            path,
            "REFUSED",
            f"a backup already exists and will not be overwritten: {backup.name}",
        )
    backup.write_bytes(raw)
    write_pulse(path, migrated)
    # The file the operator will open, read by the reader that will open it.
    if read_pulse(path) != migrated:
        return Outcome(
            path,
            "REFUSED",
            "the written file does not read back as the migrated pulse",
            backup,
        )
    return Outcome(path, "migrated", ", ".join(notes), backup)


def pulse_files(targets: tuple[Path, ...]) -> tuple[Path, ...]:
    """Every ``.json`` under the requested files and folders, named once."""

    found: list[Path] = []
    for target in targets:
        if target.is_dir():
            found.extend(sorted(target.rglob("*.json")))
        elif target.exists():
            found.append(target)
    seen: dict[Path, None] = {}
    for path in found:
        seen.setdefault(path.resolve(), None)
    return tuple(seen)


def main(argv: "list[str] | None" = None) -> int:
    """Migrate this workspace's pulses, or the files and folders given."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        targets = tuple(Path(text).expanduser() for text in arguments)
        where = ", ".join(str(target) for target in targets)
    else:
        # The same question the Pulse Editor asks, answered the same way, so
        # this migrates the folder that window opens and not another one.
        workspace = Workspace.discover()
        targets = (workspace.pulses,)
        where = str(workspace.pulses)

    print()
    print("=" * 60)
    print("ZLC - migrate pulse files to the current document schema")
    print(f"Looking in: {where}")
    print("=" * 60)
    print()

    files = pulse_files(targets)
    if not files:
        print("No .json files there. Nothing to migrate.")
        return 0

    counts: dict[str, int] = {}
    for path in files:
        outcome = migrate_file(path)
        counts[outcome.verdict] = counts.get(outcome.verdict, 0) + 1
        detail = f": {outcome.detail}" if outcome.detail else ""
        print(f"  {path.name}")
        print(f"      {outcome.verdict}{detail}")
        if outcome.verdict == "migrated" and outcome.backup is not None:
            print(f"      original kept as {outcome.backup.name}")

    print()
    print(
        f"{len(files)} file(s): "
        + ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    )
    if counts.get("migrated"):
        print(
            f"The {BACKUP_SUFFIX} files are your originals. Open the pulses, "
            "check them, then delete the backups."
        )
    refused = counts.get("REFUSED", 0)
    if refused:
        print(
            f"{refused} file(s) were NOT changed, because this tool does not "
            "know how to bring them forward. They are exactly as they were."
        )
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
