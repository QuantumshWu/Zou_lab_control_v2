"""Bring pulse files written before the bracket split up to today's schema.

A pulse is experiment content: it lives in the operator's workspace, not in
the package, and it outlives the build that wrote it.  The reader, by design,
is a strict whitelist -- a field it does not know is an error, not something
to ignore -- so ``663de9b``, which renamed ``repeat`` to ``bracket`` and made
``run_repeats`` required, turned every pulse already sitting on a disk into

    cannot open imaging_template.json: unknown pulse field(s): repeat

The three tracked template JSONs were edited by hand in that commit.  The
pulses in an operator's workspace were not, and this is the other half of
that change.  It is a ONE-SHOT: run it once on each machine that holds
pulses, then delete it -- module, launcher and test in one commit -- exactly
as the previous migration was removed in ``5d889a7``.

A second grammar change followed it: a pulse now declares its ``config``
parameters and the file it refreshes them from.  A document written before
that declares neither, which is the same "unknown pulse field(s)" wall from
the other side, so this tool carries both steps and one run takes a pulse
written before the bracket split all the way to today.

It knows those two changes and nothing else.  A document carrying anything
else it cannot account for is refused in the reader's own words and left
byte-for-byte as it was found, because a migration that silently drops what
it cannot explain turns "will not open" into "opens, and something is
missing".
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


def _rename_repeat_to_bracket(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    """One word that meant three things, split into three fields.

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


def _declare_no_config_binding(
    tree: dict[str, Any], before: Mapping[str, Any]
) -> str | None:
    """A pulse written before config parameters existed declares none.

    Empty is the honest reconstruction and the only safe one: the document
    never named a field as configured, so nothing in it may be refreshed from
    a file, and no file is named.  Whoever wants one says so in the editor.
    """

    added: list[str] = []
    if "config_parameters" not in before:
        tree["config_parameters"] = []
        added.append("config_parameters = []")
    if "config_source" not in before:
        tree["config_source"] = ""
        added.append('config_source = ""')
    return ", ".join(added) if added else None


#: In order.  Each reads the document AS FOUND for its condition, so a later
#: step cannot be misled by an earlier one having already filled a field in.
STEPS: tuple[Callable[[dict[str, Any], Mapping[str, Any]], "str | None"], ...] = (
    _rename_repeat_to_bracket,
    _add_the_missing_run_repeats,
    _declare_no_config_binding,
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
    if declared != PULSE_TREE_FORMAT:
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
