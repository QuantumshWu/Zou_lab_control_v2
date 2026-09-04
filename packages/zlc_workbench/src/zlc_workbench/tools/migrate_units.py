"""Rewrite saved pulses in the one spelling this project now stores.

A unit had several names.  ``us``, ``µs`` and ``μs`` were three strings for
one thing, and which one a document held depended on which layer had written
it: the pulse model accepted ``us``, the plot canonicalised the micro sign
INTO ``us``, and the plot's timeline canonicalised ``us`` into the micro sign.
There is one registry now, it has one symbol per unit, and a saved document
holds that symbol.

So a pulse written before this reads perfectly -- ``us`` is still an accepted
spelling, and always will be, because it is what a keyboard has -- but it
holds a name the product no longer writes.  This rewrites those documents once
so that what is on disk and what the editor would save are the same bytes,
which is the difference between a Save that changes nothing and a Save that
shows a diff nobody asked for.

A ONE-SHOT.  Run it on each machine that holds pulses, then delete it --
module, launcher and test in one commit.
"""

from __future__ import annotations

from pathlib import Path
import sys

from zlc_pulse.codec import PULSE_TREE_FORMAT, parse_pulse_tree_json

from ..pulse_state import read_pulse, write_pulse
from ..session import Workspace

#: The original, kept beside the file it came from.  Deliberately not a
#: ``.json`` name, so the Open dialog does not offer it as a pulse and a
#: second run of this tool does not try to migrate its own copies.
BACKUP_SUFFIX = ".pre-units"


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
    """Rewrite one pulse in canonical spelling, or say why it was left."""

    if path.suffix.lower() != ".json":
        return Outcome(path, "not a pulse", "not a .json file")
    try:
        # BYTES, not text: the backup is only a backup if it is what was
        # there, and reading through the text layer rewrites every line
        # ending on the way past.
        raw = path.read_bytes()
        before = parse_pulse_tree_json(raw)
    except Exception as error:  # noqa: BLE001 - every reason is worth reporting
        return Outcome(path, "not a pulse", f"{type(error).__name__}: {error}")
    if not hasattr(before, "get") or before.get("format") != PULSE_TREE_FORMAT:
        return Outcome(path, "not a pulse", "not a zlc.pulse document")
    try:
        state = read_pulse(path)
    except Exception as error:  # noqa: BLE001 - reported verbatim, file untouched
        return Outcome(path, "REFUSED", f"{type(error).__name__}: {error}")

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        return Outcome(
            path,
            "REFUSED",
            f"a backup already exists and will not be overwritten: {backup.name}",
        )
    # Write it and compare: the reader already normalised every spelling, so
    # what comes out is canonical by construction.  If that is byte-identical
    # to what was there, the file was already current and is left untouched
    # rather than given a backup it does not need.
    backup.write_bytes(raw)
    write_pulse(path, state)
    if path.read_bytes() == raw:
        backup.unlink()
        return Outcome(path, "already current")
    if read_pulse(path) != state:
        return Outcome(
            path, "REFUSED", "the written file does not read back the same", backup
        )
    changed = tuple(
        sorted({period.unit for period in state.sequence.periods})
    ) if state.sequence is not None else ()
    return Outcome(path, "rewritten", "units: " + ", ".join(changed), backup)


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
    print("ZLC - rewrite saved pulses in one unit spelling")
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
        if outcome.verdict == "rewritten" and outcome.backup is not None:
            print(f"      original kept as {outcome.backup.name}")

    print()
    print(
        f"{len(files)} file(s): "
        + ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    )
    if counts.get("rewritten"):
        print(
            f"The {BACKUP_SUFFIX} files are your originals. Open the pulses, "
            "check them, then delete the backups."
        )
    refused = counts.get("REFUSED", 0)
    if refused:
        print(
            f"{refused} file(s) were NOT changed. They are exactly as they were."
        )
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
