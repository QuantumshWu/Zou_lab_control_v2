"""A record of what one gesture actually did, written where a hand made it.

Every synthesized hand this was chased with measured something other than
what the operator sees.  A real hand is the only instrument that has been
right so far, so this rides along with it: switched off by default, and
when switched on it writes one line per gesture -- press to release --
saying what the product did at each step.

    set ZLC_GESTURE_LOG=1        (a path, to choose the file)

The line answers, for that one gesture:

  caught          did the view (or the selector, or the camera) ever move
  press           when the button went down, and what the press resolved:
                  the axes it landed on, the gesture it built, or nothing
  moves           one entry per move that reached the session, each with
                  the wait before it, whether the rate limiter admitted
                  it, and whether the thing the gesture owns changed

Nothing here decides anything; it only writes down what happened, so that
a gesture that did not catch can be read rather than guessed at.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from typing import Any


LINE_END = chr(10)


def _enabled() -> str | None:
    value = os.environ.get("ZLC_GESTURE_LOG")
    if not value:
        return None
    if value in ("1", "true", "True", "yes", "on"):
        return str(pathlib.Path.cwd() / "gesture_log.jsonl")
    return value


class GestureLog:
    """One line per gesture, appended when the button comes up.

    Costs nothing while off: every entry point returns immediately on a
    None path, which is what the whole product sees unless the operator
    asked for a recording.
    """

    def __init__(self) -> None:
        self._path = _enabled()
        self._lock = threading.Lock()
        self._open: dict[int, dict[str, Any]] = {}

    @property
    def on(self) -> bool:
        return self._path is not None

    # ------------------------------------------------------------ press
    def press(self, owner: object, **facts: Any) -> None:
        if self._path is None:
            return
        with self._lock:
            self._open[id(owner)] = {
                "started": time.strftime("%H:%M:%S"),
                "press": {"at": time.perf_counter(), **facts},
                "moves": [],
            }

    def press_resolved(self, owner: object, **facts: Any) -> None:
        if self._path is None:
            return
        with self._lock:
            record = self._open.get(id(owner))
            if record is not None:
                record["press"].update(facts)
                record["press"]["served_ms"] = round(
                    1e3 * (time.perf_counter() - record["press"]["at"]), 1
                )

    # ------------------------------------------------------------- move
    def move(self, owner: object, **facts: Any) -> None:
        if self._path is None:
            return
        with self._lock:
            record = self._open.get(id(owner))
            if record is None:
                # A move with no press before it: worth seeing, because a
                # gesture that never caught looks exactly like this.
                record = self._open[id(owner)] = {
                    "started": time.strftime("%H:%M:%S"),
                    "press": None,
                    "moves": [],
                }
            if len(record["moves"]) < 200:
                record["moves"].append(facts)

    def widget(self, owner: object, what: str, event: object) -> None:
        """A stamp taken in the widget, before the product does anything.

        The two sides of one gesture are joined by the clock, which is
        enough to see whether a press that felt late was already late
        before the session saw it.
        """

        if self._path is None:
            return
        line = {
            "widget": what,
            "at": round(time.perf_counter(), 6),
            "clock": time.strftime("%H:%M:%S"),
        }
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + LINE_END)
        except OSError:
            pass

    # ---------------------------------------------------------- release
    def release(self, owner: object, **facts: Any) -> None:
        if self._path is None:
            return
        with self._lock:
            record = self._open.pop(id(owner), None)
        if record is None:
            return
        moves = record["moves"]
        followed = [
            index for index, move in enumerate(moves) if move.get("moved")
        ]
        record["release"] = facts
        record["caught"] = bool(followed)
        record["moves_before_it_followed"] = (
            None if not followed else followed[0]
        )
        record["move_count"] = len(moves)
        press = record.get("press")
        if press is not None:
            press.pop("at", None)
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


#: One recorder for the process.  A console holds several panels and the
#: line says which one it came from, so they share it.
LOG = GestureLog()
