"""The status vocabulary, owned apart from anything that draws it.

A status severity is a plain word, and the two sides that must agree on it
sit on opposite sides of the Qt boundary: the strip that colours it, and the
headless presenters that choose it.  It used to be neither's -- bare strings
written at every call site and checked only inside the widget, by raising --
so one mistyped word ("info") travelled from a panel edit into a Qt slot
and took the whole console down with it.  Here it belongs to the package
both sides already depend on, in a module that imports nothing, so
referencing it costs no Qt; the strip builds its colour table from it.

The words a summary or status line is built from live here for the same
reason: a count, the mark between a line's parts and a declared name read
out as words are written by headless presenters and by plugin editors
alike, and every window has to word them one way.
"""

from __future__ import annotations

#: Every severity a console status line may carry, worst last.
STATUS_SEVERITIES: tuple[str, ...] = ("idle", "warning", "task", "error")

#: Every state a status DOT shows for something that lasts -- a node, a board,
#: a session, a device field, the console.
STATUS_STATES: tuple[str, ...] = ("idle", "running", "warning", "error")

#: Between the parts of one summary line -- a file and its period count, a
#: panel count and a node count, what is shown and what is hidden.  One
#: mark for every window, so no two summaries part their words differently.
SUMMARY_SEPARATOR = " · "


def counted(count: int, noun: str, plural: str = "") -> str:
    """``count`` of ``noun``, in the number it takes: "1 period", "3 periods".

    ``plural`` names an irregular plural ("axis", "axes"); otherwise the
    noun takes an "s".  Every count a window words goes through here, so
    none reads "3 period(s)" beside another that says "3 periods".
    """

    count = int(count)
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def sentence(text: str) -> str:
    """``text`` written as a sentence of its own: ending in a full stop.

    A problem shown on its own line under the row it is about reads as one
    sentence, whoever words it.  Not capitalised: a problem may start with
    a name ("load_20ms reads frame 2 twice").  Empty stays empty.
    """

    text = str(text).rstrip()
    return text if not text or text.endswith((".", "?", "!", "…")) else f"{text}."


#: The words a declared name keeps in capitals wherever it stands.
_ACRONYMS = frozenset({"api", "dac", "daq", "id", "psf", "rf", "roi", "slm", "ttl"})


def name_words(name: str) -> str:
    """A snake_case ``name`` read out as words: "SLM feedback", "Fit params".

    Sentence case -- the first word capitalised, the rest as declared --
    with the lab's acronyms (SLM, DAQ, API, ROI, DAC, PSF, ID, RF, TTL) in
    capitals, so a node kind, a device argument and a result name read
    like every form label beside them.  Not ``str.title()``: that
    capitalises every word and writes SLM as "Slm".
    """

    words = [
        word.upper() if word.lower() in _ACRONYMS else word
        for word in str(name).replace("_", " ").split()
    ]
    if words:
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


__all__ = ["STATUS_SEVERITIES", "STATUS_STATES", "SUMMARY_SEPARATOR", "counted", "name_words", "sentence"]
