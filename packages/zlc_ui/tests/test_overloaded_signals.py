"""No signal here carries the int/str overload pair.

``currentIndexChanged`` and ``activated`` each exist twice on a QComboBox --
once taking an int, once a str.  PyQt binds a bare ``connect`` to the
signal's default overload only (``currentIndexChanged(int)``), so the
overload is not by itself a double delivery; what it is, is a second
signature to get wrong at every ``connect``, and one the product has no
use for.  The house combo exists for its own reasons -- a lazy popup built
on first open, one owned item model, the shared Fluent styling, and the
popup geometry decided from the numbers -- and declares each of its
signals once.  Nothing in this package instantiates a QComboBox, so no
connect site anywhere in it can pick the wrong overload.

This used to be guarded by scanning the source for ``.activated.connect(``
and demanding a subscript.  That rule reads the signal's NAME and cannot see
the receiver's TYPE, and the two are not the same question: a QListView's
``activated`` has exactly one signature and cannot be ambiguous, so the scan
flagged two correct lines as offences.  It also never saw the connections
that go through a helper -- ``_connect_change(widget.activated, on_change)``
hands the bound signal over as an object, and no regex on the connect site
can follow it.  The guard is on the precondition instead, which the
metaobject can check: no widget with an overloaded signal is made here, and
every signal the house widgets declare has one signature.
"""

from __future__ import annotations

import inspect
import pathlib
import re


SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "zlc_ui"

#: Qt widgets whose ``activated``/``currentIndexChanged`` carry the int/str
#: pair.  Referring to one of these types is fine -- an isinstance tuple names
#: QComboBox without ever making one -- but constructing one brings the
#: overload into the package.
OVERLOADED_WIDGETS = ("QComboBox", "QFontComboBox")

#: The signal names that are overloaded on those widgets.
HAZARDOUS = ("activated", "currentIndexChanged", "highlighted", "textActivated")


def test_no_overloaded_widget_is_constructed_here() -> None:
    """The hazard cannot be connected to if it is never made."""

    pattern = re.compile(
        r"\b(?:QtWidgets\.)?(" + "|".join(OVERLOADED_WIDGETS) + r")\s*\("
    )
    offenders = [
        f"{path.relative_to(SRC)}:{number}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == [], "\n".join(offenders)


def _signatures(cls: type) -> list[str]:
    meta = getattr(cls, "staticMetaObject", None)
    if meta is None:
        return []
    return [
        bytes(meta.method(index).methodSignature()).decode()
        for index in range(meta.methodCount())
        if meta.method(index).methodType() == 1  # Signal
    ]


def test_every_signal_this_package_declares_has_one_signature() -> None:
    """The house widgets that stand in for the Qt ones, checked by Qt itself.

    A source scan cannot tell a single-signature ``activated`` from an
    overloaded one; the metaobject can, and it is the same thing PyQt reads
    when it decides what a bare ``connect`` binds.
    """

    from zlc_ui import fluent

    # Through __all__, because the package loads its widgets lazily: reading
    # vars() before anything has asked for them sees an empty namespace, and
    # a guard that examines nothing passes for the wrong reason.
    checked = 0
    for name in getattr(fluent, "__all__", ()):
        value = getattr(fluent, name)
        if not inspect.isclass(value) or not value.__module__.startswith("zlc_ui"):
            continue
        signatures = _signatures(value)
        for hazard in HAZARDOUS:
            overloads = [
                signature
                for signature in signatures
                if signature.split("(", 1)[0] == hazard
            ]
            if not overloads:
                continue
            checked += 1
            assert len(overloads) == 1, (
                f"{value.__name__}.{hazard} has {len(overloads)} signatures "
                f"{overloads}: a connect site could pick the wrong one"
            )
    # A guard that would pass on an empty search is not a guard.
    assert checked >= 2, f"only {checked} signals were examined"
