"""Wall-clock nested self-time, bound to ONE object.

Why not cProfile: it inflates functions called few times with a lot of work
inside, which is exactly the shape of every seam in this pipeline -- one
compose, one raster, one publish per frame.  This measures what the clock
says and subtracts timed children, so the rows sum to the frame and a row's
number is the work that row itself did.

Why bound to one object: ``patch(SomeClass, "method")`` times EVERY instance
in the process, and a console always holds more than one renderer.  A curve
panel's profile came back carrying another panel's image work that way, and
the wrong seam looked like the bottleneck for an afternoon.  :func:`watch`
binds to the instance you hand it; :func:`watch_module` is for module-level
functions, where there is only one.

Numba dispatchers carry ``__wrapped__`` and cannot be wrapped this way --
:func:`watch_module` says so out loud instead of silently timing nothing.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


_TOTALS: dict[str, list] = defaultdict(lambda: [0, 0.0])
_local = threading.local()
_lock = threading.Lock()


class _Frame:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children = 0.0


def _timed(name, function):
    def wrapper(*args, **kwargs):
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        frame = _Frame()
        stack.append(frame)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            gross = time.perf_counter() - started
            stack.pop()
            if stack:
                stack[-1].children += gross
            with _lock:
                row = _TOTALS[name]
                row[0] += 1
                row[1] += gross - frame.children

    return wrapper


def reset() -> None:
    """Forget every sample.  Call it after warm-up, before the window."""

    with _lock:
        _TOTALS.clear()


def calls(name: str) -> int:
    """How many times one named seam ran since the last reset."""

    with _lock:
        return int(_TOTALS.get(name, [0])[0])


def watch(instance, *names: str, prefix: str = "") -> list[str]:
    """Time these methods ON THIS INSTANCE only.  Returns what was bound.

    The method is looked up on the type and re-bound as an instance
    attribute, so sibling objects of the same class stay untimed.
    """

    label = prefix or type(instance).__name__
    bound: list[str] = []
    for name in names:
        original = getattr(type(instance), name, None)
        if original is None or not callable(original):
            continue

        def make(function, seam):
            def call(*args, **kwargs):
                return _timed(seam, function)(instance, *args, **kwargs)

            return call

        setattr(instance, name, make(original, f"{label}.{name}"))
        bound.append(name)
    return bound


def watch_module(module, *names: str, prefix: str = "") -> list[str]:
    """Time module-level functions.  Refuses numba dispatchers loudly."""

    label = prefix or module.__name__.rsplit(".", 1)[-1]
    bound: list[str] = []
    for name in names:
        original = getattr(module, name, None)
        if original is None or not callable(original):
            continue
        if hasattr(original, "__wrapped__") or hasattr(original, "py_func"):
            raise TypeError(
                f"{label}.{name} is a compiled dispatcher; wrapping it here "
                "measures nothing. Time its CALLER, or use "
                "ZLC_PLOT_KERNELS=numpy to compare against the reference."
            )
        setattr(module, name, _timed(f"{label}.{name}", original))
        bound.append(name)
    return bound


def rows(seconds: float) -> list[dict]:
    """Every seam, heaviest first, as plain numbers."""

    with _lock:
        items = sorted(
            ((name, count, total) for name, (count, total) in _TOTALS.items()),
            key=lambda row: -row[2],
        )
    return [
        {
            "seam": name,
            "calls": count,
            "self_ms_total": round(total * 1e3, 1),
            "self_ms_per_call": round(total / count * 1e3, 2) if count else 0.0,
            "per_second": round(count / seconds, 1) if seconds > 0 else 0.0,
        }
        for name, count, total in items
    ]


def report(seconds: float, top: int = 20) -> str:
    """The same rows, as a table to read."""

    lines = [
        "%-46s %6s %9s %9s %8s"
        % ("seam", "calls", "self ms", "ms/call", "per s")
    ]
    for row in rows(seconds)[:top]:
        lines.append(
            "%-46s %6d %9.1f %9.2f %8.1f"
            % (
                row["seam"],
                row["calls"],
                row["self_ms_total"],
                row["self_ms_per_call"],
                row["per_second"],
            )
        )
    return "\n".join(lines)
