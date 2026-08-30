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

Wall AND cpu, always both.  A console runs a producer, several panels and a
parallel pool on one machine, so a seam's thread is descheduled inside it
constantly: a function whose own work is 0.4 ms was reported at 4.8 ms of
wall time, and read as "this is slow" when it was "this waited".  The two
columns are the difference between an optimisation target and a scheduling
fact -- ``time.thread_time`` is per-thread CPU, so waiting does not count.

ONE TRAP in that column: ``time.thread_time`` counts the CALLING thread only.
A seam that hands work to a compiled ``parallel=True`` kernel does its work on
libomp's worker threads, so it reads like a seam that waited -- 4 per cent for
``_view_filling_rgba_front``, whose whole job is to call one.  Low cpu% means
"this thread was not running": either it waited, OR it is a compiled parallel
region.  Which one it is, is a fact about the callee, not about the number.
"""
from __future__ import annotations

import inspect
import threading
import time
from collections import defaultdict


_TOTALS: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
_local = threading.local()
_lock = threading.Lock()


class _Frame:
    __slots__ = ("children", "cpu_children")

    def __init__(self) -> None:
        self.children = 0.0
        self.cpu_children = 0.0


def _timed(name, function):
    def wrapper(*args, **kwargs):
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        frame = _Frame()
        stack.append(frame)
        started = time.perf_counter()
        cpu_started = time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            gross = time.perf_counter() - started
            cpu_gross = time.thread_time() - cpu_started
            stack.pop()
            if stack:
                stack[-1].children += gross
                stack[-1].cpu_children += cpu_gross
            with _lock:
                row = _TOTALS[name]
                row[0] += 1
                row[1] += gross - frame.children
                row[2] += cpu_gross - frame.cpu_children
                row[3] += gross
                row[4] += cpu_gross

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
        # A STATICMETHOD reached through the class is a plain function, and
        # binding a wrapper as an instance attribute means the call site no
        # longer supplies the instance -- so passing one injects an argument
        # the function does not take.  Wrapping _native_draw that way raised
        # inside every full draw, the panels stopped presenting, and the
        # bench reported 0.1 frames per second as if it were a measurement.
        raw = inspect.getattr_static(type(instance), name, None)
        takes_self = not isinstance(raw, (staticmethod, classmethod))

        def make(function, seam, pass_self):
            if pass_self:
                def call(*args, **kwargs):
                    return _timed(seam, function)(instance, *args, **kwargs)
            else:
                def call(*args, **kwargs):
                    return _timed(seam, function)(*args, **kwargs)
            return call

        setattr(instance, name, make(original, f"{label}.{name}", takes_self))
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


def watch_attribute(instance, *names: str, prefix: str = "") -> list[str]:
    """Time callable attributes already bound on one object.

    PlotPanelPort owns projection/presentation callbacks as instance fields,
    not class methods.  Reaching for the callback's class would time every
    panel (and lambdas have no useful class seam), so this is the instance-
    field counterpart of :func:`watch`.
    """

    label = prefix or type(instance).__name__
    bound: list[str] = []
    for name in names:
        original = getattr(instance, name, None)
        if original is None or not callable(original):
            continue
        setattr(instance, name, _timed(f"{label}.{name}", original))
        bound.append(name)
    return bound


def rows(seconds: float) -> list[dict]:
    """Every seam, heaviest first, as plain numbers."""

    with _lock:
        items = sorted(
            (
                (name, count, total, cpu, gross, gross_cpu)
                for name, (count, total, cpu, gross, gross_cpu) in _TOTALS.items()
            ),
            key=lambda row: -row[2],
        )
    return [
        {
            "seam": name,
            "calls": count,
            "self_ms_total": round(total * 1e3, 1),
            "self_ms_per_call": round(total / count * 1e3, 2) if count else 0.0,
            "cpu_ms_per_call": round(cpu / count * 1e3, 2) if count else 0.0,
            "gross_ms_total": round(gross * 1e3, 1),
            "gross_ms_per_call": round(gross / count * 1e3, 2) if count else 0.0,
            "gross_cpu_ms_per_call": (
                round(gross_cpu / count * 1e3, 2) if count else 0.0
            ),
            # What fraction of the wall time this thread was actually running.
            # Well under 1 means the seam WAITED; optimising it would not help.
            "cpu_share": round(cpu / total, 2) if total > 0 else 0.0,
            "per_second": round(count / seconds, 1) if seconds > 0 else 0.0,
        }
        for name, count, total, cpu, gross, gross_cpu in items
    ]


def report(seconds: float, top: int = 20) -> str:
    """The same rows, as a table to read.

    ``cpu%`` is what separates a slow seam from a waiting one: at 100 per
    cent the thread ran the whole time and the number is work; well below
    it the thread was not running -- descheduled, or inside a compiled
    parallel region whose work is on other threads.  See the module note.
    """

    lines = [
        "%-46s %6s %9s %10s %9s %5s %7s"
        % ("seam", "calls", "wall ms", "wall/call", "cpu/call", "cpu%", "per s")
    ]
    for row in rows(seconds)[:top]:
        lines.append(
            "%-46s %6d %9.1f %10.2f %9.2f %4.0f%% %7.1f"
            % (
                row["seam"],
                row["calls"],
                row["self_ms_total"],
                row["self_ms_per_call"],
                row["cpu_ms_per_call"],
                row["cpu_share"] * 100.0,
                row["per_second"],
            )
        )
    return chr(10).join(lines)
