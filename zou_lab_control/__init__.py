"""One bootstrap for this checkout and the installed product."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Mapping
import os
import sys
import tomllib


__all__ = ["DISTRIBUTION_NAME", "ROOT", "entry_specs"]

DISTRIBUTION_NAME = "zou-lab-control"
# In a checkout this is the repository root; in a wheel it is site-packages.
ROOT = Path(__file__).resolve().parent.parent
_LAYERS = (
    "zlc_data",
    "zlc_durable",
    "zlc_runtime",
    "zlc_plot",
    "zlc_ui",
    "zlc_pulse",
    "zlc_atom",
    "zlc_workbench",
)


def _configure_compiled_worker_threads() -> None:
    """Size the native pool, bound each ZLC worker team, and let it sleep.

    Parallel compiled kernels in this product run on Numba's OpenMP pool.
    Its worker threads busy-wait after each parallel
    region rather than sleeping, so once the first kernel has run the pool
    keeps burning cores for as long as the process lives.  Measured on this
    machine: arming the camera took the console from 5 per cent of one core
    to over a thousand -- sixteen native threads at 60-75 per cent each --
    and stopping the sequencer did not bring it back down, because the spin
    is not tied to any work.

    ``OMP_WAIT_POLICY=PASSIVE`` is the knob that answers it (``KMP_BLOCKTIME``
    is Intel's and this is LLVM's runtime, which ignores it).  It costs
    nothing measurable: the scene raster over 20 camera turns went from a
    10.94 ms minimum to 10.36, because the work gets a machine that is not
    already busy spinning.

    A panel is already one independent worker.  Giving every panel all
    logical CPUs oversubscribes them, while shrinking the PROCESS pool to
    four makes four same-shot panels wait on one tiny pool.  The process now
    retains the machine's logical capacity and ZLC's Raster/analysis workers
    mask their own native team to four; four panels can therefore make useful
    progress together without each claiming the machine.  Explicit operator
    thread settings remain authoritative.

    Set both policies here because the environment must be in place before
    Numba/OpenMP initializes, and this bootstrap is what every entry point
    imports first.  ``setdefault`` preserves an operator's explicit choice.
    """

    logical = max(1, os.cpu_count() or 1)
    authored = os.environ.get("NUMBA_NUM_THREADS")
    if authored is None:
        os.environ["NUMBA_NUM_THREADS"] = str(logical)
        os.environ.setdefault("ZLC_NUMBA_WORKER_THREADS", str(min(4, logical)))
    else:
        os.environ.setdefault("ZLC_NUMBA_WORKER_THREADS", authored)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


_configure_compiled_worker_threads()


def _activate_checkout() -> None:
    """Make this checkout authoritative when its bootstrap was imported."""

    if not (ROOT / "pyproject.toml").is_file():
        return
    sources = tuple(ROOT / "packages" / name / "src" for name in _LAYERS)
    missing = tuple(path for path in sources if not path.is_dir())
    if missing:
        raise ImportError(
            "Zou Lab Control checkout is missing "
            + ", ".join(str(path) for path in missing)
        )
    stale = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("zlc_") or "." in name:
            continue
        origin = getattr(module, "__file__", None)
        if origin is not None and not any(
            source in Path(origin).resolve().parents for source in sources
        ):
            stale.append(f"{name} <- {origin}")
    if stale:
        raise ImportError(
            "zlc packages were imported before the current checkout bootstrap:\n  "
            + "\n  ".join(sorted(stale))
        )
    for source in reversed(sources):
        text = str(source)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_activate_checkout()


def entry_specs(group: str) -> Mapping[str, str]:
    """Return one entry-point group from source manifest or installed metadata."""

    manifest = ROOT / "pyproject.toml"
    if manifest.is_file():
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        groups = document.get("project", {}).get("entry-points", {})
        entries = groups.get(group)
        if not isinstance(entries, dict) or not entries:
            raise RuntimeError(f"product manifest has no {group!r} entry-point group")
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in entries.items()
        ):
            raise RuntimeError(f"product manifest {group!r} entries must be text")
        return dict(sorted(entries.items()))

    try:
        installed = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"{DISTRIBUTION_NAME} is not installed") from exc
    entries = {
        item.name: item.value
        for item in installed.entry_points
        if item.group == group
    }
    if not entries:
        raise RuntimeError(f"installed product has no {group!r} entry-point group")
    return dict(sorted(entries.items()))
