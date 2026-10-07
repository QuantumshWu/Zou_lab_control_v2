"""Where a committed frame's milliseconds actually go.

``run_session`` says a revision costs N ms; this says WHICH work that is.
Every live update is profiled and the samples are folded into named buckets
-- the projection, the fit, the artist updates, Matplotlib's own draw (split
into image, text/ticks, paths/lines, and the rest), and the compose/blit
that turns a canvas into a front -- so an optimisation can be aimed instead
of guessed.

Run:  python -m bench.plot_perf.attribute [--only substring] [--updates N]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import traceback

import matplotlib

matplotlib.use("Agg", force=True)

from .cases import catalog, open_session  # noqa: E402


#: (bucket, predicate over "path:function") in priority order.  The first
#: match wins, so the specific buckets precede the general ones.  A bucket
#: reads the file's whole normalised path, so a package directory claims
#: every module under it however deep: ``zlc_plot/`` all of the plot
#: package, ``/numpy/`` numpy's ``_core``/``lib`` subpackages, and the
#: trailing ``matplotlib/`` whatever of Matplotlib's draw (Normalize and
#: Colormap calls, ``axes/_axes.py``, ``backend_bases.py``) the specific
#: Matplotlib buckets left.  Matplotlib's specific needles carry their
#: package directory because a zlc module shares some of its file names
#: (``_kinds/image.py``, ``zlc_data/axis.py``).
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compose/blit", ("rendering.py:compose", "rendering.py:present",
                      "raster.py:_compose", "_image_raster.py:",)),
    ("mpl:text/ticks", ("matplotlib/text.py:", "matplotlib/axis.py:", "matplotlib/ticker.py:",
                        "matplotlib/textpath.py:", "matplotlib/font_manager.py:",
                        "matplotlib/_mathtext", "backend_agg.py:draw_text",
                        "backend_agg.py:_prepare_font", "backend_agg.py:get_text_width_height_descent")),
    ("mpl:image", ("matplotlib/image.py:", "backend_agg.py:draw_image")),
    ("mpl:path/line", ("matplotlib/lines.py:", "matplotlib/path.py:", "matplotlib/patches.py:",
                       "matplotlib/collections.py:", "backend_agg.py:draw_path",
                       "matplotlib/transforms.py:")),
    ("mpl:draw other", ("matplotlib/",)),
    ("zlc:projection", ("data_view.py:", "_fit_projection.py:", "specs.py:",
                        "snapshot", "aggregate", "zlc_data/")),
    ("zlc:fit", ("zlc_plot/fit.py:", "_session_fit.py:", "_fit_compiled.py:",
                 "_fit_radial.py:", "_fit_scene.py:")),
    ("zlc:rendering", ("zlc_plot/",)),
    ("numpy", ("/numpy/",)),
)


def _bucket(name: str) -> str:
    for bucket, needles in _BUCKETS:
        if any(needle in name for needle in needles):
            return bucket
    return "other"


def _entry_names(entry) -> tuple[str, str]:
    """(bucketed, shown): the whole normalised path, and its last two parts."""
    path, _line, function = entry
    if path in ("~", ""):
        name = f"builtin:{function}"
        return name, name
    path = path.replace("\\", "/")
    return f"{path}:{function}", f"{'/'.join(path.split('/')[-2:])}:{function}"


def attribute(case, updates: int) -> dict:
    feed = case.feed()
    session = open_session(case, feed)
    try:
        session.rgba()
        for _ in range(2):                      # warm the caches
            session.update_data(feed.next())
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(updates):
            session.update_data(feed.next())
        profiler.disable()
    finally:
        session.close()

    stats = pstats.Stats(profiler)
    total = stats.total_tt
    buckets: dict[str, float] = {}
    leaders: list[tuple[float, str]] = []
    for entry, (_cc, _nc, tt, _ct, callers) in stats.stats.items():
        path, name = _entry_names(entry)
        if name.startswith("builtin:") and callers:
            # A builtin's self time belongs to whoever ASKED for it: a
            # ufunc reduce is the caller's reduction, not a bucket of its
            # own.  Split it across callers by call count, so the table
            # names work an optimisation can actually aim at.
            total_calls = sum(
                item[0] if isinstance(item, tuple) else item
                for item in callers.values()
            ) or 1
            for caller, item in callers.items():
                count = item[0] if isinstance(item, tuple) else item
                share = tt * count / total_calls
                caller_path, caller_name = _entry_names(caller)
                bucket = _bucket(caller_path)
                buckets[bucket] = buckets.get(bucket, 0.0) + share
                leaders.append((share, f"{name} <- {caller_name}"))
            continue
        buckets[_bucket(path)] = buckets.get(_bucket(path), 0.0) + tt
        leaders.append((tt, name))
    leaders.sort(reverse=True)
    return {
        "case": case.name,
        "updates": updates,
        "total_ms_per_update": round(total * 1e3 / max(updates, 1), 2),
        "buckets_ms_per_update": {
            bucket: round(seconds * 1e3 / max(updates, 1), 2)
            for bucket, seconds in sorted(
                buckets.items(), key=lambda item: -item[1]
            )
        },
        "top_self_ms_per_update": [
            (round(seconds * 1e3 / max(updates, 1), 2), name)
            for seconds, name in leaders[:18]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--updates", type=int, default=8)
    arguments = parser.parse_args()
    for case in catalog():
        if arguments.only and arguments.only not in case.name:
            continue
        print(f"=== {case.name} ===", flush=True)
        try:
            report = attribute(case, arguments.updates)
        except Exception:
            print(traceback.format_exc(), flush=True)
            continue
        print(f"  total {report['total_ms_per_update']} ms/update")
        for bucket, value in report["buckets_ms_per_update"].items():
            print(f"    {bucket:18s} {value:8.2f}")
        print("  top self time:")
        for value, name in report["top_self_ms_per_update"]:
            print(f"    {value:8.2f}  {name}")


if __name__ == "__main__":
    main()
