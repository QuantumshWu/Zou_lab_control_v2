"""Where a committed frame's milliseconds actually go.

``run_session`` says a revision costs N ms; this says WHICH work that is.
Every live update is profiled and the samples are folded into named buckets
-- the projection, the artist updates, Matplotlib's own draw (split into
image, text/ticks, colorbar, the distribution rail, and the rest), and the
compose/blit that turns a canvas into a front -- so an optimisation can be
aimed instead of guessed.

Run:  python -m bench.plot_perf.attribute [--only substring] [--updates N]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import traceback

import matplotlib

matplotlib.use("Agg", force=True)

from .cases import catalog  # noqa: E402


#: (bucket, predicate over "module:function") in priority order.  The first
#: match wins, so the specific buckets precede the general ones.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compose/blit", ("rendering.py:compose", "rendering.py:present",
                      "raster.py:_compose", "_image_raster.py:",)),
    ("mpl:text/ticks", ("text.py:", "axis.py:", "ticker.py:", "textpath.py:",
                        "font_manager.py:", "_mathtext", "backend_agg.py:draw_text",
                        "backend_agg.py:_prepare_font", "backend_agg.py:get_text_width_height_descent")),
    ("mpl:image", ("image.py:", "backend_agg.py:draw_image")),
    ("mpl:path/line", ("lines.py:", "path.py:", "patches.py:", "collections.py:",
                       "backend_agg.py:draw_path", "transforms.py:")),
    ("mpl:draw other", ("backend_agg.py:", "figure.py:", "axes/_base.py:",
                        "artist.py:", "spines.py:", "colorbar.py:")),
    ("zlc:projection", ("data_view.py:", "_fit_projection.py:", "specs.py:",
                        "snapshot", "aggregate")),
    ("zlc:rendering", ("rendering.py:", "session.py:", "selectors.py:",
                       "ticks.py:", "layout.py:")),
    ("numpy", ("{built-in method numpy", "numpy/", "numpy\\\\")),
)


def _bucket(name: str) -> str:
    for bucket, needles in _BUCKETS:
        if any(needle in name for needle in needles):
            return bucket
    return "other"


def _entry_name(entry) -> str:
    path, line, function = entry
    if path in ("~", ""):
        return f"builtin:{function}"
    parts = path.replace("\\", "/").split("/")
    tail = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return f"{tail}:{function}"


def attribute(case, updates: int) -> dict:
    from zlc_plot import PlotSession

    feed = case.feed()
    session = PlotSession(feed.next(), case.spec())
    try:
        session.set_size("4x4")
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
        name = _entry_name(entry)
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
                caller_name = _entry_name(caller)
                bucket = _bucket(caller_name)
                buckets[bucket] = buckets.get(bucket, 0.0) + share
                leaders.append((share, f"{name} <- {caller_name}"))
            continue
        buckets[_bucket(name)] = buckets.get(_bucket(name), 0.0) + tt
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
