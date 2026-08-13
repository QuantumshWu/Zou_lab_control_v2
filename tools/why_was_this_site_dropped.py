"""Which gate dropped this site?

Run this on the machine that has the frames.  Give it the calibration frames
and the pixel of a spot the site map left uncircled, and it says which of the
detector's four gates rejected that place -- and what it would have taken to
pass.

    python tools/why_was_this_site_dropped.py FRAMES --sigma 1.2 --at 2110,1155
    python tools/why_was_this_site_dropped.py FRAMES --sigma 1.2 --at 40,55 \
        --roi-origin 2070,1100

FRAMES is either a saved dataset (.npz, as the experiment archives signals) or
a plain .npy stack of shape (shots, y, x).  --at takes IMAGE pixels; pass
--roi-origin to give SENSOR pixels instead, which is what the site-map plot's
axes show.  --sigma must be the detection_spot_sigma the calibration ran with,
because two of the gates are derived from it.

The maps below are computed exactly as zlc_atom.nodes.calibration.detect_sites
computes them; this script only reports them per pixel instead of returning a
SiteMap.
"""

from __future__ import annotations

import argparse
from math import sqrt

import numpy as np


def _load(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        stack = np.load(path)
    else:
        from zlc_data.io import load_npz

        stack = np.asarray(load_npz(path).block.values)
    stack = np.asarray(stack, dtype=float)
    stack = stack.reshape(-1, *stack.shape[-2:])
    if stack.ndim != 3 or 0 in stack.shape:
        raise SystemExit(f"{path} is not a stack of images: {stack.shape}")
    return stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--detection-sigma", type=float, default=4.0)
    parser.add_argument(
        "--at",
        action="append",
        required=True,
        help="x,y of a spot the site map missed (repeatable)",
    )
    parser.add_argument("--roi-origin", default=None, help="x,y if --at is in sensor pixels")
    arguments = parser.parse_args()

    from scipy import ndimage
    from scipy.special import erfc
    from scipy.stats import binom

    stack = _load(arguments.frames)
    spot_sigma = float(arguments.sigma)
    detection_sigma = float(arguments.detection_sigma)
    shots = int(stack.shape[0])
    print(f"{shots} shot(s) of {stack.shape[1]}x{stack.shape[2]}, sigma={spot_sigma:g}")

    peak_window = max(3, 2 * int(np.ceil(1.1775 * spot_sigma)) + 1)
    separation = max(2.0, spot_sigma)
    margin = max(2, int(np.ceil(2.0 * spot_sigma)))
    background_sigma = max(4.0 * spot_sigma, spot_sigma + 2.0)

    response = ndimage.gaussian_filter(
        stack, sigma=(0.0, spot_sigma, spot_sigma)
    ) - ndimage.gaussian_filter(stack, sigma=(0.0, background_sigma, background_sigma))
    baseline, lower_quartile = np.quantile(response, (0.5, 0.25), axis=(1, 2), keepdims=True)
    noise = 1.4826 * (baseline - lower_quartile)
    noise = np.maximum(
        noise,
        np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(response), axis=(1, 2), keepdims=True)),
    )
    lit = response >= baseline + detection_sigma * noise
    hits = np.count_nonzero(lit, axis=0)
    conditional = np.sum(np.where(lit, response - baseline, 0.0), axis=0) / np.maximum(hits, 1)

    false_rate = max(float(0.5 * erfc(detection_sigma / sqrt(2.0))), 1e-12)
    required = shots
    for count in range(1, shots + 1):
        if hits.size * float(binom.sf(count - 1, shots, false_rate)) < 0.5:
            required = count
            break

    hit_peak = hits == ndimage.maximum_filter(hits, size=peak_window, mode="nearest")
    bright_peak = conditional == ndimage.maximum_filter(
        conditional, size=peak_window, mode="nearest"
    )
    near_bright = ndimage.binary_dilation(bright_peak, structure=np.ones((3, 3)))

    print(
        f"peak window {peak_window}px, dedupe separation {separation:g}px, "
        f"border margin {margin}px, sightings required {required}"
    )
    print()

    origin = (0, 0)
    if arguments.roi_origin:
        ox, oy = (int(part) for part in arguments.roi_origin.split(","))
        origin = (ox, oy)

    for item in arguments.at:
        x, y = (int(round(float(part))) for part in item.split(","))
        x -= origin[0]
        y -= origin[1]
        if not (0 <= y < hits.shape[0] and 0 <= x < hits.shape[1]):
            print(f"({x}, {y}) is outside the {hits.shape} image")
            continue
        # The operator reads a coordinate off a plot; take the busiest pixel
        # within a spot of it as the place they meant.
        half = max(1, peak_window // 2)
        y0, y1 = max(0, y - half), min(hits.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(hits.shape[1], x + half + 1)
        patch = hits[y0:y1, x0:x1]
        dy, dx = np.unravel_index(int(np.argmax(patch)), patch.shape)
        py, px = y0 + int(dy), x0 + int(dx)

        window = hits[
            max(0, py - half) : py + half + 1, max(0, px - half) : px + half + 1
        ]
        reasons: list[str] = []
        if hits[py, px] < required:
            reasons.append(
                f"too few sightings: {hits[py, px]} < {required} required "
                f"(loaded {100.0 * hits[py, px] / shots:.1f}% of shots)"
            )
        if not hit_peak[py, px]:
            reasons.append(
                f"not the sightings peak of its {peak_window}px window: "
                f"{hits[py, px]} against {int(window.max())} nearby -- a brighter "
                "neighbour inside the window suppresses it"
            )
        if not near_bright[py, px]:
            reasons.append(
                "not next to a peak of conditional brightness: when lit it is "
                "dimmer than a neighbour, which is what a gap between two traps "
                "looks like"
            )
        inside = margin <= py < hits.shape[0] - margin and margin <= px < hits.shape[1] - margin
        if not inside:
            reasons.append(f"within {margin}px of the border, where there is no background")

        label = f"({px + origin[0]}, {py + origin[1]})"
        print(f"{label}: {hits[py, px]} sighting(s), conditional brightness {conditional[py, px]:.1f}")
        if reasons:
            for reason in reasons:
                print(f"    REJECTED -- {reason}")
        else:
            print(
                "    passes every gate, so it was dropped by the dedupe: it "
                f"refined onto a centre already taken within {separation:g}px"
            )
        print()


if __name__ == "__main__":
    main()
