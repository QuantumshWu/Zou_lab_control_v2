"""The icon file holds the frame drawn for each size, never one frame resampled.

Run explicitly (``python -m pytest tools/tests -p no:cacheprovider``): the
tool is a development script, not a product package, so the suite's
``testpaths`` does not collect it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import make_app_icon  # noqa: E402


def test_every_size_in_the_icon_is_the_frame_drawn_for_it(tmp_path: Path) -> None:
    """A size the encoder is not handed, it resamples from the largest frame.

    ``save`` used to receive only the 256 px frame and the list of sizes, so
    Pillow copied and thumbnailed that one frame for every other entry and
    the eight smaller frames drawn at their own size were discarded -- the
    opposite of what the tool promises.
    """

    from PIL import Image

    if not Path(make_app_icon.TYPEFACE).is_file():
        pytest.skip("the chrome typeface is not installed on this machine")
    out = tmp_path / "zlc.ico"
    assert make_app_icon.main(["--out", str(out)]) == 0
    with Image.open(out) as icon:
        assert set(icon.ico.sizes()) == {
            (side, side) for side in make_app_icon.SIZES
        }
        for side in make_app_icon.SIZES:
            stored = np.asarray(icon.ico.getimage((side, side)).convert("RGBA"))
            drawn = np.asarray(make_app_icon.render(side).convert("RGBA"))
            assert stored.shape == drawn.shape == (side, side, 4)
            np.testing.assert_array_equal(
                stored, drawn, err_msg=f"{side} px is not the frame drawn for it"
            )
