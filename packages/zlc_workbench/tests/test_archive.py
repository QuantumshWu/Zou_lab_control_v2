"""An archive explains itself, and outlives the code that wrote it.

Nothing in the six packages persisted anything: not a frame, not a device
setting, not the pulse that drove the run.  Whatever was taken at the bench was
lost when the process exited.

A reader needs numpy and the standard library, nothing of ours.  An archive
whose metadata is pickled Python objects can only be opened by the code that
wrote it and stops opening the day a class is renamed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from zlc_workbench.archive import FIGURE_SCHEMA, read_archive, write_figure_file


def _sections() -> dict:
    return {
        "provenance": {
            "node": "cm",
            "captured_at": "2026-08-05 12:00:00",
            "devices": {"camera": {"exposure_seconds": 0.02, "gain": 1.0}},
            "acquisition_parameters": {"repeat": 3, "frames_per_cycle": 3},
        },
        "pulse": {"name": "calibration", "camera_windows": 3},
    }


def test_the_record_comes_back_whole(tmp_path) -> None:
    frames = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    path = write_figure_file(
        tmp_path / "shot.npz",
        arrays={"frames": frames},
        sections=_sections(),
    )

    info, arrays = read_archive(path)
    assert info["schema"] == FIGURE_SCHEMA
    assert info["name"] == "shot"
    np.testing.assert_array_equal(arrays["frames"], frames)

    provenance = info["sections"]["provenance"]
    assert provenance["devices"]["camera"]["exposure_seconds"] == 0.02
    assert provenance["acquisition_parameters"]["repeat"] == 3
    assert info["sections"]["pulse"]["camera_windows"] == 3


def test_an_archive_opens_with_numpy_alone(tmp_path) -> None:
    """The durability property: a reader must not need any zlc_* package.

    Run in a fresh interpreter that imports nothing of ours, with allow_pickle
    left at its safe default, exactly as a stranger would open it years later.
    """

    path = write_figure_file(
        tmp_path / "outlives-us.npz",
        arrays={"frames": np.arange(6.0).reshape(2, 3)},
        sections=_sections(),
    )

    reader = (
        "import json, sys, numpy as np\n"
        f"archive = np.load(r'{path}')\n"
        "info = json.loads(str(archive['info']))\n"
        "assert info['sections']['provenance']['devices']['camera']['gain'] == 1.0\n"
        "assert archive['frames'].shape == (2, 3)\n"
        "assert not [m for m in sys.modules if m.startswith('zlc_')]\n"
        "print('readable')\n"
    )
    completed = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "readable" in completed.stdout


def test_arrays_may_not_hide_inside_the_info_document(tmp_path) -> None:
    """An array in the JSON would be unreadable and unmeasurable; say so."""

    with pytest.raises(TypeError, match="arrays belong beside info"):
        write_figure_file(
            tmp_path / "wrong.npz",
            arrays={"y": np.arange(3.0)},
            sections={"panel": {"data": np.arange(3.0)}},
        )


def test_an_interrupted_save_does_not_destroy_the_previous_archive(tmp_path, monkeypatch) -> None:
    good = write_figure_file(
        tmp_path / "keep.npz",
        arrays={"y": np.ones(3)},
        sections={},
    )
    before = good.read_bytes()

    import zlc_workbench.archive as archive_module

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(archive_module, "figure_bytes", explode)
    with pytest.raises(OSError):
        write_figure_file(
            good,
            arrays={"y": np.zeros(3)},
            sections={},
        )

    assert good.read_bytes() == before
    assert tuple(tmp_path.glob("keep*.npz")) == (good,)


def test_an_array_in_device_state_is_refused_with_the_reason() -> None:
    """It used to be stringified before the refusal could fire.

    Provenance ran its own plainifier first, which turned anything it did not
    recognise into str(value) -- so a device reporting an array had it silently
    truncated by numpy's repr, and the archive's message telling the author to
    pass it as an array instead could never be reached.
    """

    import numpy as np
    import pytest

    from zlc_data.figure_archive import _jsonable

    with pytest.raises(TypeError, match="arrays belong beside info"):
        _jsonable({"camera": {"lookup": np.arange(4)}})
