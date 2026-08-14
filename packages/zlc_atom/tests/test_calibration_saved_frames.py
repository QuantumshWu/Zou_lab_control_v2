"""Frames saved as they arrive, and calibrated again without the bench."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration.outputs import CAPTURE_PREVIEW_DECLARATION
from zlc_atom.nodes.calibration.task import (
    FRAMES_FROM_FOLDER,
    CalibrationRequest,
    CalibrationTask,
    read_saved_samples,
)
from zlc_data.figure_archive import read_archive, read_dataset
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from tests.fakes import FakePlane
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE
from test_installation_and_nodes import _calibration_request


def _task(request: CalibrationRequest, directory: Path) -> CalibrationTask:
    installation = create_installation("virtual")
    return CalibrationTask(
        camera=installation.device("camera"),
        sequencer=installation.device("sequencer"),
        request=request,
        pulse_sequence=IMAGING_PULSE_RESOURCE.value,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        signal_plane=FakePlane(),
        artifact_directory=directory,
    )


def test_saved_samples_are_written_as_they_arrive_and_calibrate_again(
    tmp_path: Path,
) -> None:
    """The whole point: retune detection without holding the bench again.

    Every sample is written in the acquisition loop, not after it, so a run
    that is cancelled or that fails its analysis still leaves the frames it
    paid atoms for.  Each file is the archive Panel Edit writes -- the picture
    and the numbers behind it -- so the same viewer opens them, and the run
    record inside is where the crop and the exposure are.
    """

    request = replace(_calibration_request(repeats=12), save_frames=True)
    first = _task(request, tmp_path).run()

    folder = first.artifact_path.with_suffix("") / "frames"
    archives = sorted(folder.glob("sample_*.npz"))
    pictures = sorted(folder.glob("sample_*.png"))
    assert len(archives) == 12
    assert len(pictures) == 12

    info, arrays = read_archive(archives[0])
    snapshot = read_dataset(info, arrays, "data")
    assert np.asarray(snapshot.block.values).shape[:2] == (1, 3)
    assert info["sections"]["run_chain"], "the crop and exposure travel with the frames"

    # Read straight back: the same three frames per sample, in order.
    cycles, run_record = read_saved_samples(folder)
    assert len(cycles) == 12
    assert run_record["request"]["camera_key"] == "camera"
    np.testing.assert_array_equal(
        np.asarray(cycles[0][1].image),
        np.asarray(first.capture.cycles[0][1].image),
    )

    # And calibrated again from that folder alone.  Same frames, same
    # settings, same answer -- to the pixel.
    def replay(**changes: object):
        return _task(
            replace(
                request,
                save_frames=False,
                frame_source=FRAMES_FROM_FOLDER,
                saved_frames_path=str(folder),
                **changes,
            ),
            tmp_path,
        ).run()

    replayed = replay()
    assert replayed.calibration.site_map.n_sites == first.calibration.site_map.n_sites
    np.testing.assert_allclose(
        replayed.calibration.site_map.centers_xy,
        first.calibration.site_map.centers_xy,
        atol=1e-9,
    )

    # The reason to keep them: detection can be retuned on the SAME atoms,
    # which is what nobody could do without holding the bench for another run.
    relaxed = replay(detection_sigma=4.0)
    assert relaxed.calibration.site_map.n_sites >= replayed.calibration.site_map.n_sites
    # The geometry comes from the frames, not from whatever the camera is set
    # to now.
    assert (
        replayed.calibration.frame_contract.image_shape
        == first.calibration.frame_contract.image_shape
    )
    assert (
        replayed.calibration.frame_contract.exposure_seconds
        == first.calibration.frame_contract.exposure_seconds
    )


def test_a_replay_publishes_what_the_node_declares(tmp_path: Path) -> None:
    """Run the folder through the runtime, not just through the function.

    A node that declares Dataset outputs must publish them however it got its
    frames.  Published only on the camera branch, a replay ran to the end and
    then died at the runtime -- "declared Dataset outputs but did not publish
    final outputs" -- which no test that called ``run()`` could see, because
    that failure lives in the host and ``run()`` has no host.
    """

    request = replace(_calibration_request(repeats=6), save_frames=True)
    acquired = _task(request, tmp_path).run()
    folder = acquired.artifact_path.with_suffix("") / "frames"

    plane = SignalDataPlane()
    host = None
    try:
        task = _task(
            replace(
                request,
                save_frames=False,
                frame_source=FRAMES_FROM_FOLDER,
                saved_frames_path=str(folder),
            ),
            tmp_path,
        )
        task.signal_plane = plane
        host = NodeHost(task, plane, Event().set)
        host.start()
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            host.poll()
            if host.observation.terminal:
                break
            time.sleep(0.01)
        observation = host.observation
        assert observation.phase == "done", (
            f"replay ended in {observation.phase}: {observation.error}"
        )
        key = host.signal_key(CAPTURE_PREVIEW_DECLARATION.name)
        publication = plane.latest_publication(key)
        assert publication is not None, "the frames read back were never published"
        value = publication.value(key)
        assert np.asarray(value.snapshot.block.values).shape[:2] == (1, 3)
    finally:
        if host is not None:
            host.shutdown()
        plane.close()


def test_nothing_is_written_unless_the_operator_asks(tmp_path: Path) -> None:
    result = _task(_calibration_request(repeats=8), tmp_path).run()
    assert not (result.artifact_path.with_suffix("") / "frames").exists()


def test_a_folder_with_no_samples_says_so(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    request = replace(
        _calibration_request(repeats=4),
        frame_source=FRAMES_FROM_FOLDER,
        saved_frames_path=str(empty),
    )
    with pytest.raises(ValueError, match="no saved calibration samples"):
        _task(request, tmp_path).run()


def test_calibrating_from_a_folder_needs_the_folder() -> None:
    with pytest.raises(ValueError, match="needs the folder"):
        replace(
            _calibration_request(),
            frame_source=FRAMES_FROM_FOLDER,
            saved_frames_path="",
        )
