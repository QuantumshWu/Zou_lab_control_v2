"""Loopback acceptance for the remote camera: one server, real TCP, real bytes.

The server owns a virtual external-trigger camera whose frame source produces
a known per-ordinal pattern, so every pixel that crosses the wire can be
checked against what the sensor "saw" -- including that the native integer
dtype survives end to end, which is the repository's hard rule for frames.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading

import numpy as np
import pytest

from zlc_atom.devices.camera.contract import CameraAdapter
from zlc_atom.devices.camera.remote import CameraRemoteServer, RemoteCameraAdapter
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig


SENSOR_SHAPE_YX = (24, 32)
ROI_XYWH = (2, 3, 20, 16)


def _pattern(dtype: str, ordinal: int) -> np.ndarray:
    """A deterministic full-sensor frame that differs per trigger ordinal."""

    height, width = SENSOR_SHAPE_YX
    ramp = np.arange(height * width, dtype=np.uint32).reshape(SENSOR_SHAPE_YX)
    limit = int(np.iinfo(np.dtype(dtype)).max) + 1
    return ((ramp + ordinal * 7919) % limit).astype(np.dtype(dtype))


def _camera(dtype: str) -> VirtualCamera:
    return VirtualCamera(
        VirtualCameraConfig(
            frame_shape_yx=SENSOR_SHAPE_YX,
            exposure_seconds=0.02,
            frame_dtype=dtype,
        ),
        frame_source=lambda ordinal, exposure: _pattern(dtype, ordinal),
    )


@contextmanager
def _serving(camera: VirtualCamera):
    # Port 0 asks the kernel for a free port, so parallel test runs never race.
    server = CameraRemoteServer(("127.0.0.1", 0), camera)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


@pytest.mark.parametrize("dtype", ["<u2", "|u1"])
def test_remote_round_trip_preserves_roi_pixels_and_native_dtype(dtype: str) -> None:
    camera = _camera(dtype)
    with _serving(camera) as port:
        remote = RemoteCameraAdapter("127.0.0.1", port)
        remote.open()
        try:
            assert isinstance(remote, CameraAdapter)
            assert remote.timeout == camera.timeout

            point = remote.configure_measurement(
                exposure_seconds=0.01, roi_xywh=ROI_XYWH
            )
            assert point.roi_origin_yx == (ROI_XYWH[1], ROI_XYWH[0])
            assert point.frame_shape_yx == (ROI_XYWH[3], ROI_XYWH[2])
            assert point.dtype.str == dtype
            assert point.exposure_seconds == 0.01
            assert point.acquisition_mode == "EXTERNAL_TRIGGERED"

            remote.arm(3, source_group_sizes=(1, 1, 1), buffer_frame_count=3, timeout=2.0)
            assert remote.capture_state() is True
            # The trigger is the server machine's local fact -- exactly how a
            # sequencer edge reaches the real camera, bypassing the network.
            camera.trigger(3)
            records = remote.read_frame_records(3, timeout=5.0, exact=True)

            assert len(records) == 3
            x, y, width, height = ROI_XYWH
            for index, record in enumerate(records):
                expected = _pattern(dtype, index)[y : y + height, x : x + width]
                assert record.image.dtype.str == dtype
                assert np.array_equal(record.image, expected)
                assert record.source_ordinal == index
                assert record.produced_count == index + 1
                assert record.frame_stamp == index
                assert record.camera_stamp == index
                assert record.host_received_at_ns > 0

            terminal = remote.finish_record_capture()
            assert terminal.produced_count == 3
            assert terminal.source_stopped and terminal.no_more_frames and terminal.joined
            assert remote.capture_state() is False
            remote.close()
            assert camera.capture_state() is False
        finally:
            remote.disconnect()


def test_last_connector_takes_the_camera_and_the_displaced_client_hears_it() -> None:
    camera = _camera("<u2")
    with _serving(camera) as port:
        first = RemoteCameraAdapter("127.0.0.1", port)
        first.open()
        try:
            first.arm(2, source_group_sizes=(1, 1), buffer_frame_count=2, timeout=2.0)
            assert camera.capture_state() is True

            second = RemoteCameraAdapter("127.0.0.1", port)
            second.open()
            try:
                # The newcomer's handshake completed, so the takeover has run:
                # the orphaned arm was finished before the newcomer was served.
                assert camera.capture_state() is False
                assert second.capture_state() is False
                with pytest.raises(ConnectionError, match="camera server"):
                    first.capture_state()
                point = second.capture_working_point()
                assert point.sensor_shape_yx == SENSOR_SHAPE_YX
                second.close()
            finally:
                second.disconnect()
        finally:
            first.disconnect()
