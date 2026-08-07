"""Camera adapter contracts and implementations."""

from .contract import (
    CameraAcquisitionMode,
    CameraAdapter,
    CameraCaptureSpec,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .dcam import DcamCameraAdapter, DcamCameraConfig
from .virtual import VirtualCamera, VirtualCameraConfig
from .world import SimulationGeometry, SimulationWorld

__all__ = [
    "CameraAcquisitionMode",
    "CameraAdapter",
    "CameraCaptureSpec",
    "CameraCaptureTerminalRecord",
    "CameraFrameRecord",
    "CameraWorkingPoint",
    "DcamCameraAdapter",
    "DcamCameraConfig",
    "VirtualCamera",
    "VirtualCameraConfig",
    "SimulationGeometry",
    "SimulationWorld",
]
