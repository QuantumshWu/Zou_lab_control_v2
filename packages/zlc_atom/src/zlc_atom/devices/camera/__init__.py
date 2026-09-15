"""The camera adapter contract; each camera is a folder beside this one."""

from .contract import (
    CAMERA_PROTECTED_FIELDS,
    CameraAcquisitionMode,
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)

__all__ = [
    "CAMERA_PROTECTED_FIELDS",
    "CameraAcquisitionMode",
    "CameraAdapter",
    "CameraCaptureTerminalRecord",
    "CameraFrameRecord",
    "CameraWorkingPoint",
]
