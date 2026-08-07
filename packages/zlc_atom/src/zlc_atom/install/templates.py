"""Named starting points for an apparatus.

A template is a convenience, not a constraint: devices are installed one by one,
so any mixture is legal by construction.  A real sequencer with a virtual camera
is not a special mode -- it is what you get by picking each independently, which
is exactly how hardware bring-up works.  You verify the pulse timing against a
camera you trust, then the imaging against a sequencer you trust, and only then
both at once.  Without that, a first light session can only be attempted all at
once, and a failure has nowhere to be bisected.

The two templates below are the ends of that spectrum; everything between them
is a list of DeviceSpec the caller writes, or a saved configuration reopened.
"""

from .graph import DeviceSpec

INSTALLATION_TEMPLATES = {
    "virtual": (
        DeviceSpec("camera", "camera.virtual"),
        DeviceSpec("sequencer", "sequencer.virtual", {"camera_key": "camera"}),
    ),
    "hardware": (
        DeviceSpec("camera", "camera.dcam"),
        DeviceSpec("sequencer", "sequencer.hardware"),
    ),
}

__all__ = ["INSTALLATION_TEMPLATES"]
