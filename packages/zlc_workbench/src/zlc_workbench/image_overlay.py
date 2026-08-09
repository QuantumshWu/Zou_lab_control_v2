"""Join one image and one explicit overlay sibling from the same publication."""

from __future__ import annotations

from zlc_plot import ImageFrame, ImagePointOverlay


def image_frame_from_publication(
    value: object,
    publication: object,
    *,
    overlay_signal: str,
    overlay_revision: int,
) -> ImageFrame:
    snapshot = getattr(value, "snapshot", None)
    if snapshot is None:
        raise TypeError("image signal must carry an OwnedSnapshot")
    exact = getattr(publication, "value", None)
    if not callable(exact) or exact(getattr(value, "name", "")) is not value:
        raise ValueError("image and overlay must belong to one exact publication")
    overlay_value = exact(str(overlay_signal))
    if overlay_value is None:
        raise ValueError("the selected overlay is not a sibling of this image")
    overlay = ImagePointOverlay.from_snapshot(
        getattr(overlay_value, "snapshot", None),
        revision=overlay_revision,
    )
    return ImageFrame(snapshot, overlay)


__all__ = ["image_frame_from_publication"]
