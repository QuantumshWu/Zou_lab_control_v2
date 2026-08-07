"""Pure figure-viewer shell; archive and renderer ownership stay outside."""

from .handle import FigureViewerHandle
from .view import FigureViewerView

__all__ = ["FigureViewerHandle",
    "FigureViewerView"]
