"""Small, deterministic PSF helpers used by calibration and virtual imaging."""

from __future__ import annotations

import numpy as np


def normalized_psf_kernel(shape: tuple[int, int], sigma_xy: tuple[float, float] = (1.0, 1.0), *, angle_degrees: float = 0.0) -> np.ndarray:
    """Return a normalized, centered elliptical Gaussian kernel."""

    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("PSF kernel shape must contain positive odd dimensions")
    sx, sy = (float(sigma_xy[0]), float(sigma_xy[1]))
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("PSF sigmas must be positive")
    yy, xx = np.mgrid[:height, :width]
    xx = xx - (width - 1) / 2.0
    yy = yy - (height - 1) / 2.0
    angle = np.deg2rad(float(angle_degrees))
    xr = xx * np.cos(angle) + yy * np.sin(angle)
    yr = -xx * np.sin(angle) + yy * np.cos(angle)
    kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    total = float(kernel.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSF kernel normalization is not finite")
    kernel = np.asarray(kernel / total, dtype="<f8")
    kernel.setflags(write=False)
    return kernel


def gaussian_psf_kernel(sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    """Convenience wrapper for a square, circular Gaussian kernel."""

    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    size = 2 * radius + 1
    return normalized_psf_kernel((size, size), (float(sigma), float(sigma)))


def extract_psf_window(image: np.ndarray, box_xywh: tuple[int, int, int, int], kernel: np.ndarray, *, background: str = "none", padding: int = 2) -> float:
    """Apply one normalized kernel to a complete image window."""

    x, y, width, height = (int(value) for value in box_xywh)
    if width <= 0 or height <= 0 or kernel.shape != (height, width):
        raise ValueError("PSF box and kernel shapes must agree")
    if x < 0 or y < 0 or y + height > image.shape[0] or x + width > image.shape[1]:
        raise ValueError("PSF box lies outside image")
    cut = np.asarray(image[y : y + height, x : x + width], dtype=float)
    if background not in {"none", "annulus"}:
        raise ValueError("background must be 'none' or 'annulus'")
    offset = 0.0
    if background == "annulus":
        y0, y1 = max(0, y - int(padding)), min(image.shape[0], y + height + int(padding))
        x0, x1 = max(0, x - int(padding)), min(image.shape[1], x + width + int(padding))
        ring = np.asarray(image[y0:y1, x0:x1], dtype=float).copy()
        ring[y - y0 : y - y0 + height, x - x0 : x - x0 + width] = np.nan
        if np.isfinite(ring).any():
            offset = float(np.nanmedian(ring))
    return float(np.sum(np.asarray(kernel, dtype=float) * (cut - offset)))


__all__ = ["extract_psf_window", "gaussian_psf_kernel", "normalized_psf_kernel"]
