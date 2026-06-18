"""
Layer splitter service for SketchAI.

Takes a flat lineart PNG and produces 6 RGBA layer PNGs:
  layer_1_perspective_grid  – detected vanishing point + horizon grid
  layer_2_background        – top 30% of image
  layer_3_midground         – middle 40%
  layer_4_foreground        – bottom 30%
  layer_5_linework          – full Canny edge extraction
  layer_6_empty             – blank transparent canvas
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def split_layers(input_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Split a flat lineart PNG into 6 logical layer PNGs.

    Returns a dict mapping layer name → saved file path.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    layers = {
        "layer_1_perspective_grid": _make_perspective_grid(img_gray, w, h),
        "layer_2_background":       _make_zone(img_gray, w, h, 0.0,  0.30),
        "layer_3_midground":        _make_zone(img_gray, w, h, 0.25, 0.65),
        "layer_4_foreground":       _make_zone(img_gray, w, h, 0.60, 1.00),
        "layer_5_linework":         _make_linework(img_gray, w, h),
        "layer_6_empty":            _make_empty(w, h),
    }

    saved: dict[str, Path] = {}
    for name, rgba in layers.items():
        out_path = output_dir / f"{name}.png"
        Image.fromarray(rgba, "RGBA").save(out_path)
        saved[name] = out_path
        print(f"  saved {out_path.name}  ({rgba.shape[1]}×{rgba.shape[0]})")

    return saved


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def _make_empty(w: int, h: int) -> np.ndarray:
    """Fully transparent blank canvas."""
    return np.zeros((h, w, 4), dtype=np.uint8)


def _make_linework(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """Full Canny edge extraction – black edges on transparent background."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Where edges exist → opaque black pixel
    mask = edges > 0
    rgba[mask] = [0, 0, 0, 255]
    return rgba


def _make_zone(
    gray: np.ndarray,
    w: int,
    h: int,
    y_start_frac: float,
    y_end_frac: float,
) -> np.ndarray:
    """Extract a horizontal band of the image as an RGBA layer.

    Pixels are kept only inside the band; outside is transparent.
    A soft feather at the band edges avoids hard seams.
    """
    y0 = int(h * y_start_frac)
    y1 = int(h * y_end_frac)
    feather = max(1, int(h * 0.04))  # 4% of height

    # Build alpha mask: full inside, linear ramp at edges
    alpha_1d = np.zeros(h, dtype=np.float32)
    alpha_1d[y0:y1] = 1.0

    # Top feather
    top_end = min(y0 + feather, y1)
    alpha_1d[y0:top_end] = np.linspace(0, 1, top_end - y0)

    # Bottom feather
    bot_start = max(y1 - feather, y0)
    alpha_1d[bot_start:y1] = np.linspace(1, 0, y1 - bot_start)

    alpha_2d = np.tile(alpha_1d[:, None], (1, w))  # (h, w) float

    # Invert gray so dark lines become visible content, multiply by alpha
    content = (255 - gray).astype(np.float32)
    content_alpha = (content / 255.0 * alpha_2d * 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Black lines, alpha from band mask
    rgba[..., 3] = content_alpha
    return rgba


def _make_perspective_grid(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """Detect a vanishing point via Hough lines and draw a perspective grid."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=60)
    vp = _estimate_vanishing_point(lines, w, h)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    _draw_grid(rgba, vp, w, h)
    return rgba


def _estimate_vanishing_point(
    lines: np.ndarray | None,
    w: int,
    h: int,
) -> tuple[float, float]:
    """Return (x, y) of best-guess vanishing point.

    Uses the intersection of prominent Hough lines. Falls back to image centre
    if no reliable intersection is found.
    """
    default_vp = (w / 2.0, h * 0.35)

    if lines is None or len(lines) < 2:
        return default_vp

    # Convert (rho, theta) lines to (a, b, c) form: ax + by = c
    line_eqs: list[tuple[float, float, float]] = []
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        c = rho
        # Skip near-horizontal lines (horizon) – they don't converge to a VP
        if abs(b) > 0.97:
            continue
        line_eqs.append((a, b, c))

    if len(line_eqs) < 2:
        return default_vp

    # Collect pairwise intersections
    xs, ys = [], []
    for i in range(len(line_eqs)):
        for j in range(i + 1, len(line_eqs)):
            a1, b1, c1 = line_eqs[i]
            a2, b2, c2 = line_eqs[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            # Only accept intersections in a generous region around the image
            if -w < x < 2 * w and -h < y < 2 * h:
                xs.append(x)
                ys.append(y)

    if not xs:
        return default_vp

    # Robust median to resist outliers
    vx = float(np.median(xs))
    vy = float(np.median(ys))

    # Clamp to a sensible range (VP can be outside canvas but not wildly so)
    vx = np.clip(vx, -w * 0.5, w * 1.5)
    vy = np.clip(vy, -h * 0.5, h * 1.5)
    return (vx, vy)


def _draw_grid(
    rgba: np.ndarray,
    vp: tuple[float, float],
    w: int,
    h: int,
    n_rays: int = 12,
    n_horizontals: int = 6,
) -> None:
    """Draw vanishing-point rays and horizontal recession lines onto rgba."""
    vx, vy = vp
    color = (80, 80, 200, 180)  # blue-ish, semi-transparent

    # Radial rays from vanishing point to image edges
    for i in range(n_rays):
        angle = np.pi * i / n_rays
        dx = np.cos(angle)
        dy = np.sin(angle)
        # Extend far enough to always reach the image border
        length = (w + h) * 2.0
        x2 = int(vx + dx * length)
        y2 = int(vy + dy * length)
        x3 = int(vx - dx * length)
        y3 = int(vy - dy * length)
        cv2.line(rgba, (int(vx), int(vy)), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.line(rgba, (int(vx), int(vy)), (x3, y3), color, 1, cv2.LINE_AA)

    # Horizontal recession lines spaced from VP downward
    horizon_y = int(vy)
    for k in range(1, n_horizontals + 1):
        y_line = int(horizon_y + (h - horizon_y) * k / (n_horizontals + 1))
        if 0 <= y_line < h:
            cv2.line(rgba, (0, y_line), (w, y_line), color, 1, cv2.LINE_AA)

    # Mark the vanishing point itself
    vp_int = (int(np.clip(vx, 0, w - 1)), int(np.clip(vy, 0, h - 1)))
    cv2.circle(rgba, vp_int, 5, (200, 80, 80, 220), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_img = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/sketchai/test-outputs/forest_path_v1.png")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.expanduser("~/sketchai/test-outputs/layers")

    print(f"Input : {input_img}")
    print(f"Output: {out_dir}\n")

    results = split_layers(input_img, out_dir)

    print(f"\nDone — {len(results)} layers saved to {out_dir}")
