"""
Fast mask processing using scipy.ndimage (replaces pipeline.py's pure-Python BFS).

Bottleneck analysis:
  clean_prediction_mask   : 3.2s/frame (54%) with Python BFS → ~0.05s with ndimage
  extract_external_boundary: 2.3s/frame (38%) with Python BFS → ~0.03s with ndimage
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ── Structuring elements ──────────────────────────────────────────────────────

_CROSS  = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)   # 4-connectivity
_SQUARE = np.ones((3, 3), dtype=bool)                         # 8-connectivity


# ── Internal helpers ─────────────────────────────────────────────────────────

def _keep_large(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Keep only foreground connected components with area ≥ min_area_px."""
    if min_area_px <= 0:
        return mask.astype(bool)
    labeled, n = ndimage.label(mask, structure=_CROSS)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep_labels = np.where(np.asarray(sizes) >= min_area_px)[0] + 1
    return np.isin(labeled, keep_labels)


# ── Drop-in replacements ──────────────────────────────────────────────────────

def fast_clean_prediction_mask(
    mask: np.ndarray,
    min_area_px: int,
    close_iterations: int,
) -> np.ndarray:
    """scipy.ndimage replacement for pipeline.clean_prediction_mask.

    Steps (identical semantics to the Python version):
    1. Remove connected components smaller than min_area_px
    2. Morphological closing with 3×3 kernel (close_iterations times)
    3. Remove small components again
    """
    # ── Step 1: remove small components ──────────────────────────────────────
    cleaned = _keep_large(mask, min_area_px)

    # ── Step 2: morphological closing (dilate N times, then erode N times) ───
    if close_iterations > 0:
        dilated = ndimage.binary_dilation(cleaned, structure=_SQUARE,
                                          iterations=close_iterations)
        closed  = ndimage.binary_erosion(dilated,  structure=_SQUARE,
                                          iterations=close_iterations,
                                          border_value=False)
    else:
        closed = cleaned

    # ── Step 3: remove small components again ────────────────────────────────
    return _keep_large(closed, min_area_px)


def fast_extract_external_boundary(mask: np.ndarray) -> np.ndarray:
    """scipy.ndimage replacement for pipeline.extract_external_boundary.

    A boundary pixel is a foreground pixel whose 8-neighbourhood contains
    at least one background pixel that is connected (4-connectivity) to the
    image border — i.e. it touches the "outside" background.
    """
    background = ~mask

    # Label 4-connected background regions
    labeled_bg, _ = ndimage.label(background, structure=_CROSS)

    # Which labels touch the image border?
    border_mask = np.zeros(mask.shape, dtype=bool)
    border_mask[0, :]  = True
    border_mask[-1, :] = True
    border_mask[:, 0]  = True
    border_mask[:, -1] = True

    border_bg_labels = set(labeled_bg[border_mask & background].tolist()) - {0}

    # outside_bg: background pixels reachable from image border
    outside_bg = np.isin(labeled_bg, list(border_bg_labels)) & background

    # Boundary = foreground pixels adjacent (8-connected) to outside_bg.
    # border_value=1 matches pipeline.py's np.pad(..., constant_values=True):
    # mask pixels at the image edge always touch the implicit "True" outside.
    dilated_outside = ndimage.binary_dilation(outside_bg, structure=_SQUARE, border_value=1)
    return (mask & dilated_outside).astype(bool)
