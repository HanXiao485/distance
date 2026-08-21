"""
Elevation map construction and ray-surface intersection.

The elevation map is a 2D grid in world XY plane storing median Z at each cell.
Empty cells are filled via nearest-neighbor interpolation (scipy).
Ray intersection uses coarse sampling + bisection refinement.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElevationMap:
    """BEV elevation map in world coordinates."""
    z_grid: np.ndarray      # (rows, cols) heights, no NaN (all filled)
    coverage_mask: np.ndarray  # (rows, cols) bool: True = cell had real LiDAR data
    x_min: float
    y_min: float
    cell_size: float
    rows: int
    cols: int
    filled_cells: int       # cells that had actual lidar points
    total_cells: int
    source_points: int      # number of lidar points used

    def z_at(self, x: float, y: float) -> Optional[float]:
        """Bilinear interpolation of Z at world (x, y). None if out of bounds."""
        ci = (x - self.x_min) / self.cell_size
        ri = (y - self.y_min) / self.cell_size
        if ci < 0 or ci >= self.cols - 1 or ri < 0 or ri >= self.rows - 1:
            return None
        c0, r0 = int(ci), int(ri)
        fc, fr = ci - c0, ri - r0
        z = (self.z_grid[r0,     c0]     * (1 - fc) * (1 - fr) +
             self.z_grid[r0,     c0 + 1] * fc        * (1 - fr) +
             self.z_grid[r0 + 1, c0]     * (1 - fc) * fr +
             self.z_grid[r0 + 1, c0 + 1] * fc        * fr)
        return float(z)

    def intersect_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        t_min: float = 0.5,
        t_max: float = 60.0,
        n_coarse: int = 300,
        n_bisect: int = 25,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Find first intersection of ray with elevation surface.

        Returns (3D world point, distance t) or None if no intersection found.
        The ray starts above the surface and we look for the first downward crossing.
        """
        d = direction / np.linalg.norm(direction)
        ts = np.linspace(t_min, t_max, n_coarse)

        prev_above: Optional[bool] = None
        prev_t = t_min

        for t in ts:
            p = origin + t * d
            z_surf = self.z_at(p[0], p[1])
            if z_surf is None:
                prev_above = None
                prev_t = t
                continue

            above = p[2] > z_surf

            if prev_above is not None and above != prev_above and not above:
                # Ray just crossed from above to below → bisect
                lo, hi = prev_t, float(t)
                for _ in range(n_bisect):
                    mid = (lo + hi) * 0.5
                    p_mid = origin + mid * d
                    z_mid = self.z_at(p_mid[0], p_mid[1])
                    if z_mid is None:
                        break
                    if p_mid[2] > z_mid:
                        lo = mid
                    else:
                        hi = mid
                t_final = (lo + hi) * 0.5
                return origin + t_final * d, t_final

            prev_above = above
            prev_t = float(t)

        return None

    def z_at_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized bilinear Z interpolation.  Out-of-bounds → nan."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ci = (x - self.x_min) / self.cell_size
        ri = (y - self.y_min) / self.cell_size
        valid = (ci >= 0) & (ci < self.cols - 1) & (ri >= 0) & (ri < self.rows - 1)
        result = np.full(x.shape, np.nan, dtype=np.float64)
        if valid.any():
            c0 = ci[valid].astype(np.int32)
            r0 = ri[valid].astype(np.int32)
            fc = ci[valid] - c0
            fr = ri[valid] - r0
            result[valid] = (
                self.z_grid[r0,     c0]     * (1 - fc) * (1 - fr) +
                self.z_grid[r0,     c0 + 1] * fc        * (1 - fr) +
                self.z_grid[r0 + 1, c0]     * (1 - fc) * fr +
                self.z_grid[r0 + 1, c0 + 1] * fc        * fr
            )
        return result

    def intersect_rays_batch(
        self,
        origins: np.ndarray,      # (N, 3)
        directions: np.ndarray,   # (N, 3)
        t_min: float = 0.5,
        t_max: float = 60.0,
        n_coarse: int = 300,
        n_bisect: int = 25,
    ) -> list:
        """Vectorized batch ray-surface intersection.

        The coarse-sampling phase runs as a single numpy operation over all rays.
        Bisection (small Python loop) only runs for rays that found a crossing.

        Returns list of (3D_point, t) or None for each input ray.
        """
        N = len(origins)
        if N == 0:
            return []

        # Normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        d = directions / np.maximum(norms, 1e-12)           # (N, 3)

        ts = np.linspace(t_min, t_max, n_coarse)            # (n_coarse,)

        # All sample points: (N, n_coarse, 3)
        pts = origins[:, None, :] + ts[None, :, None] * d[:, None, :]

        # Batch elevation query: flatten → (N*n_coarse,) → reshape (N, n_coarse)
        x_flat = pts[:, :, 0].ravel()
        y_flat = pts[:, :, 1].ravel()
        z_surf = self.z_at_batch(x_flat, y_flat).reshape(N, n_coarse)

        ray_z = pts[:, :, 2]                                 # (N, n_coarse)

        # above[i,j]: ray i at step j is above surface (nan → True = no crossing there)
        above = np.where(np.isnan(z_surf), True, ray_z > z_surf)

        # First downward crossing per ray: above[:,j] and not above[:,j+1]
        crossings = above[:, :-1] & ~above[:, 1:]           # (N, n_coarse-1)

        results: list = []
        for i in range(N):
            idx_arr = np.flatnonzero(crossings[i])
            if len(idx_arr) == 0:
                results.append(None)
                continue

            j = int(idx_arr[0])
            lo, hi = float(ts[j]), float(ts[j + 1])

            # Bisect for sub-cell precision (scalar, few iterations)
            for _ in range(n_bisect):
                mid = (lo + hi) * 0.5
                p_mid = origins[i] + mid * d[i]
                z_mid = self.z_at(float(p_mid[0]), float(p_mid[1]))
                if z_mid is None:
                    break
                if p_mid[2] > z_mid:
                    lo = mid
                else:
                    hi = mid

            t_final = (lo + hi) * 0.5
            results.append((origins[i] + t_final * d[i], t_final))

        return results

    def stats(self) -> dict:
        return {
            "source_points": self.source_points,
            "grid_size": [self.rows, self.cols],
            "cell_size_m": self.cell_size,
            "coverage_fraction": round(self.filled_cells / max(self.total_cells, 1), 4),
            "filled_cells": self.filled_cells,
            "total_cells": self.total_cells,
            "z_range_m": [round(float(self.z_grid.min()), 3), round(float(self.z_grid.max()), 3)],
        }


def build_elevation_map(
    points_world: np.ndarray,
    cell_size_m: float = 0.25,
    z_min_world: float = -np.inf,
    z_max_world: float = np.inf,
) -> ElevationMap:
    """Build BEV elevation map from world-frame point cloud.

    Args:
        points_world: (N, 3) world-frame XYZ points.
        cell_size_m: BEV grid resolution in metres.
        z_min_world: Discard points below this world Z (filter sky, underground).
        z_max_world: Discard points above this world Z.
    """
    mask = (points_world[:, 2] >= z_min_world) & (points_world[:, 2] <= z_max_world)
    pts = points_world[mask]

    if len(pts) < 4:
        raise ValueError(f"Only {len(pts)} points after Z filter [{z_min_world}, {z_max_world}].")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_min, y_min = float(x.min()), float(y.min())

    # Grid indices for each point
    xi = ((x - x_min) / cell_size_m).astype(int)
    yi = ((y - y_min) / cell_size_m).astype(int)
    cols = int(xi.max()) + 2
    rows = int(yi.max()) + 2

    xi = np.clip(xi, 0, cols - 1)
    yi = np.clip(yi, 0, rows - 1)

    # Accumulate Z per cell using a flat index
    flat_idx = yi * cols + xi
    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    z_sorted = z[order]

    # Find unique cells and compute median Z per cell
    unique_cells, cell_starts = np.unique(flat_sorted, return_index=True)
    cell_ends = np.r_[cell_starts[1:], len(z_sorted)]

    z_grid_flat = np.full(rows * cols, np.nan)
    for cell, start, end in zip(unique_cells, cell_starts, cell_ends):
        z_grid_flat[cell] = float(np.median(z_sorted[start:end]))

    z_grid = z_grid_flat.reshape(rows, cols)
    coverage_mask = ~np.isnan(z_grid)          # True = real LiDAR data
    filled_cells = int(coverage_mask.sum())

    # Fill empty cells via nearest-neighbour interpolation
    known_mask = coverage_mask
    if known_mask.any() and not known_mask.all():
        krow, kcol = np.where(known_mask)
        kz = z_grid[known_mask]
        try:
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(
                np.stack([kcol, krow], axis=1).astype(np.float32), kz.astype(np.float32)
            )
            all_row, all_col = np.mgrid[0:rows, 0:cols]
            filled = interp(all_col.ravel().astype(np.float32),
                            all_row.ravel().astype(np.float32))
            z_grid = filled.reshape(rows, cols).astype(np.float64)
        except ImportError:
            global_med = float(np.nanmedian(z_grid))
            z_grid = np.where(np.isnan(z_grid), global_med, z_grid)

    return ElevationMap(
        z_grid=z_grid,
        coverage_mask=coverage_mask,
        x_min=x_min,
        y_min=y_min,
        cell_size=float(cell_size_m),
        rows=rows,
        cols=cols,
        filled_cells=filled_cells,
        total_cells=rows * cols,
        source_points=int(len(pts)),
    )
