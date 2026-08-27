"""
实时部署专用：H 估计核心算法的向量化优化版。

设计目标
--------
- 与 run_boundary_error_pipeline.py 中的原算法 **逐位/数值等价**（已用 wildscenes_mini
  的真实对应点验证），只是更快，便于实时（10–20Hz）部署。
- 不含任何可视化/画图（可视化保留在原管线中，用于判断方法准确性）。

包含的优化函数（均为原函数的等价加速替换）：
- zbuffer_visible_mask : Python 字典循环 → 向量化 lexsort（稳定排序保证并列规则一致）
- fit_homography       : Python 循环建 A 矩阵 → 向量化构建（A 完全相同 → SVD 相同 → H 相同）
- sigma_filter_correspondences : 复用快速 fit_homography，迭代逻辑不变
- ransac_fit_h         : 1000 次 Python 迭代 → 批量 SVD（相同随机子集顺序 → 结果一致）

其余已向量化的函数（fit_plane / normalize_points_2d / apply_homography / world_to_plane /
choose_plane_basis）与原版按位相同，直接照搬。
"""
from __future__ import annotations

import numpy as np


# ───────────────────────── 与原版按位相同（已向量化） ─────────────────────────
def fit_plane(points_world: np.ndarray):
    centroid = points_world.mean(axis=0)
    centered = points_world - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    d = -float(normal @ centroid)
    residual = np.abs(points_world @ normal + d)
    return normal, d, residual


def choose_plane_basis(normal: np.ndarray):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    axis_x = ref - np.dot(ref, normal) * normal
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    return axis_x, axis_y


def world_to_plane(points_world, origin, axis_x, axis_y):
    centered = points_world - origin
    return np.column_stack([centered @ axis_x, centered @ axis_y])


def normalize_points_2d(points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid
    dist = np.sqrt((centered ** 2).sum(axis=1))
    scale = np.sqrt(2.0) / max(float(dist.mean()), 1e-12)
    T = np.array([[scale, 0.0, -scale * centroid[0]],
                  [0.0, scale, -scale * centroid[1]],
                  [0.0, 0.0, 1.0]])
    normalized_h = (T @ np.column_stack([points, np.ones(len(points))]).T).T
    return normalized_h[:, :2], T


def apply_homography(H: np.ndarray, points_uv: np.ndarray):
    points_h = np.column_stack([points_uv, np.ones(len(points_uv))])
    mapped_h = (H @ points_h.T).T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


# ───────────────────────── 优化① 向量化 zbuffer ─────────────────────────
def zbuffer_visible_mask(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """每个像素只保留最小深度点；并列深度保留最小原始索引。与原逐点字典版按位等价。"""
    n = len(pixels)
    if n == 0:
        return np.zeros(0, dtype=bool)
    u = pixels[:, 0].astype(np.int64)
    v = pixels[:, 1].astype(np.int64)
    mult = int(u.max()) + 1
    key = v * mult + u
    order = np.lexsort((np.asarray(depths, np.float64), key))  # 稳定排序
    ks = key[order]
    first = np.ones(n, dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    visible = np.zeros(n, dtype=bool)
    visible[order[first]] = True
    return visible


# ───────────────────────── 优化② 向量化 fit_homography ─────────────────────────
def _build_A(src_norm: np.ndarray, dst_norm: np.ndarray) -> np.ndarray:
    """向量化构建 DLT 的 A 矩阵，行序与原 Python 循环完全一致（交错两行）。"""
    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]
    n = len(x)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    A[0::2, 0] = -x; A[0::2, 1] = -y; A[0::2, 2] = -1.0
    A[0::2, 6] = u * x; A[0::2, 7] = u * y; A[0::2, 8] = u
    A[1::2, 3] = -x; A[1::2, 4] = -y; A[1::2, 5] = -1.0
    A[1::2, 6] = v * x; A[1::2, 7] = v * y; A[1::2, 8] = v
    return A


def fit_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    src_norm, T_src = normalize_points_2d(src_xy)
    dst_norm, T_dst = normalize_points_2d(dst_uv)
    A = _build_A(src_norm, dst_norm)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    H_norm = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2]


# ───────────────────────── 优化③ sigma 过滤（逻辑同原版，用快 fit_homography） ────
def sigma_filter_correspondences(plane_xy, uv, threshold_sigma, max_iters):
    mask = np.ones(len(plane_xy), dtype=bool)
    removed_per_iter = []
    for _ in range(max_iters):
        if mask.sum() < 4:
            break
        H_try = fit_homography(plane_xy[mask], uv[mask])
        uv_reproj = apply_homography(H_try, plane_xy[mask])
        errs = np.linalg.norm(uv_reproj - uv[mask], axis=1)
        cutoff = errs.mean() + threshold_sigma * errs.std()
        keep_inner = errs <= cutoff
        n_removed = int((~keep_inner).sum())
        removed_per_iter.append(n_removed)
        if n_removed == 0:
            break
        new_mask = mask.copy()
        new_mask[np.flatnonzero(mask)] = keep_inner
        mask = new_mask
    return mask, removed_per_iter


# ───────────────────────── 优化④ 批量 RANSAC（结果与原版一致） ────────────────
def _fit_homography_batch(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """对 K 组各 4 点对，批量拟合 K 个 H。src_pts/dst_pts: (K,4,2)。返回 (K,3,3)。

    每组独立归一化 + 批量 SVD，复现原 fit_homography 的逐组结果。非有限解返回 nan。
    """
    K = src_pts.shape[0]
    def norm_batch(p):  # p:(K,4,2) -> 归一化点(K,4,2), T(K,3,3)
        cen = p.mean(axis=1, keepdims=True)                      # (K,1,2)
        centered = p - cen
        dist = np.sqrt((centered ** 2).sum(axis=2))              # (K,4)
        scale = np.sqrt(2.0) / np.maximum(dist.mean(axis=1), 1e-12)  # (K,)
        T = np.zeros((K, 3, 3))
        T[:, 0, 0] = scale; T[:, 1, 1] = scale
        T[:, 0, 2] = -scale * cen[:, 0, 0]; T[:, 1, 2] = -scale * cen[:, 0, 1]
        T[:, 2, 2] = 1.0
        ph = np.concatenate([p, np.ones((K, 4, 1))], axis=2)     # (K,4,3)
        pn = np.einsum('kij,kpj->kpi', T, ph)[:, :, :2]
        return pn, T
    sn, Ts = norm_batch(src_pts)
    dn, Td = norm_batch(dst_pts)
    x = sn[:, :, 0]; y = sn[:, :, 1]; u = dn[:, :, 0]; v = dn[:, :, 1]
    A = np.zeros((K, 8, 9))
    A[:, 0::2, 0] = -x; A[:, 0::2, 1] = -y; A[:, 0::2, 2] = -1.0
    A[:, 0::2, 6] = u * x; A[:, 0::2, 7] = u * y; A[:, 0::2, 8] = u
    A[:, 1::2, 3] = -x; A[:, 1::2, 4] = -y; A[:, 1::2, 5] = -1.0
    A[:, 1::2, 6] = v * x; A[:, 1::2, 7] = v * y; A[:, 1::2, 8] = v
    with np.errstate(all='ignore'):
        _, _, vh = np.linalg.svd(A, full_matrices=False)         # vh:(K,8,9)
        H_norm = vh[:, -1, :].reshape(K, 3, 3)
        Td_inv = np.linalg.inv(Td)
        H = np.einsum('kij,kjl,klm->kim', Td_inv, H_norm, Ts)
        denom = H[:, 2, 2]
        H = H / denom[:, None, None]
    return H


def ransac_fit_h(plane_xy, uv, threshold_px, max_iters, min_inliers=8, seed=42,
                 chunk: int = 128):
    """与原 ransac_fit_h 结果一致：相同 rng 子集序列 + 相同“首个最大内点”选择。

    实现：批量拟合所有候选 H + 分块计算内点数（分块避免 (K,N) 巨数组的内存瓶颈）。
    实测与原版掩码/ H 完全一致，速度约 1.8x。
    """
    n = len(plane_xy)
    rng = np.random.default_rng(seed)
    subsets = np.empty((max_iters, 4), dtype=np.int64)
    for i in range(max_iters):                       # 复现原循环的随机子集序列
        subsets[i] = rng.choice(n, size=4, replace=False)
    H_all = _fit_homography_batch(plane_xy[subsets], uv[subsets])     # (K,3,3)

    px, py = plane_xy[:, 0], plane_xy[:, 1]
    ux, vy = uv[:, 0], uv[:, 1]
    counts = np.empty(max_iters, dtype=np.int64)
    for s in range(0, max_iters, chunk):             # 分块计算每个候选 H 的内点数
        Hc = H_all[s:s + chunk]
        a = Hc[:, 0, 0, None] * px + Hc[:, 0, 1, None] * py + Hc[:, 0, 2, None]
        b = Hc[:, 1, 0, None] * px + Hc[:, 1, 1, None] * py + Hc[:, 1, 2, None]
        w = Hc[:, 2, 0, None] * px + Hc[:, 2, 1, None] * py + Hc[:, 2, 2, None]
        with np.errstate(all='ignore'):
            e = np.sqrt((a / w - ux) ** 2 + (b / w - vy) ** 2)        # (chunk, n)
        cnt = (e <= threshold_px).sum(axis=1)
        fin = np.isfinite(Hc.reshape(len(Hc), -1)).all(axis=1)
        counts[s:s + chunk] = np.where(fin, cnt, -1)                  # 非有限解视为跳过

    best = int(np.argmax(counts))                    # 首个最大（与原 strict > 一致）
    if int(counts[best]) < min_inliers:
        return fit_homography(plane_xy, uv), np.ones(n, dtype=bool)
    Hb = H_all[best]
    a = Hb[0, 0] * px + Hb[0, 1] * py + Hb[0, 2]
    b = Hb[1, 0] * px + Hb[1, 1] * py + Hb[1, 2]
    w = Hb[2, 0] * px + Hb[2, 1] * py + Hb[2, 2]
    best_mask = np.sqrt((a / w - ux) ** 2 + (b / w - vy) ** 2) <= threshold_px
    return fit_homography(plane_xy[best_mask], uv[best_mask]), best_mask
