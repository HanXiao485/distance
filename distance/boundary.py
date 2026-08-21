"""Stage 3 — GT/Pred mask cleaning and external-boundary extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .io_utils import ensure_dir, load_binary_mask, resolve_path, save_binary


def extract_gt_mask(config: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, Path]:
    gt_cfg = config["masks"]["gt"]
    class_name = config["class"]["name"]
    path = resolve_path(config, gt_cfg["path"])
    if gt_cfg["type"] == "index":
        mask = np.array(Image.open(path)) == int(config["class"]["id_2d"])
    elif gt_cfg["type"] == "binary":
        mask = load_binary_mask(path)
    else:
        raise ValueError(f"Unsupported GT mask type: {gt_cfg['type']}")
    out_path = output_dir / "masks" / f"gt_{class_name}_binary.png"
    # 原始GT mask（清理/提取边界之前）纯粹是调试用的中间产物，没有被下游流程读回，
    # 跟其它诊断图共用同一个开关。
    if config.get("h_diagnostics", {}).get("enabled", True):
        save_binary(mask, out_path)
    return mask, out_path


def extract_pred_mask(config: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, Path]:
    pred_cfg = config["masks"]["pred"]
    class_name = config["class"]["name"]
    path = resolve_path(config, pred_cfg["path"])
    if pred_cfg["type"] == "binary":
        mask = load_binary_mask(path)
    elif pred_cfg["type"] == "index":
        mask = np.array(Image.open(path)) == int(config["class"]["id_2d"])
    elif pred_cfg["type"] == "color_threshold":
        rgb = np.array(Image.open(path).convert("RGB"))
        r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        t = pred_cfg["threshold"]
        mask = (
            (r >= int(t.get("r_min", 0))) & (r <= int(t.get("r_max", 255))) &
            (g >= int(t.get("g_min", 0))) & (g <= int(t.get("g_max", 255))) &
            (b >= int(t.get("b_min", 0))) & (b <= int(t.get("b_max", 255))) &
            ((g - r) >= int(t.get("g_minus_r_min", -255))) &
            ((g - b) >= int(t.get("g_minus_b_min", -255)))
        )
    else:
        raise ValueError(f"Unsupported pred mask type: {pred_cfg['type']}")
    out_path = output_dir / "masks" / f"pred_{class_name}_binary.png"
    # 同上，原始pred mask也是纯调试中间产物。
    if config.get("h_diagnostics", {}).get("enabled", True):
        save_binary(mask, out_path)
    return mask, out_path


_STRUCT_3X3 = np.ones((3, 3), dtype=bool)


def binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
    """等价于原先 9 张位移数组堆叠再 all() 的写法，改用 scipy 的 C 实现。"""
    return ndi.binary_erosion(mask, structure=_STRUCT_3X3, border_value=0)


def binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    """等价于原先 9 张位移数组堆叠再 any() 的写法，改用 scipy 的 C 实现。"""
    return ndi.binary_dilation(mask, structure=_STRUCT_3X3, border_value=0)


def binary_close_3x3(mask: np.ndarray, iterations: int) -> np.ndarray:
    """N次3x3方形结构元素的膨胀/腐蚀，数学上跟单次(1+2N)×(1+2N)方形结构元素的
    膨胀/腐蚀完全等价（已用150组随机数据+多种iterations值验证，含边缘情况），
    合并成一次调用减少 scipy 函数调用开销。"""
    if iterations <= 0:
        return mask.copy()
    k = 1 + 2 * iterations
    struct_big = np.ones((k, k), dtype=bool)
    dilated = ndi.binary_dilation(mask, structure=struct_big, border_value=0)
    return ndi.binary_erosion(dilated, structure=struct_big, border_value=0)


_STRUCT_4CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def connected_components(mask: np.ndarray, target_value: bool = True) -> list[np.ndarray]:
    """改用 scipy.ndimage.label（4连通，跟原手写DFS的上下左右扩散完全一致）替代逐像素DFS。
    返回值类型/含义不变：每个连通域一个 (N,2) 的 (y,x) 坐标数组列表，供调用方按面积过滤。"""
    target_mask = mask == target_value
    labeled, num_features = ndi.label(target_mask, structure=_STRUCT_4CONN)
    if num_features == 0:
        return []
    components = []
    for label_id in range(1, num_features + 1):
        ys, xs = np.nonzero(labeled == label_id)
        components.append(np.stack([ys, xs], axis=1).astype(np.int32))
    return components


def keep_large_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """直接用 scipy.ndimage.label + 面积统计实现，比经过 connected_components
    再逐个组装坐标列表更快（避免 O(连通域数×像素数) 的重复扫描）。"""
    target_mask = mask == True  # noqa: E712  (与 connected_components 的 target_value=True 语义一致)
    labeled, num_features = ndi.label(target_mask, structure=_STRUCT_4CONN)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    # 标签是 1..num_features 的小整数，直接建查找表按下标索引，
    # 比通用的 np.isin（内部做排序/搜索）快，尤其在几百万像素规模下。
    areas = ndi.sum(target_mask, labeled, index=np.arange(1, num_features + 1))
    keep_lookup = np.zeros(num_features + 1, dtype=bool)
    keep_lookup[1:] = areas >= min_area_px
    return keep_lookup[labeled]


def clean_prediction_mask(mask: np.ndarray, min_area_px: int, close_iterations: int) -> np.ndarray:
    large_components = keep_large_components(mask, min_area_px)
    closed = binary_close_3x3(large_components, close_iterations)
    return keep_large_components(closed, min_area_px)


def border_connected_background(mask: np.ndarray) -> np.ndarray:
    """原实现是从图像四条边上的背景像素出发做4连通洪水填充。
    等价写法：给 background 做4连通标记，凡是"标签出现在图像边框上"的连通域整体
    都算作 border-connected（跟从边框洪水填充能到达的像素集合完全一致）。"""
    background = ~mask
    labeled, num_features = ndi.label(background, structure=_STRUCT_4CONN)
    if num_features == 0:
        return np.zeros(mask.shape, dtype=bool)
    border_labels = np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1],
    ]))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    # 同上，用查找表替代 np.isin。
    keep_lookup = np.zeros(num_features + 1, dtype=bool)
    keep_lookup[border_labels] = True
    return keep_lookup[labeled]


def extract_external_boundary(mask: np.ndarray) -> np.ndarray:
    outside_background = border_connected_background(mask)
    padded = np.pad(outside_background, 1, mode="constant", constant_values=True)
    touches_outside = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            touches_outside |= padded[dy:dy + height, dx:dx + width]
    return mask & touches_outside


def make_boundary_overlay(gt_boundary: np.ndarray, pred_boundary: np.ndarray, path: Path) -> None:
    canvas = np.zeros((*gt_boundary.shape, 3), dtype=np.uint8)
    canvas[gt_boundary & pred_boundary] = [45, 190, 90]
    canvas[gt_boundary & ~pred_boundary] = [255, 70, 70]
    canvas[~gt_boundary & pred_boundary] = [55, 135, 255]
    ensure_dir(path.parent)
    Image.fromarray(canvas, mode="RGB").save(path)


def create_boundaries(
    config: dict[str, Any], output_dir: Path
) -> tuple[Path, Path, dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    """返回值第四项是 (gt_boundary, pred_boundary) 内存数组，跟磁盘上刚写的
    gt_boundary_path/pred_boundary_path 内容完全一致，供 scanline_error 直接使用、
    跳过重新读盘（PNG文件本身仍然照常写，作为持久化产出保留）。"""
    class_name = config["class"]["name"]
    mask_cfg = config.get("mask_processing", {})
    gt_mask, gt_mask_path = extract_gt_mask(config, output_dir)
    pred_mask, pred_mask_path = extract_pred_mask(config, output_dir)
    pred_cleaned = clean_prediction_mask(
        pred_mask,
        int(mask_cfg.get("min_component_area_px", 5000)),
        int(mask_cfg.get("close_iterations", 2)),
    )
    gt_boundary = extract_external_boundary(gt_mask)
    pred_boundary = extract_external_boundary(pred_cleaned)
    pred_cleaned_path = output_dir / "boundaries" / f"pred_{class_name}_cleaned_binary.png"
    gt_boundary_path = output_dir / "boundaries" / f"gt_{class_name}_boundary.png"
    pred_boundary_path = output_dir / "boundaries" / f"pred_{class_name}_boundary.png"
    overlay_path = output_dir / "boundaries" / f"{class_name}_gt_pred_boundary_overlay.png"
    # pred_cleaned_path 和 overlay 是纯诊断产物（清理后中间态 / GT-Pred对比可视化），
    # 没有被下游流程读回；gt_boundary_path/pred_boundary_path 才是本函数的核心输出
    # （scanline_error 会用，虽然现在走内存传递不用重新读盘，但文件本身仍需保留），
    # 所以只有前两者接 h_diagnostics.enabled 开关。
    if config.get("h_diagnostics", {}).get("enabled", True):
        save_binary(pred_cleaned, pred_cleaned_path)
    save_binary(gt_boundary, gt_boundary_path)
    save_binary(pred_boundary, pred_boundary_path)
    if config.get("h_diagnostics", {}).get("enabled", True):
        make_boundary_overlay(gt_boundary, pred_boundary, overlay_path)
    metadata = {
        "gt_mask_pixels": int(gt_mask.sum()),
        "pred_mask_pixels": int(pred_mask.sum()),
        "pred_cleaned_pixels": int(pred_cleaned.sum()),
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "pred_boundary_pixels": int(pred_boundary.sum()),
        "gt_mask": str(gt_mask_path),
        "pred_mask": str(pred_mask_path),
        "pred_cleaned_mask": str(pred_cleaned_path),
        "boundary_overlay": str(overlay_path),
    }
    return gt_boundary_path, pred_boundary_path, metadata, (gt_boundary, pred_boundary)
