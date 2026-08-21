"""用 wildscenes_mini 的真实对应点验证 realtime_opt.py 与原算法结果一致 + 测速。"""
from __future__ import annotations
import csv, glob, time
from pathlib import Path
import numpy as np

import run_boundary_error_pipeline as orig   # 原版（含可视化的管线，导入其算法函数）
import realtime_opt as opt                     # 优化版

ROOT = Path(__file__).resolve().parents[2]


def load_csv(path):
    uv, world = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            uv.append([float(row["u"]), float(row["v"])])
            world.append([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])])
    return np.asarray(uv), np.asarray(world)


def prep(world, uv):
    """复现 estimate_h 的前处理：平面拟合 → 0.5m 内点 → 投到平面 2D。"""
    normal, d, res = orig.fit_plane(world)
    inl = res <= 0.5
    wi, ui = world[inl], uv[inl]
    cen = wi.mean(0)
    ax, ay = orig.choose_plane_basis(normal)
    return orig.world_to_plane(wi, cen, ax, ay), ui


def main():
    csvs = sorted(glob.glob(str(ROOT / "outputs/**/dirt_pixel_correspondences.csv"), recursive=True))
    csvs = csvs[:30]   # 取 30 帧验证
    print(f"验证 {len(csvs)} 帧\n")

    max_H_fit = max_H_sig = max_H_ran = 0.0
    mask_sig_ok = mask_ran_ok = True
    t_orig = {"fit": 0.0, "sig": 0.0, "ran": 0.0}
    t_opt = {"fit": 0.0, "sig": 0.0, "ran": 0.0}

    for c in csvs:
        uv, world = load_csv(c)
        pxy, ui = prep(world, uv)
        if len(pxy) < 10:
            continue

        # ① fit_homography
        t = time.perf_counter(); Ho = orig.fit_homography(pxy, ui); t_orig["fit"] += time.perf_counter() - t
        t = time.perf_counter(); Hn = opt.fit_homography(pxy, ui);  t_opt["fit"] += time.perf_counter() - t
        max_H_fit = max(max_H_fit, np.abs(Ho - Hn).max())

        # ② sigma_filter（默认 5 迭代, thr 2.5）
        t = time.perf_counter(); mo, _ = orig.sigma_filter_correspondences(pxy, ui, 2.5, 5); t_orig["sig"] += time.perf_counter() - t
        t = time.perf_counter(); mn, _ = opt.sigma_filter_correspondences(pxy, ui, 2.5, 5); t_opt["sig"] += time.perf_counter() - t
        mask_sig_ok &= np.array_equal(mo, mn)
        if mo.sum() >= 4 and mn.sum() >= 4:
            max_H_sig = max(max_H_sig, np.abs(orig.fit_homography(pxy[mo], ui[mo]) -
                                              opt.fit_homography(pxy[mn], ui[mn])).max())

        # ③ ransac（默认 1000 迭代, thr 5.0, seed 42）
        t = time.perf_counter(); Hro, mro = orig.ransac_fit_h(pxy, ui, 5.0, 1000); t_orig["ran"] += time.perf_counter() - t
        t = time.perf_counter(); Hrn, mrn = opt.ransac_fit_h(pxy, ui, 5.0, 1000);  t_opt["ran"] += time.perf_counter() - t
        mask_ran_ok &= np.array_equal(mro, mrn)
        max_H_ran = max(max_H_ran, np.abs(Hro - Hrn).max())

    n = len(csvs)
    print("==================== 一致性（越接近 0 越好）====================")
    print(f"fit_homography  H 最大逐元素差 : {max_H_fit:.3e}")
    print(f"sigma_filter    内点掩码完全一致: {mask_sig_ok}   H 最大差: {max_H_sig:.3e}")
    print(f"ransac          内点掩码完全一致: {mask_ran_ok}   H 最大差: {max_H_ran:.3e}")
    print("\n==================== 速度（{} 帧累计）====================".format(n))
    for k, label in [("fit", "fit_homography"), ("sig", "sigma_filter"), ("ran", "ransac")]:
        so, sn = t_orig[k] * 1000, t_opt[k] * 1000
        print(f"{label:16}: 原版 {so/n:7.2f} ms/帧  →  优化 {sn/n:7.2f} ms/帧   提速 {so/max(sn,1e-9):5.1f}x")
    tot_o = sum(t_orig.values()) * 1000 / n
    tot_n = sum(t_opt.values()) * 1000 / n
    print(f"{'合计(三项)':16}: 原版 {tot_o:7.2f} ms/帧  →  优化 {tot_n:7.2f} ms/帧   提速 {tot_o/max(tot_n,1e-9):5.1f}x")


if __name__ == "__main__":
    main()
