"""
GOOSE 可通行区域「覆盖精确率」评测（在 val 集上，用已训练的 64 类模型）。

指标:  Precision = |Pred_可通行 ∩ GT_可通行| / |Pred_可通行|
       预测区域完全落在真值内（哪怕偏小）即 100%；只惩罚误报，不惩罚漏报。

用法（在容器 /root/distance/WildScenes 下，已建好 val 软链接）:
    cd /root/distance/WildScenes
    export CUDA_VISIBLE_DEVICES=0
    python eval_traversable.py

若 val 软链接还没建，先跑一次:
    python scripts/data/setup_goose.py \
        --goose_root data/Goose/Goose/Goose \
        --save_path  data/processed/goose2d
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from mmseg.apis import init_model, inference_model

# ── 路径（绝对路径，和 cwd 无关）────────────────────────────────────────────
ROOT    = "/root/distance/WildScenes"
CONFIG  = ROOT + "/wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose-512x512.py"
CKPT    = ROOT + "/best_mIoU_iter_72000.pth"
IMG_DIR = ROOT + "/data/processed/goose2d/val/image"
LBL_DIR = ROOT + "/data/processed/goose2d/val/indexLabel"

# ── GOOSE 64 类 id → 类名（用于逐类报告）────────────────────────────────────
CLASS_NAME = {
    0: "undefined", 2: "snow", 3: "cobble", 5: "leaves", 7: "bikeway",
    9: "pedestrian_crossing", 11: "road_marking", 18: "moss", 21: "sidewalk",
    23: "asphalt", 24: "gravel", 31: "soil", 50: "low_grass", 51: "high_grass",
    56: "outlier",
}

# ── 可通行类（GOOSE 64 类 id），按需增删 ────────────────────────────────────
#   23 asphalt  24 gravel  31 soil  3 cobble  50 low_grass
#   7 bikeway   21 sidewalk 9 pedestrian_crossing  11 road_marking
#   可选追加: 51 high_grass, 18 moss, 5 leaves, 2 snow
TRAVERSABLE = {23, 24, 31, 3, 50, 7, 21, 9, 11}

# ── 评测时忽略的真值类（不计入分子分母）────────────────────────────────────
IGNORE = {0, 56}          # 0 undefined, 56 outlier

# ── 不确定性过滤：>0 时，softmax 置信度低于该阈值的像素判为「不可通行」───────
#   softmax 置信度是概率，范围 0~1。想冲高 precision 可设 0.6 ~ 0.8（会牺牲 recall）
#   填 0 = 关闭过滤（基线）。切勿填 >=1，否则会把所有预测滤光导致全 0！
CONF_THRESH = 0.0
# ───────────────────────────────────────────────────────────────────────────


def main():
    assert 0.0 <= CONF_THRESH < 1.0, (
        f"CONF_THRESH={CONF_THRESH} 不合法！它是 softmax 置信度（概率），"
        f"必须在 [0, 1) 区间。0=关闭过滤，常用 0.6~0.9。填 >=1 会把所有预测滤光。")
    assert os.path.exists(CKPT), f"找不到权重: {CKPT}"
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    assert imgs, f"找不到 val 图像: {IMG_DIR}（先跑 setup_goose 建软链接）"

    model = init_model(CONFIG, CKPT, device="cuda:0")
    TR = np.array(sorted(TRAVERSABLE))
    IG = np.array(sorted(IGNORE))

    tp = fp = gt_tot = 0
    per_prec = []   # 每张图的精确率
    per_rec = []    # 每张图的召回率（仅统计含可通行真值的图）
    # 逐类（像素级累加，用原始预测，不受 CONF_THRESH 影响）
    cls_tp = {c: 0 for c in TRAVERSABLE}
    cls_fp = {c: 0 for c in TRAVERSABLE}
    cls_fn = {c: 0 for c in TRAVERSABLE}
    for i, ip in enumerate(imgs):
        lp = os.path.join(LBL_DIR, os.path.basename(ip))
        if not os.path.exists(lp):
            continue
        res = inference_model(model, ip)
        pred = res.pred_sem_seg.data.squeeze().cpu().numpy()

        low_conf = np.zeros_like(pred, dtype=bool)
        if CONF_THRESH > 0:
            prob = torch.softmax(res.seg_logits.data, 0).max(0).values.cpu().numpy()
            low_conf = prob < CONF_THRESH

        gt = np.array(Image.open(lp))
        valid = ~np.isin(gt, IG)
        pred_t = np.isin(pred, TR) & valid & (~low_conf)
        gt_t = np.isin(gt, TR) & valid

        a = int((pred_t & gt_t).sum())     # 预测可通行且真值可通行 (TP)
        b = int((pred_t & ~gt_t).sum())    # 预测可通行但真值不可通行 (FP, 误报)
        tp += a
        fp += b
        gt_tot += int(gt_t.sum())

        ps = int(pred_t.sum())             # 这张图预测可通行像素数
        gs = int(gt_t.sum())               # 这张图真值可通行像素数
        per_prec.append(a / (ps + 1e-9))   # 每图精确率
        if gs > 0:                         # 没有可通行真值的图不计入召回
            per_rec.append(a / gs)         # 每图召回率

        # 逐类累加（区分类内混淆：预测 asphalt 但真值 gravel 算 asphalt 的 FP）
        for c in TRAVERSABLE:
            pc = (pred == c) & valid
            gc = (gt == c) & valid
            cls_tp[c] += int((pc & gc).sum())
            cls_fp[c] += int((pc & ~gc).sum())
            cls_fn[c] += int((gc & ~pc).sum())

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(imgs)}")

    pp = np.array(per_prec)
    pr = np.array(per_rec)

    def dist(name, arr):
        """打印某指标的 per-image 分布区间 + 均值/中位/达标占比。"""
        bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
        labels = ["  <50%", "50-70%", "70-80%", "80-90%", " >=90%"]
        print("\n[%s] per-image 分布 (共 %d 张):" % (name, len(arr)))
        for j in range(len(labels)):
            m = (arr >= bins[j]) & (arr < bins[j + 1])
            print("    %s : %5.1f %%   (%d 张)" % (labels[j], m.mean() * 100, int(m.sum())))
        print("    --> mean = %.2f %%   median = %.2f %%   >=90%% 占比 = %.1f %%"
              % (arr.mean() * 100, np.median(arr) * 100, (arr >= 0.9).mean() * 100))

    print("\n==================== 可通行区域评测 (val %d imgs) ====================" % len(imgs))
    print("可通行类 id : %s" % sorted(TRAVERSABLE))
    print("置信度阈值  : %.2f" % CONF_THRESH)
    print("--------------------------------------------------------------------")
    print("【二分类·可通行/不可通行】(per-image 算术平均)")
    dist("Precision 精确率", pp)
    dist("Recall 召回率", pr)

    # ── 逐类精确率/召回率（揭示二分类掩盖的类内混淆，原始预测）──────────────
    print("\n--------------------------------------------------------------------")
    print("【逐类·像素级 P/R】(不受置信度阈值影响，揭示类内混淆)")
    print("  %-20s %8s %8s %12s" % ("类别", "精确率", "召回率", "真值像素"))
    p_list, r_list = [], []
    for c in sorted(TRAVERSABLE):
        tpc, fpc, fnc = cls_tp[c], cls_fp[c], cls_fn[c]
        pc = tpc / (tpc + fpc + 1e-9)
        rc = tpc / (tpc + fnc + 1e-9)
        sup = tpc + fnc
        p_list.append(pc)
        r_list.append(rc)
        name = CLASS_NAME.get(c, "cls%d" % c)
        print("  %-20s %7.2f%% %7.2f%% %12d" % (name, pc * 100, rc * 100, sup))
    print("  %-20s %7.2f%% %7.2f%%   (各类算术平均/macro)"
          % ("算术平均", float(np.mean(p_list)) * 100, float(np.mean(r_list)) * 100))
    print("====================================================================")


if __name__ == "__main__":
    main()
