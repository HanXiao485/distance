"""
audit_labels.py — 扫描 GOOSE train 标签，找出可能导致 CUDA illegal memory access 的异常文件。

检查项：
1. PIL 能否正常打开（排除损坏/截断文件）
2. 图像 mode（应为单通道 'L' 或 'P'，不应是 RGB/RGBA）
3. 标签尺寸是否和对应 RGB 图一致
4. 像素唯一值是否超出预期范围（GOOSE 64 类 fine label 应 < 64；
   任何 >= 64 但 < 256 的值会被 RemapLabel 映射为 ignore(255)，本身不是 bug，
   但 mode 异常或者尺寸不匹配才是真正会导致下游 tensor 形状错乱的原因）

用法:
    python audit_labels.py \
        --img-dir ../data/processed/goose2d/train/image \
        --lbl-dir ../data/processed/goose2d/train/indexLabel
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--lbl-dir", required=True)
    args = ap.parse_args()

    lbl_paths = sorted(glob.glob(os.path.join(args.lbl_dir, "*.png")))
    print(f"共 {len(lbl_paths)} 个标签文件，开始检查...")

    issues = []
    max_val_seen = 0

    for i, lbl_path in enumerate(lbl_paths):
        stem = os.path.basename(lbl_path)
        img_path = os.path.join(args.img_dir, stem)

        # 1. 标签能否打开
        try:
            lbl_im = Image.open(lbl_path)
            lbl_im.load()
        except Exception as e:
            issues.append((stem, f"标签无法打开: {e}"))
            continue

        # 2. mode 检查
        if lbl_im.mode not in ("L", "P", "I"):
            issues.append((stem, f"标签 mode 异常: {lbl_im.mode}（期望单通道 L/P）"))

        lbl_arr = np.array(lbl_im)

        # 3a. 维度检查：分割标签必须是 2D (H, W)，如果是 3D 说明误存成了 RGB/RGBA，
        #     这种形状错乱在下游会导致 tensor shape 与 logits 不匹配，
        #     GPU 端表现为非法内存访问而不是清晰的 Python 报错。
        if lbl_arr.ndim != 2:
            issues.append((stem, f"标签不是单通道: shape={lbl_arr.shape}（期望 2D）"))
            continue

        # 3b. 像素值范围检查（uint8 情况下最大只能是255，重点看是不是超过合理的 fine-label 范围）
        local_max = int(lbl_arr.max())
        local_min = int(lbl_arr.min())
        max_val_seen = max(max_val_seen, local_max)
        if local_min < 0 or local_max > 255:
            issues.append((stem, f"像素值超出 uint8 范围: min={local_min} max={local_max}"))
        elif local_max >= 64 and local_max != 255:
            # GOOSE fine label 应该 < 64；如果不是 255(ignore)，就是异常值
            issues.append((stem, f"出现未知类别值: max={local_max}（GOOSE fine label 应 < 64）"))

        # 4. 尺寸是否和 RGB 图一致
        if os.path.exists(img_path):
            try:
                img_im = Image.open(img_path)
                if img_im.size != lbl_im.size:
                    issues.append((stem, f"尺寸不匹配: 图 {img_im.size} vs 标签 {lbl_im.size}"))
            except Exception as e:
                issues.append((stem, f"对应 RGB 图无法打开: {e}"))
        else:
            issues.append((stem, "找不到对应的 RGB 图"))

        if (i + 1) % 1000 == 0:
            print(f"  已检查 {i + 1}/{len(lbl_paths)}")

    print(f"\n检查完成。全局最大像素值: {max_val_seen}")
    print(f"发现 {len(issues)} 个问题文件：\n")
    for stem, msg in issues:
        print(f"  {stem}: {msg}")

    if not issues:
        print("  未发现异常，标签数据看起来是干净的。")


if __name__ == "__main__":
    main()
