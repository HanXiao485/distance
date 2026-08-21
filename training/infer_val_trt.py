"""
infer_val_trt.py
用 TRT FP16 引擎在 GOOSE val 集上推理 N 张图，报告 per-class IoU、mIoU 和推理速度。

用法:
    python infer_val_trt.py \
        --engine trt_engines/deeplabv3_fp16.trt \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --name DeepLabV3

    python infer_val_trt.py \
        --engine trt_engines/ddrnet_fp16.trt \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --name DDRNet
"""
import os, sys, argparse, time, random, glob
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401

FINE_TO_CATEGORY = {
    0: 0, 8: 0, 56: 0,
    53: 1,
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2,
    30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4, 43: 4, 44: 4, 55: 4, 58: 4,
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5, 35: 5,
    36: 5, 37: 5, 49: 5, 57: 5, 63: 5,
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,
    14: 9, 32: 9,
    54: 10,
    33: 11,
}
LABEL_LUT = np.full(256, 255, dtype=np.uint8)
for _src, _dst in FINE_TO_CATEGORY.items():
    LABEL_LUT[_src] = _dst

CLASS_NAMES  = ["void", "sky", "vegetation", "terrain", "construction",
                "vehicle", "road", "object", "sign", "human", "water", "animal"]
NUM_CLASSES  = 12
IGNORE_INDEX = 255
DATA_ROOT    = "/root/distance/WildScenes/DISTANCE/datasets/Goose/goose_2d_val"
H, W         = 1000, 2048


def get_norm_params(config_path):
    """从 mmseg config 读取 data_preprocessor 的 mean/std，失败则用 ImageNet 默认值。"""
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        dp  = cfg.get("data_preprocessor", {})
        mean = np.array(dp.get("mean", [123.675, 116.28,  103.53]), dtype=np.float32)
        std  = np.array(dp.get("std",  [58.395,  57.12,   57.375]), dtype=np.float32)
        bgr2rgb = dp.get("bgr_to_rgb", True)
        return mean, std, bgr2rgb
    except Exception:
        return (np.array([123.675, 116.28, 103.53], dtype=np.float32),
                np.array([58.395,  57.12,  57.375], dtype=np.float32),
                True)


def preprocess(img_path, mean, std, bgr2rgb, inp_dtype):
    img = np.array(Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR),
                   dtype=np.float32)          # (H, W, 3) RGB
    if not bgr2rgb:
        img = img[:, :, ::-1]                # RGB → BGR if model expects BGR
    img = (img - mean) / std
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(inp_dtype))  # (1, 3, H, W)
    return x


def accumulate_intersect_union(pred, gt, total_inter, total_union):
    """按 mmseg IoUMetric 的方式在整个数据集上累积每类的交集/并集像素数。"""
    valid = gt != IGNORE_INDEX
    for c in range(NUM_CLASSES):
        p = (pred == c) & valid
        g = (gt == c) & valid
        total_inter[c] += int((p & g).sum())
        total_union[c] += int((p | g).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TRT .trt engine 路径")
    ap.add_argument("--config", required=True, help="mmseg config 路径（读取归一化参数）")
    ap.add_argument("--name",   default="model")
    ap.add_argument("--num",    type=int, default=50)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    # ── 加载 TRT 引擎 ───────────────────────────────────────────────────────
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    inp_trt  = engine.get_tensor_dtype(inp_name)
    out_trt  = engine.get_tensor_dtype(out_name)
    inp_np   = np.float16 if inp_trt == trt.DataType.HALF else np.float32
    out_np   = np.float16 if out_trt == trt.DataType.HALF else np.float32

    print(f"引擎输入 dtype: {inp_np.__name__}  输出 dtype: {out_np.__name__}")

    input_shape  = (1, 3, H, W)
    output_shape = (1, NUM_CLASSES, H, W)
    d_input  = cuda.mem_alloc(int(np.prod(input_shape))  * np.dtype(inp_np).itemsize)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(out_np).itemsize)
    h_output = np.empty(output_shape, dtype=out_np)
    stream   = cuda.Stream()

    ctx.set_tensor_address(inp_name, int(d_input))
    ctx.set_tensor_address(out_name, int(d_output))

    # ── 读归一化参数 ────────────────────────────────────────────────────────
    mean, std, bgr2rgb = get_norm_params(args.config)
    print(f"归一化 mean={mean}  std={std}  bgr2rgb={bgr2rgb}")

    # ── 抽取 val 图片 ───────────────────────────────────────────────────────
    img_dir = os.path.join(DATA_ROOT, "images", "val")
    ann_dir = os.path.join(DATA_ROOT, "labels", "val")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "**", "*_windshield_vis.png"), recursive=True))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {img_dir}")

    random.seed(args.seed)
    sample = random.sample(img_paths, min(args.num, len(img_paths)))

    print(f"\n{'='*60}")
    print(f"模型  : {args.name}")
    print(f"引擎  : {args.engine}")
    print(f"Val图 : {len(img_paths)} 张，抽样 {len(sample)} 张")
    print(f"{'='*60}")

    total_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    times   = []
    skipped = 0

    for rank, img_path in enumerate(sample):
        rel      = os.path.relpath(img_path, img_dir)
        ann_path = os.path.join(ann_dir, rel.replace("_windshield_vis.png", "_labelids.png"))
        if not os.path.exists(ann_path):
            skipped += 1
            continue

        gt_raw  = np.array(Image.open(ann_path), dtype=np.uint8)
        gt      = LABEL_LUT[gt_raw]
        orig_wh = (gt_raw.shape[1], gt_raw.shape[0])  # PIL format (W, H)

        h_input = preprocess(img_path, mean, std, bgr2rgb, inp_np)

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

        # argmax → class label map
        logits = h_output[0].astype(np.float32)           # (12, H, W)
        pred   = np.argmax(logits, axis=0).astype(np.uint8)  # (H, W) at engine resolution
        if orig_wh != (W, H):
            pred = np.array(Image.fromarray(pred).resize(orig_wh, Image.NEAREST))

        accumulate_intersect_union(pred, gt, total_inter, total_union)

        if (rank + 1) % 10 == 0:
            print(f"  [{rank+1}/{len(sample)}] 完成")

    if times == []:
        print("没有有效样本，检查数据路径")
        return

    with np.errstate(invalid="ignore", divide="ignore"):
        per_cls = np.where(total_union > 0, total_inter / total_union, np.nan) * 100
    miou     = float(np.nanmean(per_cls))
    avg_ms   = float(np.mean(times[3:])) if len(times) > 3 else float(np.mean(times))

    print(f"\n{'─'*44}")
    print(f"{'类别':<20} {'IoU (%)':>10}")
    print(f"{'─'*44}")
    for name, iou in zip(CLASS_NAMES, per_cls):
        val = f"{iou:>9.1f}%" if not np.isnan(iou) else "       n/a"
        print(f"{name:<20} {val}")
    print(f"{'─'*44}")
    print(f"{'mIoU':<20} {miou:>9.2f}%")
    print(f"{'─'*44}")
    print(f"\n推理速度 (含IO+预处理): {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
    if skipped:
        print(f"跳过: {skipped} 张（无对应标注）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
