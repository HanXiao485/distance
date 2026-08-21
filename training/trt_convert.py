"""
将 mmseg 模型导出 ONNX，再转为 TensorRT engine，对比 PyTorch vs TRT 推理速度。

用法 (在容器 /root/distance/WildScenes 下):
    # DeepLabV3
    python trt_convert.py \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_deeplabv3/best_mIoU_iter_78000.pth \
        --name deeplabv3 --device cuda:0

    # DDRNet
    python trt_convert.py \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_ddrnet/best_mIoU_iter_80000.pth \
        --name ddrnet --device cuda:0
"""
import os
import sys
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
from mmseg.apis import init_model

WARMUP = 20
REPEAT = 200
H, W   = 1512, 2016


# ── mmseg 模型推理包装（绕过 data_preprocessor，直接输入归一化 tensor）────────
class SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone     = model.backbone
        self.decode_head  = model.decode_head
        self.align_corners = model.decode_head.align_corners

    def forward(self, x):
        feats = self.backbone(x)
        # decode_head.forward 返回 logits（不做 resize）
        out = self.decode_head.forward(feats)
        # 统一 resize 到输入尺寸
        out = torch.nn.functional.interpolate(
            out, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=self.align_corners)
        return out


def benchmark_pytorch(wrapper, dummy, device):
    wrapper.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = wrapper(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = wrapper(dummy)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / REPEAT * 1000
    return ms


def export_onnx(wrapper, dummy, onnx_path, fp16=False):
    if fp16:
        wrapper = wrapper.half()
        dummy = dummy.half()
        print("  ONNX export in FP16")
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  ONNX saved: {onnx_path}")


def build_trt_engine(onnx_path, engine_path, fp16=True):
    try:
        import tensorrt as trt
    except ImportError:
        print("  [skip] tensorrt not installed")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network()
    parser     = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    # TRT 11 removed BuilderFlag.FP16; FP16 is achieved via FP16 ONNX export
    print("  Engine precision matches ONNX dtype")

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"  TRT engine saved: {engine_path}")
    return engine


def benchmark_trt(engine_path):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        print("  [skip] tensorrt/pycuda not available")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # 分配 IO buffer
    input_shape  = tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))
    output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    d_input  = cuda.mem_alloc(int(np.prod(input_shape))  * 4)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * 4)
    # detect engine input dtype
    inp_dtype = np.float16 if engine.get_tensor_dtype(engine.get_tensor_name(0)) == trt.DataType.HALF else np.float32
    h_input  = np.random.randn(*input_shape).astype(inp_dtype)
    d_input  = cuda.mem_alloc(int(np.prod(input_shape)) * h_input.itemsize)
    out_dtype = np.float16 if engine.get_tensor_dtype(engine.get_tensor_name(1)) == trt.DataType.HALF else np.float32
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(out_dtype).itemsize)
    stream   = cuda.Stream()

    def run_once():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        ctx.set_tensor_address("input",  int(d_input))
        ctx.set_tensor_address("output", int(d_output))
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()

    for _ in range(WARMUP):
        run_once()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        run_once()
    ms = (time.perf_counter() - t0) / REPEAT * 1000
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--name",   required=True, help="模型名称，用于输出文件名")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16",   action="store_true", default=True)
    args = parser.parse_args()

    out_dir = "/root/distance/WildScenes/trt_engines"
    os.makedirs(out_dir, exist_ok=True)
    onnx_path   = os.path.join(out_dir, f"{args.name}.onnx")
    engine_path = os.path.join(out_dir, f"{args.name}_fp16.trt")

    print(f"\n{'='*55}")
    print(f"模型: {args.name}")
    print(f"{'='*55}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print("加载模型...")
    model   = init_model(args.config, args.ckpt, device=args.device)
    wrapper = SegWrapper(model).to(args.device).eval()
    dummy   = torch.randn(1, 3, H, W, device=args.device)

    # ── PyTorch 基准 ──────────────────────────────────────────────────────────
    print(f"\n[1] PyTorch 推理 ({REPEAT} 次)...")
    ms_pt = benchmark_pytorch(wrapper, dummy, args.device)
    print(f"    平均: {ms_pt:.1f} ms  ({1000/ms_pt:.1f} FPS)")

    # ── 导出 ONNX ─────────────────────────────────────────────────────────────
    print("\n[2] 导出 ONNX...")
    try:
        if args.fp16:
            export_onnx(wrapper, dummy, onnx_path, fp16=True)          # stay on GPU for FP16
        else:
            export_onnx(wrapper.cpu(), dummy.cpu(), onnx_path, fp16=False)
    except Exception as e:
        print(f"  ONNX 导出失败: {e}")
        return

    # ── 构建 TRT engine ───────────────────────────────────────────────────────
    print("\n[3] 构建 TensorRT engine (FP16)...")
    try:
        build_trt_engine(onnx_path, engine_path, fp16=args.fp16)
    except Exception as e:
        print(f"  TRT 构建失败: {e}")

    # ── TRT 推理基准 ──────────────────────────────────────────────────────────
    if os.path.exists(engine_path):
        print(f"\n[4] TensorRT 推理 ({REPEAT} 次)...")
        ms_trt = benchmark_trt(engine_path)
        if ms_trt:
            print(f"    平均: {ms_trt:.1f} ms  ({1000/ms_trt:.1f} FPS)")
            print(f"\n  加速比: {ms_pt/ms_trt:.2f}×  ({ms_pt:.1f} ms → {ms_trt:.1f} ms)")

    print(f"\n{'='*55}")


if __name__ == "__main__":
    main()
