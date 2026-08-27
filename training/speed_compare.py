"""
多模型推理速度对比（512×512 输入，随机权重，纯 GPU 前向时间）
不需要 checkpoint，只测速度。

用法 (在容器 .. 下):
    python speed_compare.py --device cuda:0
"""
import os
import sys
import time
import argparse
import torch
from mmengine import Config
from mmseg.registry import MODELS
from mmengine.registry import init_default_scope

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401

WARMUP  = 20
REPEAT  = 200
H, W    = 512, 512
N_CLS   = 12


# ── 模型配置字典（inline，不需要独立 config 文件）────────────────────────────
NORM_CFG = dict(type="SyncBN", requires_grad=True)

MODELS_CFG = {

    "DeepLabV3-R50 (baseline)": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="ResNetV1c", depth=50, num_stages=4,
            out_indices=(0,1,2,3), dilations=(1,1,2,4),
            strides=(1,2,1,1), norm_cfg=NORM_CFG,
            norm_eval=False, style="pytorch", contract_dilation=True),
        decode_head=dict(
            type="ASPPHead", in_channels=2048, in_index=3,
            channels=512, dilations=(1,12,24,36), dropout_ratio=0.1,
            num_classes=N_CLS, norm_cfg=NORM_CFG, align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss")),
        auxiliary_head=dict(
            type="FCNHead", in_channels=1024, in_index=2,
            channels=256, num_convs=1, concat_input=False,
            dropout_ratio=0.1, num_classes=N_CLS, norm_cfg=NORM_CFG,
            align_corners=False, loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "DDRNet-23-slim (baseline)": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="DDRNet", in_channels=3, channels=32,
            ppm_channels=128, norm_cfg=NORM_CFG, align_corners=False),
        decode_head=dict(
            type="DDRHead", in_channels=128, channels=64,
            dropout_ratio=0., num_classes=N_CLS, align_corners=False,
            norm_cfg=NORM_CFG,
            loss_decode=[
                dict(type="CrossEntropyLoss", loss_weight=1.0),
                dict(type="CrossEntropyLoss", loss_weight=0.4)]),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "SegFormer-B0": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="MixVisionTransformer", in_channels=3,
            embed_dims=32, num_stages=4, num_layers=[2,2,2,2],
            num_heads=[1,2,5,8], patch_sizes=[7,3,3,3],
            sr_ratios=[8,4,2,1], out_indices=(0,1,2,3),
            mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.1),
        decode_head=dict(
            type="SegformerHead",
            in_channels=[32,64,160,256], in_index=[0,1,2,3],
            channels=256, dropout_ratio=0.1, num_classes=N_CLS,
            norm_cfg=dict(type="BN", requires_grad=True), align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "SegFormer-B2": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="MixVisionTransformer", in_channels=3,
            embed_dims=64, num_stages=4, num_layers=[3,4,6,3],
            num_heads=[1,2,5,8], patch_sizes=[7,3,3,3],
            sr_ratios=[8,4,2,1], out_indices=(0,1,2,3),
            mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.1),
        decode_head=dict(
            type="SegformerHead",
            in_channels=[64,128,320,512], in_index=[0,1,2,3],
            channels=256, dropout_ratio=0.1, num_classes=N_CLS,
            norm_cfg=dict(type="BN", requires_grad=True), align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "BiSeNetV2": dict(
        type="EncoderDecoder",
        backbone=dict(type="BiSeNetV2", detail_channels=(64,64,128),
                      semantic_channels=(16,32,64,128), semantic_expansion_ratio=6,
                      bga_channels=128, out_indices=(0,1,2,3,4),
                      init_cfg=None, align_corners=False),
        decode_head=dict(
            type="FCNHead", in_channels=128, in_index=0,
            channels=1024, num_convs=1, concat_input=False,
            dropout_ratio=0.1, num_classes=N_CLS,
            norm_cfg=dict(type="BN", requires_grad=True), align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "MobileNetV2-DeepLabV3+": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="MobileNetV2", widen_factor=1., strides=(1,2,2,1,1,1,2),
            dilations=(1,1,1,1,1,2,4), out_indices=(1,2,4,6),
            norm_cfg=NORM_CFG, norm_eval=False),
        decode_head=dict(
            type="DepthwiseSeparableASPPHead",
            in_channels=320, in_index=3, channels=256,
            dilations=(1,12,24,36), c1_in_channels=24, c1_channels=48,
            dropout_ratio=0.1, num_classes=N_CLS, norm_cfg=NORM_CFG,
            align_corners=False, loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),

    "FastSCNN": dict(
        type="EncoderDecoder",
        backbone=dict(
            type="FastSCNN", downsample_dw_channels=(32,48),
            global_in_channels=64, global_block_channels=(64,96,128),
            global_block_strides=(2,2,1), global_out_channels=128,
            higher_in_channels=64, lower_in_channels=128,
            fusion_out_channels=128, out_indices=(0,1,2),
            norm_cfg=NORM_CFG, align_corners=False),
        decode_head=dict(
            type="FCNHead", in_channels=128, in_index=2,
            channels=128, num_convs=1, concat_input=False,
            dropout_ratio=0.1, num_classes=N_CLS,
            norm_cfg=NORM_CFG, align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss")),
        train_cfg=dict(), test_cfg=dict(mode="whole")),
}


def time_model(model, device, name):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)

    def forward_once():
        feats = model.backbone(dummy)
        out   = model.decode_head.forward(feats)
        # DDRNet 返回 tuple，取第一个
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    try:
        with torch.no_grad():
            for _ in range(WARMUP):
                forward_once()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                forward_once()
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / REPEAT * 1000
        return ms
    except Exception as e:
        print(f"  ⚠  {name} 推理失败: {e}")
        return None


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    init_default_scope("mmseg")

    print(f"\n{'='*65}")
    print(f"{'模型':<30} {'参数量':>8}  {'时间(ms)':>10}  {'FPS':>8}  {'状态'}")
    print(f"{'-'*65}")

    results = []
    for name, cfg in MODELS_CFG.items():
        try:
            model = MODELS.build(cfg)
            n_params = count_params(model)
            ms = time_model(model, args.device, name)
            if ms is not None:
                fps = 1000 / ms
                print(f"  {name:<28} {n_params:>7.1f}M  {ms:>10.1f}  {fps:>8.1f}  ✓")
                results.append((name, n_params, ms, fps))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:<28}   {'--':>7}  {'--':>10}  {'--':>8}  ✗ {e}")

    print(f"\n{'='*65}")
    if results:
        print("\n按 FPS 排序:")
        for name, p, ms, fps in sorted(results, key=lambda x: -x[3]):
            print(f"  {fps:6.1f} FPS  {ms:7.1f} ms  {p:5.1f}M  {name}")
    print(f"{'='*65}")
    print(f"输入尺寸: {H}×{W}, batch=1, {REPEAT} 次平均（已预热 {WARMUP} 次）\n")


if __name__ == "__main__":
    main()
