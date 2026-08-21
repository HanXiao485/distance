# DDRNet-23-slim on GOOSE 2D (64 classes), mmsegmentation.
# 复用 GOOSE 数据集配置(_base_/datasets/goose.py)+ 标准 80k 计划。
# DDRNet 网络代码由 mmseg 自带，无需额外模型代码。
_base_ = [
    '../_base_/datasets/goose.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

crop_size = _base_.crop_size        # (512, 512)
num_classes = _base_.num_classes    # 64
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    # 整图推理(2048x1000)时把尺寸补齐到 32 的倍数；若报尺寸不匹配错误改成 64
    test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ImageNet 预训练(与 DeepLabV3、GOOSE 官方对齐)。
        # 权重位于 /workspace/WildScenes/pretrained_models/ 下
        init_cfg=dict(type='Pretrained',
                      checkpoint='pretrained_models/ddrnet23s-in1kpre_3rdparty-1ccac5b1.pth')),
    decode_head=dict(
        type='DDRHead',
        in_channels=32 * 4,
        channels=64,
        dropout_ratio=0.,
        num_classes=num_classes,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# checkpoint：每 1000 步存一次、保留 2 个、存最优 mIoU(与 deeplabv3 一致)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
                    max_keep_ckpts=2, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
