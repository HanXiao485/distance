_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/scene204.py',
    '../_base_/default_runtime.py',
]

crop_size   = (512, 512)
num_classes = 38

data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='whole'),
)

# ── Fine-tune from GOOSE Category checkpoint ──────────────────────────────────
# Head shape mismatch (12→38 classes) is skipped automatically (strict=False)
load_from = (
    '../work_dirs/goose_category_deeplabv3/'
    'best_mIoU_iter_78000.pth'
)

# ── Training schedule (5k iters, small dataset) ───────────────────────────────
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=500)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4),
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=0,
        end=5000,
        by_epoch=False,
    )
]

work_dir = '../work_dirs/scene204_deeplabv3_finetune'
