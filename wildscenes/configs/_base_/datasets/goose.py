dataset_type = "GooseDataset"

# ── Modify this path ─────────────────────────────────────────────────────────
# Point to the flattened mmseg-style GOOSE directory produced by
# scripts/data/setup_goose.py  (it contains train/ and val/ subdirs, each with
# image/ and indexLabel/).
data_root = "../data/processed/goose2d"
# ─────────────────────────────────────────────────────────────────────────────

# GOOSE images are 2048x1000.
crop_size = (512, 512)
num_classes = 64

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 1000),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/image', seg_map_path='train/indexLabel'),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=20,   # lower to 2-4 if you hit GPU OOM
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='val/image', seg_map_path='val/indexLabel'),
    pipeline=test_pipeline)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

# GOOSE has no separate public test split here; reuse val for the test loader.
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='val/image', seg_map_path='val/indexLabel'),
    pipeline=test_pipeline)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
randomness = dict(seed=0)
