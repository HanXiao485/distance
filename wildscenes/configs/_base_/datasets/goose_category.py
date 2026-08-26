dataset_type = "GooseCategoryDataset"

# GOOSE 64 fine-class → 12 category mapping (inline, no import allowed in mmseg config)
FINE_TO_CATEGORY = {
    0: 0, 8: 0, 56: 0,           # Void
    53: 1,                        # Sky
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2,
    30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,  # Vegetation
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,            # Terrain
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4,
    43: 4, 44: 4, 55: 4, 58: 4,                 # Construction
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5,
    35: 5, 36: 5, 37: 5, 49: 5, 57: 5, 63: 5,  # Vehicle
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,    # Road
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,    # Object
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,  # Sign
    14: 9, 32: 9,                # Human
    54: 10,                      # Water
    33: 11,                      # Animal
}

# Same data root as the 64-class GOOSE config (produced by setup_goose.py).
data_root = "../data/processed/goose2d"

crop_size = (512, 512)
num_classes = 12   # Void, Sky, Vegetation, Terrain, Construction, Vehicle,
                   # Road, Object, Sign, Human, Water, Animal

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RemapLabel', label_map=FINE_TO_CATEGORY),   # 64 → 12 on-the-fly
    dict(
        type='RandomResize',
        scale=(2048, 1000),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RemapLabel', label_map=FINE_TO_CATEGORY),
    dict(type='PackSegInputs'),
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/image', seg_map_path='train/indexLabel'),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=20,
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
