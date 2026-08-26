dataset_type = 'Scene204Dataset'

# scene204 40-class annotation spec → 38 contiguous classes (no import allowed here)
SCENE204_LABEL_MAP = {
    0: 255,           # void → ignore
    1: 0,             # road
    2: 1,             # lane_marking
    3: 2,             # other_road_surface
    4: 3,             # dirt
    5: 4,             # gravel
    6: 5,             # other_terrain
    7: 6,             # low_grass
    8: 7,             # high_grass
    9: 8,             # bush
    10: 9,            # tree_trunk
    11: 10,           # tree_foliage
    12: 11,           # sky
    13: 12,           # building
    14: 13,           # fence
    15: 14,           # wall
    16: 15,           # bridge
    17: 16,           # tunnel
    18: 17,           # other_construction
    19: 18,           # car
    20: 19,           # truck
    21: 20,           # bus
    22: 21,           # motorcycle
    23: 22,           # bicycle
    24: 23,           # other_vehicle
    25: 24,           # person
    26: 25,           # rider
    27: 26,           # traffic_sign
    28: 27,           # traffic_light
    29: 28,           # pole
    30: 29,           # log
    31: 30,           # other_object
    32: 31,           # water
    33: 32,           # sky_reflection
    34: 33,           # other_unknown
    35: 255,          # unused → ignore
    36: 255,          # unused → ignore
    37: 34,           # curb
    38: 35,           # rock
    39: 36,           # mud
    40: 37,           # sand
}

data_root   = '../data/scene204/WildScenes2d/merged'
crop_size   = (512, 512)
num_classes = 38

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RemapLabel', label_map=SCENE204_LABEL_MAP),
    dict(type='RandomResize', scale=(1920, 1080),
         ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop',  crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip',  prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RemapLabel', label_map=SCENE204_LABEL_MAP),
    dict(type='PackSegInputs'),
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/image', seg_map_path='train/indexLabel'),
    pipeline=train_pipeline,
)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='val/image', seg_map_path='val/indexLabel'),
    pipeline=test_pipeline,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)

test_dataloader = val_dataloader

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
randomness     = dict(seed=0)
