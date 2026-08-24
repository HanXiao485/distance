import numpy as np
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


# scene204 40-class annotation spec (IDs 1-34, 37-40; 0=void→ignore)
# Original label ID → contiguous class index (0-37), void/missing → 255
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


@DATASETS.register_module()
class Scene204Dataset(BaseSegDataset):
    """scene204 self-collected dataset — 38 fine classes.

    Images: WildScenes2d/<CAM>/image/*.jpg
    Labels: WildScenes2d/<CAM>/indexLabel/*.png  (IDs per annotation spec v1.1)

    Label IDs are remapped to contiguous 0-37 by RemapLabel in the pipeline.
    ID 0 (void) and IDs 35/36 (unused) are mapped to ignore_index=255.
    """

    METAINFO = {
        "classes": (
            "road",              # 0
            "lane_marking",      # 1
            "other_road_surface",# 2
            "dirt",              # 3
            "gravel",            # 4
            "other_terrain",     # 5
            "low_grass",         # 6
            "high_grass",        # 7
            "bush",              # 8
            "tree_trunk",        # 9
            "tree_foliage",      # 10
            "sky",               # 11
            "building",          # 12
            "fence",             # 13
            "wall",              # 14
            "bridge",            # 15
            "tunnel",            # 16
            "other_construction",# 17
            "car",               # 18
            "truck",             # 19
            "bus",               # 20
            "motorcycle",        # 21
            "bicycle",           # 22
            "other_vehicle",     # 23
            "person",            # 24
            "rider",             # 25
            "traffic_sign",      # 26
            "traffic_light",     # 27
            "pole",              # 28
            "log",               # 29
            "other_object",      # 30
            "water",             # 31
            "sky_reflection",    # 32
            "other_unknown",     # 33
            "curb",              # 34
            "rock",              # 35
            "mud",               # 36
            "sand",              # 37
        ),
        "palette": [
            (128,  64, 128),  # road
            (255, 255, 255),  # lane_marking
            (128, 128,   0),  # other_road_surface
            ( 60, 180,  75),  # dirt
            (255, 215,   0),  # gravel
            (153,  76,   0),  # other_terrain
            (  0, 102,   0),  # low_grass
            (  0, 204,   0),  # high_grass
            (  0, 255, 128),  # bush
            (  0,  51,   0),  # tree_trunk
            (  0, 153,  51),  # tree_foliage
            (135, 206, 235),  # sky
            ( 70,  70,  70),  # building
            (190, 153, 153),  # fence
            (102, 102, 156),  # wall
            (150, 100, 100),  # bridge
            (150, 120,  90),  # tunnel
            (128,   0,   0),  # other_construction
            (  0,   0, 142),  # car
            (  0,   0,  70),  # truck
            (  0,  60, 100),  # bus
            (  0,   0, 230),  # motorcycle
            (119,  11,  32),  # bicycle
            (  0,  80, 100),  # other_vehicle
            (220,  20,  60),  # person
            (255,   0,   0),  # rider
            (220, 220,   0),  # traffic_sign
            (250, 170,  30),  # traffic_light
            (153, 153, 153),  # pole
            (102,  51,   0),  # log
            (111,  74,   0),  # other_object
            (  0,   0, 128),  # water
            (135, 206, 215),  # sky_reflection
            (  0,   0,   0),  # other_unknown
            (196, 132, 178),  # curb
            (120,  90,  60),  # rock
            (139,  90,  43),  # mud
            (194, 178, 128),  # sand
        ],
    }

    def __init__(self, img_suffix=".jpg", seg_map_suffix=".png",
                 reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
