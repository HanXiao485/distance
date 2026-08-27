import numpy as np
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmcv.transforms import BaseTransform

# GOOSE 64 fine-class → 12 category mapping (GOOSE paper Table II, arxiv 2310.16788)
# Category indices 0-11: Void, Sky, Vegetation, Terrain, Construction, Vehicle,
#                        Road, Object, Sign, Human, Water, Animal
FINE_TO_CATEGORY = {
    # Void (0)
    0: 0, 8: 0, 56: 0,
    # Sky (1)
    53: 1,
    # Vegetation (2)
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2,
    30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,
    # Terrain (3)
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,
    # Construction (4)
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4, 43: 4, 44: 4, 55: 4, 58: 4,
    # Vehicle (5)
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5, 35: 5,
    36: 5, 37: 5, 49: 5, 57: 5, 63: 5,
    # Road (6)
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,
    # Object (7)
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,
    # Sign (8)
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,
    # Human (9)
    14: 9, 32: 9,
    # Water (10)
    54: 10,
    # Animal (11)
    33: 11,
}


@TRANSFORMS.register_module()
class RemapLabel(BaseTransform):
    """Remap gt_seg_map pixel values via LUT. Used to convert GOOSE 64-class
    labels to 12-category labels on-the-fly during data loading.
    Any source value not in label_map is mapped to ignore_index (255).
    """

    def __init__(self, label_map: dict, ignore_index: int = 255):
        lut = np.full(256, ignore_index, dtype=np.uint8)
        for src, dst in label_map.items():
            if 0 <= int(src) < 256:
                lut[int(src)] = int(dst)
        self.lut = lut

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results:
            seg = results['gt_seg_map']
            results['gt_seg_map'] = self.lut[seg.astype(np.uint8)]
        return results


@DATASETS.register_module()
class GooseCategoryDataset(BaseSegDataset):
    """GOOSE 2D semantic segmentation — 12 category level.

    Fine labels (0-63, from *_labelids.png) are remapped to 12 categories
    by the RemapLabel transform inserted in the data pipeline.
    Uses the same data layout as GooseDataset (created by setup_goose.py).

    Category 0  Void         : undefined, ego_vehicle, outlier
    Category 1  Sky          : sky
    Category 2  Vegetation   : forest, bush, moss, leaves, tree_crown,
                               tree_trunk, crops, low_grass, high_grass,
                               scenery_vegetation, hedge, tree_root
    Category 3  Terrain      : snow, cobble, asphalt, gravel, soil
    Category 4  Construction : debris, building, wall, fence, guard_rail,
                               bridge, tunnel, wire, container
    Category 5  Vehicle      : car, bicycle, bus, motorcycle, truck, on_rails,
                               caravan, trailer, kick_scooter, heavy_machinery,
                               military_vehicle
    Category 6  Road         : bikeway, pedestrian_crossing, road_marking,
                               sidewalk, curb, rail_track
    Category 7  Object       : obstacle, street_light, rock, pole, barrel, pipe
    Category 8  Sign         : traffic_cone, road_block, traffic_light,
                               boom_barrier, traffic_sign, misc_sign,
                               barrier_tape
    Category 9  Human        : person, rider
    Category 10 Water        : water
    Category 11 Animal       : animal
    """

    METAINFO = {
        "classes": (
            "void", "sky", "vegetation", "terrain", "construction",
            "vehicle", "road", "object", "sign", "human", "water", "animal",
        ),
        "palette": [
            (0,   0,   0),    # void         - black
            (128, 200, 255),  # sky           - light blue
            (50,  200,  50),  # vegetation    - green
            (180, 130,  70),  # terrain       - brown
            (120, 120, 120),  # construction  - gray
            (255, 100,   0),  # vehicle       - orange
            (180, 180, 180),  # road          - light gray
            (255, 200,   0),  # object        - yellow
            (255,   0,   0),  # sign          - red
            (0,     0, 255),  # human         - blue
            (0,   100, 255),  # water         - deep blue
            (255, 128, 200),  # animal        - pink
        ],
    }

    def __init__(self, img_suffix=".png", seg_map_suffix=".png",
                 reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
