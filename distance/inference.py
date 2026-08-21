"""
Semantic segmentation inference for DISTANCE pipeline.
Generates pure colour masks (palette[pred]) from RGB images using mmseg models.
Output format is compatible with DISTANCE's color_threshold mask reader.
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def run_inference(
    image_paths: list[str],
    output_dir: Path,
    config: str,
    checkpoint: str,
    device: str = "cuda:0",
) -> dict[str, Path]:
    """
    Run segmentation inference on a list of images and save pure colour masks.

    Parameters
    ----------
    image_paths  : list of RGB image paths to process
    output_dir   : directory where pred masks are saved
    config       : mmseg model config file path
    checkpoint   : model checkpoint (.pth) file path
    device       : torch device string

    Returns
    -------
    dict mapping original image path → saved mask path
    """
    import sys as _sys
    import os as _os
    # Register custom transforms (e.g. RemapLabel) from WildScenes package
    _ws_root = _os.environ.get("WILDSCENES_ROOT", "/root/distance/WildScenes")
    if _ws_root not in _sys.path:
        _sys.path.insert(0, _ws_root)
    try:
        from wildscenes.mmseg_wildscenes.dataset import goose_category  # noqa: F401
    except ImportError:
        pass  # not a WildScenes model — custom transforms not needed

    from mmseg.apis import init_model, inference_model
    from mmseg.registry import VISUALIZERS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model from {checkpoint}")
    model = init_model(config, checkpoint, device=device)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    palette = np.array(visualizer.dataset_meta["palette"], dtype=np.uint8)

    results: dict[str, Path] = {}

    for img_path in tqdm(image_paths, desc="Phase 0 (inference)", unit="frame"):
        result = inference_model(model, img_path)

        pred = (
            result.pred_sem_seg.data
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        color_mask = palette[pred]   # H × W × 3

        out_file = output_dir / Path(img_path).name
        Image.fromarray(color_mask, mode="RGB").save(out_file)

        results[img_path] = out_file

    print(f"[INFO] Inference done. {len(results)} masks saved to {output_dir}")
    return results
