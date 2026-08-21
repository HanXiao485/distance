from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "labels"
OUT_DIR = PROJECT_ROOT / "outputs" / "mask_dirt_preview"

GT_INDEX_PATH = LABEL_DIR / "2d_ground_truth" / "1623379838.017682788_gt_indexlabels.png"
GT_COLOR_PATH = LABEL_DIR / "2d_ground_truth" / "1623379838.017682788_gt_color.png"
PRED_COLOR_PATH = LABEL_DIR / "2d_prediction" / "1623379838.017682788_pred_color.png"

GT_DIRT_ID = 2


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def overlay_mask(image_path: Path, mask: np.ndarray, path: Path, color: tuple[int, int, int]) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.nonzero(mask)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(*color, 130))
    Image.alpha_composite(image, overlay).save(path)


def extract_pred_dirt_mask(pred_rgb: np.ndarray) -> np.ndarray:
    r = pred_rgb[:, :, 0].astype(np.int16)
    g = pred_rgb[:, :, 1].astype(np.int16)
    b = pred_rgb[:, :, 2].astype(np.int16)

    # The prediction file is a transparent overlay on RGB, not an index mask.
    # Dirt appears as a green overlay: G is dominant, with moderate R/B values.
    green_dominant = (g > 95) & (r < 145) & (b < 145)
    separated_from_gray = (g - r > 35) & (g - b > 35)
    not_yellow_foliage = r < 135
    not_dark_sky = g > 110
    return green_dominant & separated_from_gray & not_yellow_foliage & not_dark_sky


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_index = np.array(Image.open(GT_INDEX_PATH))
    gt_dirt = gt_index == GT_DIRT_ID

    pred_rgb = np.array(Image.open(PRED_COLOR_PATH).convert("RGB"))
    pred_dirt = extract_pred_dirt_mask(pred_rgb)

    save_binary(gt_dirt, OUT_DIR / "gt_dirt_binary.png")
    save_binary(pred_dirt, OUT_DIR / "pred_dirt_binary.png")
    overlay_mask(GT_COLOR_PATH, gt_dirt, OUT_DIR / "gt_dirt_overlay.png", (0, 255, 0))
    overlay_mask(PRED_COLOR_PATH, pred_dirt, OUT_DIR / "pred_dirt_overlay.png", (255, 255, 255))

    print(f"GT dirt pixels: {int(gt_dirt.sum())}")
    print(f"Pred dirt pixels: {int(pred_dirt.sum())}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
