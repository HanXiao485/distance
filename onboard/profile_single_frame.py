"""
profile_single_frame.py — 在同一个进程里直接调用底层函数处理若干帧的完整流程
（跳过 run_sequence.py 的 ProcessPoolExecutor 调度层，避免 cProfile 与多进程 pickle 冲突），
用 cProfile 精确剖析真实单帧串行耗时分布在哪些函数上。

用法:
    python3 -m cProfile -o outputs/single_frame.pstats profile_single_frame.py \
        --config path/to/config.yaml --num-frames 5
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root，让 distance 包可被 import

from distance.run_sequence import (
    read_config, discover_frames, discover_all_images, find_candidate_images,
    process_frame_phase1, process_frame_phase2, ensure_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--num-frames", type=int, default=5, help="剖析前 N 帧（串行处理）")
    args = ap.parse_args()

    seq_cfg = read_config(args.config)
    output_dir = Path(seq_cfg["output_dir"]) if Path(seq_cfg["output_dir"]).is_absolute() \
                 else Path(seq_cfg["project_root"]) / seq_cfg["output_dir"]
    ensure_dir(output_dir)

    seg_enabled = seq_cfg.get("segmentation", {}).get("enabled", False)
    frames, discovery_meta = discover_frames(seq_cfg, require_pred_mask=not seg_enabled)
    frames = frames[: args.num_frames]

    h_window = float(seq_cfg.get("h_search_window_sec", 15.0))
    all_images = discover_all_images(seq_cfg)

    print(f"[INFO] 串行剖析 {len(frames)} 帧，候选池共 {len(all_images)} 张图，窗口 ±{h_window}s")

    for frame in frames:
        anchor_ts = float(frame["frame_id"])
        candidates = find_candidate_images(anchor_ts, all_images, h_window)
        print(f"  帧 {frame['frame_id']}: {len(candidates)} 个候选")

        # skip_existing=False 保证每次都是真实重新计算，不会因为磁盘上已有结果而抄近路
        p1_result = process_frame_phase1(seq_cfg, frame, candidates, output_dir, skip_existing=False)
        if "h_json" not in p1_result:
            print(f"  [WARN] 帧 {frame['frame_id']} 没有找到有效H，跳过phase2")
            continue
        process_frame_phase2(seq_cfg, frame, output_dir, Path(p1_result["h_json"]), skip_existing=False)

    print("[INFO] 剖析跑完")


if __name__ == "__main__":
    main()
