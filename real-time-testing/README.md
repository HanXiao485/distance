# Real-time full-pipeline test

本目录提供独立的 scene204 全流程实时性实验，不修改 DISTANCE、第一视角或 BEV 的现有程序。最终视频使用 H.264、`yuv420p` 和 fast-start，首次运行前安装独立依赖：

```bash
pip install -r real-time-testing/requirements.txt
```

## 实验流程

```text
时间戳图片 → 5 FPS 输入视频
             ↓
逐帧视频解码 → 分割 → H/物理边界误差数值 → 轻量第一视角 overlay → BEV
             ↓
左侧实时输入 + 右侧最近完成的 FPV/BEV → 延迟对照视频
```

处理采用单进程串行方式，模型仅初始化一次。程序真正逐帧读取生成的 `input_video.mp4`，后续阶段消费解码帧；H 估计只搜索截至当前时刻已经解码的帧，不使用未来画面。若当前时刻无法拟合有效 H，则复用最近一次成功的 H，并在该帧报告中写入 `h_fallback: true` 和 `h_fallback_source`，保证实时因果性且明确披露回退。实时测试会强制关闭 `h_diagnostics` 和 `scanline.save_overlays`，并关闭 GT/Pred boundary PNG 输出，误差阶段仅保留数值结果；H 搜索与拟合逻辑保持不变。误差阶段与 BEV 共享当前帧的内存点云，BEV 使用缓存静态背景的 OpenCV 画布。FPV、BEV 不再逐帧写入和读回 PNG，而是以内存图像直接交给最终视频编码器。每帧记录实际墙钟耗时：

- `capture_decode_sec`：视频帧读取、解码及为现有流水线落盘；
- `segmentation_sec`：预处理、GPU 推理、后处理及二值 mask 保存；
- `error_calculation_sec`：H 候选与边界误差计算；
- `fpv_sec`：独立的轻量预测第一视角叠加渲染；
- `bev_sec`：BEV 图渲染与保存；
- `processing_sec`：上述完整帧处理的实际总耗时。

模型和所有启用阶段会在正式时间线之前完成预热。预热期间不消费输入视频帧、不播放输入画面，预热耗时记录为 `warmup_ms`，但不计入处理延迟。视频时间线从预热完成后归零，并按输入 FPS 产生帧到达事件。右侧结果只有在实测处理完成时刻之后才显示；若处理速度低于输入速度，左侧继续前进，右侧保持最近完成结果，从而呈现真实积压和可视延迟。输入结束后，视频会继续播放到最后一项结果完成。

## 配置阶段开关

默认读取 `real-time-testing/pipeline_config.yaml`。在 `stages` 中将任一阶段设为 `false` 即可跳过：

```yaml
stages:
  segmentation: true
  error_calculation: false
  fpv_visualization: true
  bev_visualization: true
  comparison_video: true
```

各开关相互独立：

- `segmentation=false`：不加载分割模型；误差或第一视角仍启用时，从 `segmentation.cached_mask_dir` 读取缓存 mask。该值为 `null` 时使用所选 DISTANCE 任务的 `pred_mask_dir`。
- `error_calculation=false`：完全跳过 H 估计、扫描线匹配和误差数值计算。此时若保留第一视角，只绘制预测 mask；若保留 BEV，则输出 LiDAR-only BEV。
- `fpv_visualization=false`：不生成第一视角图；误差计算仍可独立运行。
- `bev_visualization=false`：不加载现有 BEV 模块，也不生成 BEV 图。
- `comparison_video=false`：只保留输入视频、阶段产物和计时报告，不合成最终对照视频。

运行配置：

```bash
python real-time-testing/run_realtime_test.py \
  --pipeline-config real-time-testing/pipeline_config.yaml
```

命令行中的 `--config`、`--task`、`--output-dir`、`--fps`、`--device`、`--max-frames`、`--warmup` 和 `--max-output-duration` 仍可覆盖配置文件中的对应值。

## 1. 单独生成输入视频

```bash
cd /root/distance
conda activate distance
python real-time-testing/create_input_video.py   --image-dir data/scene204/WildScenes2d/FRONT_CAMERA/image   --output real-time-testing/outputs/input_video.mp4   --fps 5
```

## 2. 小规模全流程验证

```bash
python real-time-testing/run_realtime_test.py   --max-frames 2   --warmup 1   --output-dir real-time-testing/outputs/smoke
```

## 3. 全流实验

```bash
python real-time-testing/run_realtime_test.py   --config configs/evaluation.yaml   --task scene204_road   --fps 5   --device cuda:0   --warmup 1   --output-dir real-time-testing/outputs/scene204_road
```

默认不限制最终视频长度，以完整显示处理积压。若仅需预览，可添加例如
`--max-output-duration 60`，但这会截断尚未完成的尾部结果。

## 输出

```text
real-time-testing/outputs/scene204_road/
├── input_video.mp4              # 原始数据集图片拼接视频
├── realtime_comparison.mp4      # 左输入、右 FPV+BEV 的真实延迟视频
├── timing_report.json           # 每帧阶段耗时及汇总统计
├── decoded_frames/              # 从输入视频实际解码的流水线输入
├── pred_masks/                  # 分割二值 mask
└── pipeline/frames/             # H、扫描线误差及数值汇总

FPV/BEV 帧仅在内存中传递，不逐帧落盘。
```

`timing_report.json` 中同时保存模型加载时间、每帧输入到达时刻、开始处理时刻、完成时刻、端到端延迟，以及各阶段 mean/P50/P90/max。
