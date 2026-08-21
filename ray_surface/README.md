# Ray-Surface: 表面重建 + 相机射线求交

H 矩阵方法之外的第二套物理边界误差计算方案。

## 核心思路

1. **表面重建**：把每帧的 LiDAR 点云变换到世界坐标系，在 BEV（俯视）平面构建高程图（elevation map）
2. **射线生成**：对每条扫描线上的 GT / Pred 边界像素点，从相机中心出发发射射线
3. **射线求交**：找到射线与高程图表面的交点，得到边界点的 3D 物理坐标
4. **误差计算**：GT 与 Pred 边界点在 3D 世界坐标系中的欧氏距离

与 H 矩阵方法不同：**不依赖单平面假设**，能处理非平坦地形。

---

## 文件结构

```
ray_surface/
├── run_ray.py          # 主运行脚本
配置文件位于 `../configs/ray_surface.yaml`
├── modules/
│   ├── surface.py      # ElevationMap 类 + build_elevation_map()
│   └── visualize.py    # 可视化工具
└── README.md
```

---

## 运行方式

```bash
cd /root/distance/WildScenes/DISTANCE
python ray_surface/run_ray.py --config configs/ray_surface.yaml
```

---

## 输出

结果保存在 `outputs/ray_surface_local/`：

```
outputs/ray_surface_local/
├── sequence_summary.json               # 汇总统计
├── aggregate_error_histogram.png       # 全局误差直方图
└── <frame_id>/
    ├── frame_result.json               # 单帧统计
    ├── scanline_overlay.png            # 扫描线 + 边界点可视化（叠加在原图上）
    ├── bev_intersections.png           # 俯视图：GT / Pred 交点位置
    └── error_histogram.png             # 单帧误差直方图
```

### 关键指标（mini 数据集 43 帧）

| 指标 | Ray-Surface | H-Matrix |
|------|-------------|----------|
| mean_of_frame_means | **0.2655m** | 0.293m |
| median_of_frame_medians | **0.0325m** | 0.043m |
| P90 | **0.520m** | 0.748m |
| P95 | **0.567m** | 1.355m |
| 处理帧数 | 43/43 | 43/43 |

---

## 参数说明

### `surface` 表面重建

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cell_size_m` | 0.25 | BEV 格子大小（米），越小越精细但越慢 |
| `lidar_z_min` | -1.5 | LiDAR 坐标系下地面点 Z 下界 |
| `lidar_z_max` | 1.0  | LiDAR 坐标系下地面点 Z 上界（过滤树木等） |

### `ray` 射线参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `t_min` | 0.5 | 射线最近搜索距离（米） |
| `t_max` | 60.0 | 射线最远搜索距离（米） |
| `n_coarse` | 300 | 粗搜索采样点数 |
| `n_bisect` | 25 | 二分精化次数 |
| `max_distance_m` | 30.0 | 超过此距离的交点丢弃（类似 H 矩阵 20m 过滤） |

---

## 表面重建原理

使用 **BEV 高程图**（Elevation Map）：

1. LiDAR 点云（~83k 点）→ Z 范围过滤保留地面点（~27k 点）
2. 世界坐标系 XY 平面栅格化，格子大小 0.25m
3. 每个格子取中位数 Z 值
4. 空格子用 scipy 最近邻插值填充
5. 结果：覆盖约 9% 格子（有实测点），其余由插值填充

---

## 射线求交算法

```
f(t) = origin_z + t·dir_z - Z(origin_x + t·dir_x, origin_y + t·dir_y)

1. 沿射线均匀采样 t ∈ [t_min, t_max]（300步）
2. 找到第一个 f(t) 从正变负（射线从地面上方穿入地面）
3. 在该区间内二分搜索（25次），精化交点位置
4. 返回 3D 交点坐标和距离 t
```

---

## 当前局限

1. **性能**：每帧约 6 秒（Python 纯循环，可用 NumPy 向量化提速 10x）
2. **表面稀疏区域**：地表覆盖率约 9%，边缘区域依赖插值，精度较低
3. **射线漏掉情况**：约 15% 的扫描线射线未命中（边界在远场或高程图边界外）
4. **误差度量**：当前是 3D 欧氏距离，可改为沿扫描线方向的横向分量

## 后续改进方向

- **性能**：向量化射线采样（一次 numpy 操作处理所有扫描线）
- **精度**：使用多帧点云累积建图，提升表面密度
- **鲁棒性**：对 H 矩阵方法和 Ray-Surface 方法结果做加权融合
- **误差分解**：分别输出横向误差和纵向误差
