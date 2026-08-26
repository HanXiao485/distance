_base_ = [
    '../../../training/scene204/mmseg_configs/train_finetune.py',
]

# ── 从头训练（数据集够时使用，去掉 GOOSE 权重依赖） ──────────────────────────
# 与 train_finetune.py 唯一的区别：不加载预训练权重 + 训练更长
load_from = None

train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000, val_interval=2000)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=0,
        end=50000,
        by_epoch=False,
    )
]

work_dir = '/root/distance/WildScenes/work_dirs/scene204_deeplabv3_scratch'
