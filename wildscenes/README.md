# DISTANCE workspace

源码和运行说明位于 [`DISTANCE/`](DISTANCE/README.md)。本仓库以当前目录为根，目录约定如下：

- `DISTANCE/`：评测、训练和部署源码
- `wildscenes/`：MMSeg/MMDetection 配置与数据集注册代码
- `environment.yml`：Conda 环境说明
- `data/`：本地数据集（已由 Git 忽略）
- `pretrained_models`：本地预训练模型目录或软链接（已由 Git 忽略）

运行程序时进入源码目录：

```bash
conda activate distance
cd DISTANCE
python -m distance.run_sequence --config configs/evaluation.yaml --list-tasks
```
