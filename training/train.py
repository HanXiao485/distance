"""
train.py — GOOSE 训练入口（单卡 / 多卡分布式通用）

基于 mmseg 官方 tools/train.py，唯一区别：在 build runner 之前先
import wildscenes.mmseg_wildscenes.dataset.goose_category，
以触发 GooseCategoryDataset / RemapLabel 注册到 mmseg registry
（这两个类不在 mmseg 官方包里，配置文件用字符串 type= 引用，
必须先 import 才能被 Registry.build() 找到）。

用法（单卡，与之前一致）:
    python train.py \
        wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --work-dir work_dirs/goose_category_ddrnet

用法（双卡分布式）:
    torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 \
        train.py \
        wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --launcher pytorch \
        --work-dir work_dirs/goose_category_ddrnet_dist
"""
import argparse
import logging
import os
import os.path as osp
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401  registers GooseCategoryDataset + RemapLabel

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    cfg.resume = args.resume

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
