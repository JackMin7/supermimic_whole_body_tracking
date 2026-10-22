# 运行说明

## 运行前提

需要 Linux、NVIDIA GPU 和已经可运行的 G1/mjlab 项目。本仓库**没有附带机器人网格、MJCF、训练动作数据、第三方源码或模型权重**。安装 Python 依赖本身不足以完成复现。

外部项目需要提供 `src/assets/robots`、相应 G1 资产与动作 NPZ。`vendor/mjlab` 如存在将优先使用。测试环境版本见 [requirements-tested.txt](../requirements-tested.txt)，具体适配边界见 [复现说明](REPRODUCIBILITY.md)。

```bash
export MBRL_PROJECT_ROOT=/path/to/your/configured-g1-project
export MUJOCO_GL=egl
python -m unittest discover -s tests -v
python validate_environment.py --motion /path/to/standing.npz --output validation.json
```

首次对照，各训练 10 分钟：

```bash
python run_comparison.py --motion /path/to/standing.npz \
  --seconds 600 --output runs/standing_10min
```

继续各训练 50 分钟：

```bash
python run_comparison.py --motion /path/to/standing.npz \
  --seconds 3000 --resume-from runs/standing_10min --output runs/standing_1h
```

长时评测（只有静态参考允许扩展片段长度）：

```bash
python evaluate.py --checkpoint runs/standing_1h/mbrl/final.pt \
  --motion /path/to/standing.npz --seconds 60 --output runs/standing_1h/mbrl/long60.json
```
