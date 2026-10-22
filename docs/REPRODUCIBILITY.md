# 复现与实现差异

## 方法

先用当前策略采集刚体仿真转移，再通过 8 步自回归窗口拟合动力学模型，最后冻结模型参数，通过 32 步想象轨迹反向传播更新策略。价值网络提供末端 bootstrap，并在固定 lambda-return 目标上优化。冻结模型权重不会切断动作和状态的梯度。

默认 small 配置：动力学 MLP 2048×3；encoder/decoder 隐藏层 512/256/128；critic 256/128；模型/策略 batch 各 512，256 环境。paper 配置扩大 encoder/decoder 和批量，仅代表对应主要网络参数，并非完整复现认证。学习率初始 2e-4，250000 轮余弦衰减到 0.03 倍。

## 明确的项目适配

使用项目 G1 资产、力矩转速曲线及摩擦模型。物理状态为 qpos/qvel/前一动作；动力学采用标准化残差输出及四元数归一化。本体观测解析计算，奖励与终止头学习预测。损失做标准化和维度平均，不能称与论文未详列的数值处理完全相同。

保留小策略网络。采集间隔默认为 4 步；论文 UTD 的计数口径未完整解析，没有强行宣称匹配。回报采样使用 1 秒 bin 的 50/40/10 混合，估计器为 128 步截断 bootstrap 和 EMA；当前只支持一个运动文件，不是大规模多文件动作库。

可选 `--max-state-z` 是额外的逐轨迹保护，默认关闭。

## 外部依赖

测试平台为 Linux / CUDA，Torch 2.7.0+cu128、MuJoCo 3.11.0、mjlab 1.2.0、RSL-RL 5.0.1。实际仿真使用项目中的 vendor mjlab 快照，因此相同 pip 版本不能保证完全一致。需要显式提供 `MBRL_PROJECT_ROOT`，不能只运行 pip install 就宣称环境已就绪。

运行前先用 `validate_environment.py` 验证关节顺序、参考动作的身体位置，以及解析重力/角速度与仿真读数的一致性。模型输入固定为自由基座、29 自由度 G1；不兼容时应报错，而不是静默套用索引。

现有训练权重采用 checkpoint `format_version=3`。这是内部数据格式编号，与公开方法名无关。改包名不改变 tensor state_dict 的键；加载旧格式权重应使用相同结构并传入新的 `--motion` 路径。仅加载可信来源的 checkpoint。

## 验证

`python -m unittest discover -s tests -v` 覆盖梯度链、lambda-return、回放窗口、reset掩码、目标编码、解析观测、逐轨迹保护及重置采样。此前训练实现通过环境验证和完整一小时对照。发布重组后的验证结果记录在 `RELEASE_CHECKS.txt`；没有因改名而重新训练一小时。

## 图表与动图重建

运行 `python tools/plot_results.py` 生成中文图表。Linux 安装 Noto CJK 字体，或将环境变量 `MBRL_CJK_FONT` 指向本机中文字体文件。运行 `python tools/make_gifs.py` 将 MP4 转成正常速度 GIF，需要系统提供 ffmpeg。
