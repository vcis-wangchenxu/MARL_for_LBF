# MARL for LB-Foraging 🍃

> 一个基于 [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) 环境的多智能体强化学习实验平台，用来快速对比和可视化各类 MARL 算法的表现。

## 项目亮点 ✨

- **专注多智能体协作场景**：基于 LBF 环境，支持多智能体协同采集、资源竞争等典型任务  
- **清晰的算法与网络结构划分**：`algorithms/`、`networks/`、`memories/` 分层实现，方便扩展新算法  
- **一键训练与评估**：通过 `train.py` / `eval.py` 配合 YAML 配置快速启动实验  
- **完备的实验记录**：`results/` 中自动按时间戳保存配置、日志与模型，方便复现实验结果  
- **便于二次开发**：环境封装、配置解析、日志工具都进行了模块化封装

---

## 安装与依赖 📦

确保你的 Python 环境（建议 Python 3.8+）已安装以下依赖：

```bash
# 核心依赖
pip install torch numpy gym

# 环境依赖
pip install lb-foraging

# 实验记录与可视化
pip install swanlab matplotlib pyyaml
```

---

## 快速开始 🚀

本项目使用 `run.py` 作为统一入口，支持 **多种子并行训练**、**全量模型评估** 和 **一键结果绘图**。

### 1. 训练 (Training)

使用 `train-multi` 模式运行实验，程序会依次运行配置文件中定义的所有随机种子：

```bash
# 默认使用 configs/idqn_lbf.yaml 配置
python run.py --mode train-multi --config configs/idqn_lbf.yaml
```

- **配置**：默认读取 `configs/idqn_lbf.yaml`，你也可以创建自己的 yaml 配置文件。修改 configs/idqn_lbf.yaml 中的 seeds 列表可控制运行次数（如 [1, 42, 100]）。
- **输出**：结果会自动保存在 results/算法/环境/时间戳/ 目录下（例如 results/IDQN/Foraging-8x8.../20251212-101841/）。
- **日志**：训练过程会使用 [SwanLab](https://swanlab.cn) 在线/本地记录日志。

### 2. 评估 (Evaluation)

支持两种评估模式：批量评估所有种子（推荐）和 单模型评估。

方式 A：批量评估 -> 指定训练生成的 run_dir，脚本会自动寻找该目录下所有随机种子的最佳模型，依次评估并计算均值/标准差：
```bash
# 将 YOUR_TIMESTAMP_DIR 替换为你实际生成的时间戳文件夹名
python run.py --mode eval --config configs/idqn_lbf.yaml --run_dir results/IDQN/Foraging-8x8-2p-2f-v3/YOUR_TIMESTAMP_DIR
```

方式 B：单模型评估 -> 手动指定某个具体的 .pth 模型文件：

```bash
python run.py --mode eval --config configs/idqn_lbf.yaml --checkpoint results/.../models/best_model_seed42.pth
```

### 3. 绘图 (Plotting)

对训练过程中保存的 `.pkl` 数据进行可视化（生成学习曲线）：

```bash
# 默认扫描 results 目录下该环境的所有数据
# 程序会自动聚合不同时间戳、不同算法（如 IDQN vs QMIX）的数据生成对比图
python run.py --mode plot --run_dir results
```

---

## 配置说明 ⚙️

配置文件位于 `configs/` 目录下（如 `idqn_lbf.yaml`），关键参数如下：

```yaml
# 基础设置
algo: algorithms.idqn.IDQN       # 算法入口类
env: Foraging-8x8-2p-2f-v3       # LBF 环境 ID
seeds: [1, 42, 100]              # 训练使用的随机种子列表

# 训练参数
max_steps: 200000                # 总训练步数
eval_freq: 20                    # 评估频率 (每隔多少个 Episode)
batch_size: 32                   # 批次大小
hidden_dim: 64                   # 网络隐藏层维度

# 探索策略 (Epsilon-Greedy)
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 20000
```

---

## 多智能体算法进度表 ✅

> 当你完成一个算法的实现和基本验证后，就把对应算法打上对勾，形成自己的 MARL 学习/研究路线图。

| 算法类别 | 算法名称 | 状态 |
| :------: | :------: | :--: |
| 值函数法 | IDQN (Independent DQN) | ✅ |
| 值函数法 | VDN (Value Decomposition Networks) | ⬜ |
| 值函数法 | QMIX | ⬜ |
| 策略梯度 | COMA | ⬜ |
| 策略梯度 | Multi-Agent PPO (MAPPO) | ⬜ |
| 策略梯度 | MADDPG | ⬜ |
| 模型自由 | IQL (Independent Q-Learning) | ⬜ |
| 模型自由 | IAC (Independent Actor-Critic) | ⬜ |
| 其它 | Mean-Field MARL | ⬜ |
| 其它 | Centralized Critic Variants | ⬜ |

> 说明：目前仓库中已经包含 `IDQN`（`algorithms/idqn.py` + `networks/DRQN.py` 等），因此该项已标记为 ✅。  
> 其他算法可以在现有代码结构基础上逐步补齐，并同步更新此表。

---

## 目录结构 🧭

项目的主要结构如下：

```text
marl_for_lbf/
├── run.py                 # 项目统一入口（训练/评估/绘图）
├── train.py               # 单次训练逻辑实现
├── eval.py                # 评估逻辑实现
├── configs/               # 配置文件目录
│   └── idqn_lbf.yaml
├── algorithms/            # MARL 算法实现
│   └── idqn.py
├── networks/              # 神经网络模型
│   └── DRQN.py
├── memories/              # 经验回放缓冲区
│   └── ReplayBuffer.py
├── envs/                  # 环境封装
│   └── lbf_wrapper.py
├── utils/                 # 工具函数（日志、配置、绘图等）
└── results/               # 实验结果存储
```

---

## 如何扩展新的 MARL 算法 🧩

基于当前项目结构，你可以按如下步骤添加新的算法（例如 `VDN`、`QMIX` 等）：

1. **在 `algorithms/` 中新建文件**  
	- 如 `vdn.py` / `qmixin.py`，实现对应算法的训练逻辑和损失计算。
2. **在 `networks/` 中添加/复用网络结构**  
	- 例如集中式或分布式 Q 网络、混合网络等。
3. **在 `configs/` 中新建 YAML 配置**  
	- 如 `vdn_lbf.yaml`，指明使用的算法类、网络结构、超参数等。
4. **在 `train.py` / `eval.py` 中挂接新算法**  
	- 根据配置选择不同算法/网络。
5. **更新 README 中的进度表**  
	- 在上面的“多智能体算法进度表”中为对应算法打上 ✅。

通过不断填满这张表，你就可以把这个仓库打造成一个属于自己的 **“多智能体强化学习算法博物馆”**。

---

## 参考与致谢 🙏

- 环境基于：**lb-foraging**  
  - GitHub: `https://github.com/semitable/lb-foraging`
- 论文与资料：  
  - Cooperative Multi-Agent Reinforcement Learning in Sequential Social Dilemmas  
  - Value-Decomposition Networks For Cooperative Multi-Agent Learning  
  - QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning  
  - 等等（可根据你的阅读列表自行补充）

---

## TODO & 未来计划 🧭

- [ ] 完成 VDN / QMIX 等值分解类算法  
- [ ] 引入 MAPPO / MADDPG 等策略梯度方法  
- [ ] 添加训练过程 GIF / 视频展示智能体协作  
- [ ] 支持更多地图规模与任务变体  
- [ ] 写一篇系统的实验报告或博客，整理不同算法在 LBF 上的表现  

> 欢迎 Star、Fork 或提 Issue，一起把这个多智能体强化学习小仓库打磨得更好！
