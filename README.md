# MARL for Level-Based Foraging (LBF) 🍃

> A Multi-Agent Reinforcement Learning experimental platform based on the [Level-Based Foraging (LBF)](https://github.com/semitable/lb-foraging) environment. This project implements **VDN (Value Decomposition Networks)** and **IPPO (Independent PPO)** algorithms, supporting parameter sharing and independent networks, and integrates RNN (GRU) to handle partial observability.

## ✨ Highlights

- **Algorithms**: Includes Off-Policy **VDN** and On-Policy **IPPO**.
- **Network Architecture**: Supports **CNN + GRU** architecture, effectively handling LBF's grid image input and partial observability issues.
- **Parallel Training**: Implements multi-environment parallel sampling based on `AsyncVectorEnv`, significantly improving training efficiency.
- **Unified Management**: Manages training, evaluation, and plotting through `run.py`, supporting automatic execution of multiple random seeds.
- **Experiment Tracking**: Integrates [SwanLab](https://swanlab.cn) for visual experiment monitoring.

## 📂 Project Structure

```text
marl_for_lbf_pre/
├── algorithms/         # Core algorithm implementations (ippo.py, vdn.py)
├── configs/            # YAML configuration files (ippo_lbf.yaml, vdn_lbf.yaml)
├── envs/               # Environment wrappers (lbf_wrapper.py, multi_envs.py)
├── memories/           # Experience replay buffers (ReplayBuffer, RolloutBuffer)
├── networks/           # Neural network models (ActorCritic, DRQN)
├── results/            # Directory for saving training results
├── utils/              # Utility functions (config, evaluator, logger)
├── run.py              # Main entry point (Train/Eval/Plot)
├── train_offpolicy.py  # Off-Policy training loop
└── train_onpolicy.py   # On-Policy training loop
```

## 📦 Installation

Recommended environment: Python 3.8+

```bash
# 1. Install core dependencies
pip install torch numpy matplotlib pyyaml tqdm swanlab

# 2. Install LBF environment
pip install lbforaging
```

## 🚀 Quick Start

### 1. Training

Use `run.py` to start training. The script will automatically read the configuration file and run the specified random seeds sequentially.

**Train VDN (Off-Policy):**
```bash
python run.py --config configs/vdn_lbf.yaml
```

**Train IPPO (On-Policy):**
```bash
python run.py --config configs/ippo_lbf.yaml
```

*   **Parameters**:
    *   `--config`: Path to the YAML configuration file (Required).
    *   `--run_dir`: Root directory for saving results (Default: `results`).

### 2. Evaluation

Load a trained model for testing. Supports automatically finding the best model or specifying a model path.

**Basic Mode** (Test on the training environment):
```bash
# Automatically load results/Foraging-8x8-2p-3f-v2/VDN/seed_1/models/model_best.pth
python run.py --mode eval --config configs/vdn_lbf.yaml --env Foraging-8x8-2p-3f-v2 --seed 1
```

**Generalization Test Mode** (Test on different environments or seeds):
```bash
# Load model from seed 1, but test on environment with seed 2024
python run.py --mode eval --config configs/vdn_lbf.yaml --env Foraging-8x8-2p-3f-v2 --seed 1 --eval_seed 2024

# Load model trained on 8x8 environment, test on 10x10 environment (Requires specifying checkpoint)
python run.py --mode eval --config configs/vdn_lbf.yaml --checkpoint results/Foraging-8x8.../model_best.pth --eval_env Foraging-10x10-2p-3f-v2
```

**Specific Model Path Mode**:
```bash
python run.py --mode eval --config configs/vdn_lbf.yaml --checkpoint results/Foraging-8x8-2p-3f-v2/VDN/seed_1/models/model_best.pth
```

### 3. Plotting

Automatically scan all experiment data in the `results` folder and plot training and evaluation curves.

```bash
python run.py --mode plot --env Foraging-8x8-2p-3f-v2 --run_dir results
```
After running, the generated images (e.g., `benchmark_Foraging-8x8-2p-3f-v2_eval.png`) will be saved in the corresponding environment results directory.

## ⚙️ Configuration

Configuration files are located in the `configs/` directory. Key parameters are explained below (using `ippo_lbf.yaml` as an example):

```yaml
# --- Environment Settings ---
env: Foraging-8x8-2p-3f-v2   # Environment ID
sight: 2                     # Sight range (2 indicates partial observability)
eval_env: Foraging-10x10-2p-3f-v2 # (Optional) Environment ID for evaluation during training. Defaults to same as env if not specified.
eval_seed: 2024              # (Optional) Fixed random seed for evaluation. Defaults to seed + 10000 if not specified.

# --- Algorithm Settings ---
algo: algorithms.ippo.IPPO   # Algorithm class path
policy_type: on-policy       # Policy type: on-policy or off-policy
share_parameters: True       # Whether to share network parameters

# --- Training Parameters ---
seeds: [1, 42, 123]          # List of random seeds to run
num_envs: 8                  # Number of parallel environments
max_steps: 1000000           # Total training steps
rnn_hidden_dim: 64           # RNN hidden layer dimension

# --- PPO Specific Parameters ---
ppo_epochs: 4                # PPO update epochs
clip_param: 0.2              # PPO clipping parameter
entropy_coef: 0.01           # Entropy regularization coefficient
```

## 🌍 Environment Naming

LBF Environment ID Format: `Foraging{Sight}-{Size}x{Size}-{Players}p-{Foods}f{Suffix}-v3`

*   **Sight**: `Foraging-2s-...` means sight radius is 2; `Foraging-...` means full map visibility.
*   **Size**: Map dimensions, e.g., `8x8`.
*   **Players**: Number of players, e.g., `2p`.
*   **Foods**: Number of food items, e.g., `3f`.
*   **Suffix**: `-coop` (Forced cooperation), `-grid` (Returns grid image observation).

For more details, please refer to [LBF_use.md](LBF_use.md).

## 📚 Algorithm List

| Algorithm | Status | Reference |
| :--- | :---: | :--- |
| **VDN** | ✅ | [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) |
| **IPPO** | ✅ | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| **QMIX** | ✅ | [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) |
| **MAPPO** | ✅ | [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) |
| **other** | ❌ | [pass](https://arxiv.org/abs/2103.01955) |
