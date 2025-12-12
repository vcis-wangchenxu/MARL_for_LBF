import os
import torch
import numpy as np
import glob  # [新增] 需要导入 glob

def evaluate(agent, env, n_episodes: int = 5) -> float:
    """评估智能体在环境中的表现。"""
    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden_state = agent.init_hidden(batch_size=1)

        done = False
        episode_return = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            # evaluation=True 关闭 epsilon-greedy
            actions, next_hidden_state = agent.take_action(
                obs_tensor, hidden_state, current_step=0, evaluation=True
            )

            next_obs, rewards, dones, info = env.step(actions)
            episode_return += np.sum(rewards)

            obs = next_obs
            hidden_state = next_hidden_state

            if np.any(dones):
                done = True

        returns.append(episode_return)

    return float(np.mean(returns))


def _find_latest_checkpoint(run_dir: str) -> str:
    """寻找最新的单个模型文件"""
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models dir not found: {models_dir}")

    for name in ["best_model.pth", "last_model.pth"]:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return path

    pths = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pth")
    ]
    if not pths:
        raise FileNotFoundError(f"No .pth found in {models_dir}")

    pths.sort(key=os.path.getmtime, reverse=True)
    return pths[0]


# 查找所有种子的模型
def find_all_seed_checkpoints(run_dir: str) -> list[str]:
    """
    在 run_dir/models 下寻找所有种子的最佳模型。
    文件名模式: best_model_seed*.pth
    """
    models_dir = os.path.join(run_dir, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # 搜索模式: best_model_seed*.pth
    # 例如: best_model_seed1.pth, best_model_seed42.pth
    search_pattern = os.path.join(models_dir, "best_model_seed*.pth")
    model_files = glob.glob(search_pattern)
    
    # 排序，保证输出顺序一致 (如 seed1, seed100, seed42...)
    # 如果想按数字排序可能需要额外的 lambda，但这里字符串排序通常足够
    model_files.sort() 

    # 兼容性处理：如果没找到带 seed 的，尝试找通用的 best_model.pth
    if not model_files:
        fallback = os.path.join(models_dir, "best_model.pth")
        if os.path.exists(fallback):
            model_files = [fallback]

    if not model_files:
        raise FileNotFoundError(f"No valid model files found in {models_dir}")

    return model_files