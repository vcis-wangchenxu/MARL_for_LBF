import numpy as np
import swanlab
import torch
import os
from tqdm import tqdm

from utils.evaluator import Evaluator 

def train(cfg, env, eval_env, agent, buffer):
    """
    Main training loop for Off-Policy MARL algorithms (e.g., VDN, QMIX).

    Args:
        cfg (argparse.Namespace): Configuration object containing training parameters.
        env (VectorEnv): Vectorized training environment.
        eval_env (VectorEnv): Environment for evaluation.
        agent (object): The off-policy agent instance (must implement take_action, update, save).
        buffer (ReplayBuffer): Experience replay buffer.

    Returns:
        train_stats (list): List of dictionaries containing training metrics (steps, reward, length).
        eval_stats (list): List of dictionaries containing evaluation metrics (steps, reward).
    """
    print(f"--> Start Off-Policy Training | Seed: {cfg.seed}")

    evaluator = Evaluator(eval_env, agent, cfg.device, policy_type='off-policy', seed=getattr(cfg, 'eval_seed', None))

    train_stats = []
    eval_stats = []

    total_steps = 0
    i_episode = 0

    best_return  = -float('inf')

    obs, _ = env.reset()  # obs: (num_envs, N, C, H, W)

    # Using agent's init_hidden (B, N, L, H) [B:num_envs, L:Layers]
    hidden_state = agent.init_hidden(cfg.num_envs)
    
    episode_returns = np.zeros(cfg.num_envs)
    episode_lengths = np.zeros(cfg.num_envs)
    
    pbar = tqdm(total=cfg.max_steps, desc=f"Seed {cfg.seed}")
    current_loss = 0.0

    model_dir = os.path.join(cfg.run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    while total_steps < cfg.max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
        actions, next_hidden_state = agent.take_action(obs_tensor, hidden_state, current_step=total_steps)

        next_obs, rewards, dones, truncated, infos = env.step(actions.tolist())
        # rewards: (num_envs, N)
        # dones: (num_envs, N)
        # truncated: (num_envs, N)

        current_hidden_np = hidden_state.cpu().numpy()
        
        # Construct State if needed (for QMIX)
        # LBFWrapper defines state_shape but doesn't return state in step()
        # We construct it by flattening obs: (num_envs, N, C, H, W) -> (num_envs, N*C, H, W)
        state = None
        next_state = None
    
        # obs: (num_envs, N, C, H, W)
        B, N, C, H, W = obs.shape
        state = obs.reshape(B, N*C, H, W)
        next_state = next_obs.reshape(B, N*C, H, W)

        # Push BATCH data to buffer outside the loop
        buffer.push(
            obs=obs,
            hidden=current_hidden_np,
            state=state,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs,
            next_state=next_state
        )

        for i in range(cfg.num_envs):
            episode_returns[i] += np.sum(rewards[i]) # Sum over agents 
            episode_lengths[i] += 1                  # Increment episode length
            
            done_bool = np.any(dones[i]) or np.any(truncated[i])
            
            if done_bool:
                i_episode += 1
                
                if i_episode % getattr(cfg, 'log_freq', 100) == 0:
                    swanlab.log({
                        "Train/Loss": current_loss,
                        "Train/Epsilon": agent.epsilon,
                        "Train/Reward": episode_returns[i],
                    }, step=total_steps)

                    pbar.write(f" [Train][Seed {cfg.seed}] Steps {total_steps} | Episodes {i_episode}: Train Reward = {episode_returns[i]:.2f}")

                    train_stat = {
                    'steps': total_steps,
                    'reward': episode_returns[i],
                    'episode_length': episode_lengths[i],
                    'seed': cfg.seed
                    }
                    train_stats.append(train_stat)

                if i_episode % getattr(cfg, 'eval_freq', 1000) == 0:
                    avg_reward = evaluator.evaluate(n_episodes=5, seed=cfg.seed)
                    
                    swanlab.log({"Eval/Return": avg_reward}, step=total_steps)
                    pbar.write(f" [Eval][Seed {cfg.seed}] Steps {total_steps} | Episodes {i_episode}: Eval Reward = {avg_reward:.2f}")
                    
                    eval_stat = {
                        'steps': total_steps,
                        'reward': avg_reward,
                        'seed': cfg.seed
                    }
                    eval_stats.append(eval_stat)

                    # Save Latest Model
                    latest_path = os.path.join(model_dir, "model_latest.pth")
                    agent.save(latest_path)

                    if avg_reward > best_return:
                        best_return = avg_reward
                        best_path = os.path.join(model_dir, "model_best.pth")
                        agent.save(best_path)
                        pbar.write(f"   >>> [New Best] Model Saved: {best_path} (Return: {best_return:.2f})")

                episode_returns[i] = 0
                episode_lengths[i] = 0

        obs = next_obs
        hidden_state = next_hidden_state.detach()    # (B, N, L, H)
        
        env_dones = np.any(dones, axis=1) | np.any(truncated, axis=1)    # (num_envs)
        if np.any(env_dones):
            # Mask hidden state (B, N, L, H) with (B,) [B:num_envs]
            hidden_state[env_dones] = 0.0

        total_steps += cfg.num_envs
        pbar.update(cfg.num_envs)

        if len(buffer) >= cfg.batch_size and total_steps > getattr(cfg, 'warmup_steps', 1000):
            batch = buffer.sample()
            if batch is not None:
                current_loss = agent.update(batch)

    pbar.close()
    return train_stats, eval_stats