import numpy as np
import swanlab
import torch
import os
from tqdm import tqdm

from utils.evaluator import Evaluator  

def train(cfg, env, eval_env, agent, buffer):
    """
    Main training loop for On-Policy MARL algorithms (e.g., IPPO).

    Args:
        cfg (argparse.Namespace): Configuration object containing training parameters.
        env (VectorEnv): Vectorized training environment.
        eval_env (VectorEnv): Environment for evaluation.
        agent (object): The on-policy agent instance (must implement take_action, update, save).
        buffer (ParallelRolloutBuffer): Rollout buffer for storing trajectories.

    Returns:
        train_stats (list): List of dictionaries containing training metrics (steps, reward, length).
        eval_stats (list): List of dictionaries containing evaluation metrics (steps, reward).
    """
    print(f"--> Start On-Policy Training (IPPO) | Seed: {cfg.seed}")
    
    evaluator = Evaluator(eval_env, agent, cfg.device, policy_type='on-policy', seed=getattr(cfg, 'eval_seed', None))

    train_stats = []
    eval_stats = []
    
    total_steps = 0
    i_episode = 0 
    
    best_return = -float('inf')
    
    obs, _ = env.reset()    # obs (B, N, C, H, W) [B:num_envs]
    
    # Init hidden returns (B, N, L, H)
    hidden_state = agent.init_hidden(cfg.num_envs)
    
    episode_returns = np.zeros(cfg.num_envs)
    episode_lengths = np.zeros(cfg.num_envs)
    
    pbar = tqdm(total=cfg.max_steps, desc=f"Seed {cfg.seed}")
    current_loss = 0.0

    model_dir = os.path.join(cfg.run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    while total_steps < cfg.max_steps:
        with torch.no_grad():
            while not buffer.is_full():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
                
                actions, log_probs, values, next_hidden_state = agent.take_action(obs_tensor, hidden_state)
                # actions: (B, N)
                # log_probs: (B, N)
                # values: (B, N)
                # next_hidden_state: (B, N, L, H) [L:Layers]

                next_obs, rewards, dones, truncated, _ = env.step(actions.tolist())
                # rewards, dones, truncated: (B, N)

                mask_bool = torch.from_numpy(dones | truncated).to(cfg.device).bool() # (B, N)
                next_hidden_state[mask_bool] = 0.0

                hidden_np = hidden_state.cpu().numpy()
                
                # Construct State (for MAPPO compatibility or just storage)
                B, N, C, H, W = obs.shape
                state = obs.reshape(B, N*C, H, W)

                buffer.push(
                    obs=obs,
                    state=state,
                    hidden_states=hidden_np,
                    actions=actions.cpu().numpy(),
                    rewards=rewards,
                    dones=dones,
                    log_probs=log_probs.cpu().numpy(),
                    values=values.cpu().numpy()
                )
                
                for i in range(cfg.num_envs):
                    episode_returns[i] += np.sum(rewards[i])
                    episode_lengths[i] += 1
                    
                    if np.any(dones[i]) or np.any(truncated[i]):
                        i_episode += 1
                        
                        if i_episode % getattr(cfg, 'log_freq', 100) == 0:
                            swanlab.log({
                                "Train/Loss": current_loss, 
                                "Train/Reward": episode_returns[i]
                            }, step=total_steps)

                            pbar.write(f" [Train][Seed {cfg.seed}] Steps {total_steps} | Episodes {i_episode}: Train Reward = {episode_returns[i]:.2f}")
                            
                            train_stat = {
                            'steps': total_steps,
                            'reward': episode_returns[i],
                            'episode_length': episode_lengths[i],
                            'seed': cfg.seed
                            }
                            train_stats.append(train_stat)

                        # Evaluation
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

                            # Save Best Model
                            if avg_reward > best_return:
                                best_return = avg_reward
                                best_path = os.path.join(model_dir, "model_best.pth")
                                agent.save(best_path)
                                pbar.write(f"   >>> [New Best] Model Saved: {best_path} (Return: {best_return:.2f})")

                        episode_returns[i] = 0
                        episode_lengths[i] = 0
                
                obs = next_obs
                hidden_state = next_hidden_state
                total_steps += cfg.num_envs
                pbar.update(cfg.num_envs)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
        with torch.no_grad():
            _, _, next_values, _ = agent.take_action(obs_tensor, hidden_state)
        
        hidden_np = hidden_state.cpu().numpy()
        
        buffer.insert_last_step(
            obs=obs,
            state=None,
            hidden_states=hidden_np,
            values=next_values.cpu().numpy(),
            dones=dones 
        )

        stats = agent.update(buffer)
        current_loss = stats['loss_value'] + stats['loss_action']
        buffer.clear()

    pbar.close()
    return train_stats, eval_stats