from typing import List, Tuple, Dict
import os
import random
import torch
import swanlab
import numpy as np

from eval import evaluate

def train(agent, env, eval_env, config, replay_buffer):
    print(f"--- Start Training on {config.env} (Seed: {config.seed})---")
    # 准备保存路径
    save_dir = os.path.join(config.run_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_seed{config.seed}.pth")
    last_model_path = os.path.join(save_dir, f"last_model_seed{config.seed}.pth")

    best_eval_reward = -float('inf')

    # 统计数据
    train_stats = []
    eval_stats = []

    max_steps = getattr(config, 'max_steps', 200000) 
    eval_freq = getattr(config, 'eval_freq', 20)
    seq_len = getattr(config, 'sequence_length', 8)
                
    total_steps = 0
    i_episode = 0

    current_seed = config.seed
    # Warmup 阶段：随机采样动作填充经验回放缓冲区
    warmup_steps = getattr(config, 'learning_starts', 2000)
    if warmup_steps > 0:
        print(f"Start warming up for {warmup_steps} steps...")
        obs, info = env.reset(seed=current_seed)    
        current_seed = None

        state = obs.reshape(-1, *obs.shape[2:])

        while total_steps < warmup_steps:
            # 随机动作采样
            actions = [random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
            next_obs, rewards, dones, info = env.step(actions)
            next_state = next_obs.reshape(-1, *next_obs.shape[2:])

            replay_buffer.push(
                obs, state, np.array(actions), np.array(rewards), np.array(dones),
                next_obs, next_state, info
            )

            obs = next_obs
            state = next_state
            total_steps += 1

            if np.any(dones):
                obs, info = env.reset(seed=current_seed)
                state = obs.reshape(-1, *obs.shape[2:])
        
        print(f"Warming up completed. Total steps: {total_steps}")

    # 主训练循环
    while total_steps < max_steps:
        i_episode += 1
        obs, info = env.reset(seed=current_seed)    
        state = obs.reshape(-1, *obs.shape[2:])

        hidden_state = agent.init_hidden(batch_size=1)

        done = False
        episode_return = 0.0
        episode_steps = 0

        while not done: 
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)  # (1, N, obs_dim)
            actions, next_hidden_state = agent.take_action(obs_tensor, hidden_state, total_steps)

            next_obs, rewards, dones, info = env.step(actions)
            next_state = next_obs.reshape(-1, *next_obs.shape[2:])

            done = np.any(dones) 

            replay_buffer.push(
                obs, state, np.array(actions), np.array(rewards), np.array(dones),
                next_obs, next_state, info
            )

            obs = next_obs
            state = next_state
            hidden_state = next_hidden_state

            episode_return += np.sum(rewards)
            total_steps += 1
            episode_steps += 1

            if len(replay_buffer) > config.batch_size:
                batch = replay_buffer.sample()
                
                loss = agent.update(batch)

                if total_steps % 100 == 0:
                    swanlab.log({"Train/Loss": loss}, step=total_steps)
                    swanlab.log({"Train/Epsilon": agent.epsilon}, step=total_steps)

            if total_steps >= max_steps:
                break
        # 记录训练数据
        train_stat = {'steps': total_steps, 'reward': episode_return, 'seed': config.seed}
        train_stats.append(train_stat) # [Dict]
        # Episode 结束日志记录
        swanlab.log({
            "Return/Episode": episode_return, 
            "Episode_Length": episode_steps,
            "Episode_Count": i_episode
        }, step=total_steps)

        print(f"Step {total_steps}/{max_steps} | Ep {i_episode}: Return = {episode_return:.2f}, Steps = {episode_steps}, Epsilon = {agent.epsilon:.3f}")

        # 评估
        if i_episode % eval_freq == 0:
            print(f"  [Eval] Starting evaluation at step {total_steps}...")
            eval_return = evaluate(agent, eval_env, n_episodes=5)
            
            eval_stat = {'steps': total_steps, 'reward': eval_return, 'seed': config.seed}
            eval_stats.append(eval_stat)

            swanlab.log({"Eval/Return": eval_return}, step=total_steps)
            print(f"  [Eval] Step {total_steps}: Average Return = {eval_return:.2f}")
            
            # 保存最佳模型
            if eval_return >= best_eval_reward:
                best_eval_reward = eval_return
                agent.save(best_model_path)
                print(f"  [Save] New best model saved to {best_model_path}")

    agent.save(last_model_path)
    env.close()
    eval_env.close()
    print(f"Training completed. Last model saved to {last_model_path}")

    return train_stats, eval_stats

