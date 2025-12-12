import argparse
import os
import torch
import swanlab
import glob
import pickle
import numpy as np

from utils.config import get_config, load_algorithm
from utils.util import set_seed, save_results, plot_learning_curve
from envs.lbf_wrapper import LBFWrapper
from memories.ReplayBuffer import OffPolicyReplayBuffer as ReplayBuffer 
from train import train
from eval import evaluate, _find_latest_checkpoint, find_all_seed_checkpoints

def run_train_single_seed(cfg):
    """
    单个 seed 的训练流程 (底层函数)。
    """
    algo_path = cfg.algo
    algo_name = algo_path.split(".")[-1]

    # 设备与随机种子
    device_str = "cuda" if torch.cuda.is_available() and getattr(cfg, "device", "cpu") == "cuda" else "cpu"
    cfg.device = device_str
    print(f"[{algo_name}] Seed {cfg.seed} | Device: {cfg.device}")

    set_seed(getattr(cfg, "seed", 0))

    # SwanLab 初始化 (如果不需要云端记录，mode="disabled" 或 "local")
    swanlab.init(
        project=getattr(cfg, "project_name", "LBF-MARL-Benchmark"),
        experiment_name=f"{algo_name}_{cfg.env}_seed{cfg.seed}",
        config=vars(cfg),
        logdir=cfg.run_dir,
        mode="cloud", 
    )

    # 创建环境
    env_kwargs = {}
    if hasattr(cfg, "sight"):
        env_kwargs["sight"] = cfg.sight
    env = LBFWrapper(env_id=cfg.env, **env_kwargs)
    eval_env = LBFWrapper(env_id=cfg.env, **env_kwargs)

    # 回放缓冲区
    replay_buffer = ReplayBuffer(
        env_info=env.get_env_info(),
        capacity=getattr(cfg, "buffer_capacity", 50000),
        batch_size=getattr(cfg, "batch_size", 32),
        sequence_length=getattr(cfg, "sequence_length", 8),
        device=cfg.device
    )

    # 算法实例
    AlgoClass = load_algorithm(cfg.algo)
    agent = AlgoClass(env.get_env_info(), cfg)

    # 训练 (返回 train_stats, eval_stats)
    train_stats, eval_stats = train(agent, env, eval_env, cfg, replay_buffer)

    swanlab.finish()
    
    # 清理资源
    env.close()
    eval_env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return algo_name, train_stats, eval_stats

def run_train_multi_seeds(base_cfg, seeds):
    """
    运行单个算法的多个种子训练。
    训练结束后只保存数据，不绘图。
    """
    all_train_data = [] 
    all_eval_data = []  
    
    algo_name = base_cfg.algo.split(".")[-1]

    print(f"\n{'='*60}")
    print(f"Algorithm: {algo_name} | Environment: {base_cfg.env}")
    print(f"Seeds: {seeds}")
    print(f"Results Dir: {base_cfg.run_dir}")
    print(f"{'='*60}")

    for seed in seeds:
        print(f"\n>>> Running Seed: {seed}")
        # 浅拷贝配置并修改 seed
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.seed = seed
        
        # 运行训练
        _, train_stats, eval_stats = run_train_single_seed(cfg)
        
        # 收集数据
        all_train_data.extend(train_stats) # [Dict]
        all_eval_data.extend(eval_stats)

    # === 保存聚合数据 ===
    # 格式: results/{algo}/{env}/{timestamp}/{algo}_train_data.pkl
    train_file = os.path.join(base_cfg.run_dir, f"{algo_name}_train_data.pkl")
    eval_file = os.path.join(base_cfg.run_dir, f"{algo_name}_eval_data.pkl")
    
    save_results(all_train_data, train_file)
    print(f"\n[Saved] Training data saved to: {train_file}")
    
    if all_eval_data:
        save_results(all_eval_data, eval_file)
        print(f"\n[Saved] Evaluation data saved to: {eval_file}")

    print(f"\nAll seeds finished. To visualize, run: python run.py --mode plot --env {base_cfg.env} --run_dir results")

def run_plot(cfg):
    """
    绘图模式：扫描 cfg.run_dir 目录下该环境(env)的所有算法数据，
    并分别绘制【训练曲线】和【评估曲线】。
    """
    target_env = cfg.env
    root_dir = cfg.run_dir  # 使用配置中的 run_dir
    
    if not os.path.exists(root_dir):
        print(f"[Error] Results directory not found: {root_dir}")
        return

    print(f"\n--- Scanning results in '{root_dir}' for environment: {target_env} ---")
    
    # === 定义通用处理函数 ===
    def process_and_plot(data_type, file_suffix, title_suffix, output_suffix, smooth_window):
        """
        data_type: 'training' or 'evaluation' (用于日志)
        file_suffix: '_train_data.pkl' or '_eval_data.pkl'
        title_suffix: 图表标题后缀
        output_suffix: 输出文件名后缀
        smooth_window: 平滑窗口大小
        """
        # 搜索模式: root_dir/<Algo>/<TargetEnv>/*/*{file_suffix}
        # 这里的目录结构通常是: results/IDQN/Foraging-8x8.../timestamp/IDQN_train_data.pkl
        search_pattern = os.path.join(root_dir, "*", target_env, "*", f"*{file_suffix}")
        found_files = glob.glob(search_pattern)
        
        if not found_files:
            print(f"No {data_type} data found matching pattern: {search_pattern}")
            return

        experiments_data = {} # 结构: { "IDQN": [data...], "QMIX": [data...] }
        
        for fpath in found_files:
            # 从文件名提取算法名: IDQN_train_data.pkl -> IDQN
            fname = os.path.basename(fpath)
            algo_name = fname.replace(file_suffix, "")
            
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                
                if algo_name not in experiments_data:
                    experiments_data[algo_name] = []
                
                # 合并数据
                if isinstance(data, list):
                    experiments_data[algo_name].extend(data)
                else:
                    experiments_data[algo_name].append(data)
                    
                print(f"[{data_type.capitalize()}] Loaded {algo_name} from {os.path.basename(os.path.dirname(fpath))} ({len(data)} records)")
                
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        # 调用绘图工具
        if experiments_data:
            # 图片保存在 root_dir 下，例如 results/benchmark_Foraging-8x8..._training.png
            output_file = os.path.join(root_dir, f"benchmark_{target_env}{output_suffix}.png")
            
            plot_learning_curve(
                experiments_data, 
                title=f"Benchmark {title_suffix}: {target_env}",
                output_filename=output_file,
                bin_size=smooth_window
            )
        else:
            print(f"No valid {data_type} data loaded.")

    # 1. 生成训练曲线 (Training Curve)
    process_and_plot(
        data_type="training",
        file_suffix="_train_data.pkl",
        title_suffix="(Training)",
        output_suffix="_training",
        smooth_window=1000  # 训练数据点多，震荡大，平滑窗口设大一点
    )
    
    # 2. 生成评估曲线 (Evaluation Curve)
    process_and_plot(
        data_type="evaluation",
        file_suffix="_eval_data.pkl",
        title_suffix="(Evaluation)",
        output_suffix="_evaluation",
        smooth_window=1     # 评估数据点少，通常不需过度平滑
    )

def run_eval(cfg):
    """
    评估模式：
    1. 如果指定了 --checkpoint，则只评估该模型（单种子）。
    2. 如果只指定了 --run_dir，则自动搜索所有种子的模型并评估（多种子），最后输出平均分。
    """
    
    # === 1. 确定要评估的模型文件列表 ===
    model_files = []
    eval_mode = "unknown"

    if getattr(cfg, "checkpoint", None) is not None:
        # 情况 A: 用户指定了具体模型 -> 单个评估
        print(f"\n[Eval] Mode: Single Checkpoint")
        if not os.path.exists(cfg.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")
        model_files = [cfg.checkpoint]
        eval_mode = "single"
    else:
        # 情况 B: 用户指定了 run_dir -> 自动搜索所有种子
        if not hasattr(cfg, "run_dir") or not cfg.run_dir:
            raise ValueError("Evaluation requires either --checkpoint or --run_dir.")
        
        print(f"\n[Eval] Mode: Multi-Seed (Scanning {cfg.run_dir})")
        try:
            model_files = find_all_seed_checkpoints(cfg.run_dir)
            eval_mode = "multi"
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            return

    print(f"[Eval] Found {len(model_files)} models to evaluate.")

    # === 2. 初始化环境和算法 (只创建一次，复用) ===
    env_kwargs = {}
    if hasattr(cfg, "sight"):
        env_kwargs["sight"] = cfg.sight
    eval_env = LBFWrapper(env_id=cfg.env, **env_kwargs)
    
    AlgoClass = load_algorithm(cfg.algo)
    # 注意：这里使用 cfg 初始化，假设所有种子的超参数是一致的
    agent = AlgoClass(eval_env.get_env_info(), cfg) 

    # === 3. 循环加载模型并评估 ===
    n_episodes = getattr(cfg, "eval_episodes", 5)
    all_returns = []

    print(f"{'-'*60}")
    for model_path in model_files:
        # 提取 Seed 名称用于显示 (例如从 best_model_seed42.pth 提取 "seed42")
        fname = os.path.basename(model_path)
        seed_label = fname.replace("best_model_", "").replace(".pth", "")
        
        print(f"Loading {seed_label:<15} ...", end=" ", flush=True)
        
        try:
            agent.load(model_path)
            # 运行 n_episodes 次评估
            mean_ret = evaluate(agent, eval_env, n_episodes=n_episodes)
            all_returns.append(mean_ret)
            print(f"Mean Return: {mean_ret:.2f}")
        except Exception as e:
            print(f"[Failed] {e}")
    print(f"{'-'*60}")

    # === 4. 汇总统计 (仅在多种子模式下有意义) ===
    if all_returns:
        avg_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        
        if eval_mode == "multi":
            print(f"\n[Summary] Evaluation over {len(model_files)} seeds:")
            print(f"  > Average Return: {avg_return:.4f} ± {std_return:.4f}")
            print(f"  > Max Return:     {np.max(all_returns):.4f}")
            print(f"  > Min Return:     {np.min(all_returns):.4f}")
        else:
            print(f"\n[Summary] Single model return: {avg_return:.4f}")
            
    eval_env.close()

def main():
    parser = argparse.ArgumentParser(description="Run MARL for LB-Foraging")
    parser.add_argument("--mode", type=str, default="train-multi",
                        choices=["train-multi", "eval", "plot"],
                        help="运行模式: train-multi (多种子训练), eval (评估), plot (绘图)")
    
    # 可以在命令行指定 run_dir，方便 plot 模式使用
    parser.add_argument("--run_dir", type=str, default="results",
                        help="结果存储根目录。在 plot 模式下，程序会在此目录下搜索 .pkl 文件")

    # 解析已知参数，剩余传给 config
    args, unknown = parser.parse_known_args()
    
    # 模拟 sys.argv 给 config.py 解析
    import sys
    sys.argv = [sys.argv[0]] + unknown

    # 根据 mode 决定是否为训练模式 (影响 run_dir 的生成逻辑)
    is_train = args.mode == "train-multi"
    cfg = get_config(is_train=is_train)

    # 手动注入 run_dir 到 cfg 中
    # 因为 get_config 在 is_train=False 时可能不会从命令行读取 run_dir，
    # 或者从 config.yaml 读取的 run_dir 是旧的路径，我们需要优先使用命令行指定的路径。
    if not is_train:
        if args.run_dir != "results": # 如果用户指定了非默认路径
            cfg.run_dir = args.run_dir
        elif not hasattr(cfg, 'run_dir'): # 如果配置文件里也没有
            cfg.run_dir = "results"

    if args.mode == "train-multi":
        seeds = getattr(cfg, "seeds", [1, 42, 100])
        run_train_multi_seeds(cfg, seeds)

    elif args.mode == "plot":
        run_plot(cfg)

    elif args.mode == "eval":
        run_eval(cfg)

if __name__ == "__main__":
    main()