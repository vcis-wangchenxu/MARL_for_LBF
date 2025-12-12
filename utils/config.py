import yaml
import argparse
import os
import datetime
import importlib
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """构建统一的命令行解析器。

    约定：
    - 训练：python train.py --config configs/idqn_lbf.yaml
    - 评估：python eval.py  --config path/to/config.yaml --checkpoint path/to/best_model.pth

    兼容旧参数：--algo / --env / --yaml
    """

    parser = argparse.ArgumentParser(description="MARL for LB-Foraging")

    # 推荐用法
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径 (推荐)" )
    
    # 旧版/底层参数（可被 YAML 覆盖）
    parser.add_argument("--algo", type=str, default=None,
                        help="Algorithm class path, e.g. algorithms.idqn.IDQNAgent")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment ID, e.g. Foraging-8x8-2p-2f-v3")
    
    # 评估相关参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="评估时使用的模型路径 (.pth)。若不提供，则尝试在 run_dir 下自动查找最新模型。")

    return parser


def _load_yaml_config(path: Optional[str]) -> dict:
    """从 YAML 文件加载配置，如果路径为空或不存在则返回空 dict。"""
    if path is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(is_train: bool = True) -> argparse.Namespace:
    """统一获取配置。

    参数：
        is_train: True 表示训练模式，会为本轮实验生成新的 run_dir；
                  False 表示评估模式，不再生成新的 run_dir，而是从已有 config.yaml 中读取。
    """

    parser = build_parser()
    args = parser.parse_args()

    # 1) 先用 --config
    yaml_path = args.config
    config = _load_yaml_config(yaml_path)

    # 2) 命令行参数覆盖 YAML 配置（None 不覆盖）
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # 重要字段检查
    if "algo" not in config or config["algo"] is None:
        raise ValueError("Missing 'algo' field in configuration. Please specify algorithm class path in YAML or command line (e.g. algorithms.idqn.IDQNAgent).")
    if "env" not in config or config["env"] is None:
        raise ValueError("Missing 'env' field in configuration. Please specify environment ID in YAML or command line.")

    # 3) 训练模式：创建新的 run_dir，并把最终配置写入该目录
    if is_train:
        algo_name = config["algo"].split(".")[-1]
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = f"results/{algo_name}/{config['env']}/{curr_time}"

        config["run_dir"] = run_dir
        os.makedirs(f"{run_dir}/models", exist_ok=True)
        os.makedirs(f"{run_dir}/logs", exist_ok=True)

        with open(f"{run_dir}/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

    # 评估模式：通常应从已有 run_dir/config.yaml 读取，这部分逻辑
    # 可以在 eval.py 中根据 --checkpoint / --config 的约定进行补充。

    return argparse.Namespace(**config)


def load_algorithm(class_path: str):
    """动态加载算法类，例如 algorithms.idqn.IDQNAgent。"""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load algorithm: {class_path}. Error: {e}")


