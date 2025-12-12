import logging
import os
import sys

def get_logger(run_dir):
    """
    创建一个 Logger，同时输出到控制台和 run_dir/logs/train.log
    """
    logger = logging.getLogger("MARL_Trainer")
    logger.setLevel(logging.INFO)
    
    # 清除之前的 handlers (防止重复打印)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 Handler
    log_file = os.path.join(run_dir, "logs", "train.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # 控制台只打印消息，保持简洁
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger