import logging
import os
import sys

def get_logger(run_dir):
    """
    Create a Logger that outputs to both the console and run_dir/logs/train.log
    """
    logger = logging.getLogger("MARL_Trainer")
    logger.setLevel(logging.INFO)
    
    # Clear previous handlers (to prevent duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    log_file = os.path.join(run_dir, "logs", "train.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # Console only prints messages, keep it simple
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger