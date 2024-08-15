import csv
import logging
import os
import random
import configparser
import shutil

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

from run import LOGGER


def create_base_save_folder(dataset_name):
    base_save_dir = "results"
    os.makedirs(base_save_dir, exist_ok=True)
    save_path = os.path.join(base_save_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def create_exp_save_folder(base_save_path, times_str):
    exp_save_path = os.path.join(base_save_path, times_str)
    os.makedirs(exp_save_path, exist_ok=True)

    save_file = os.path.join(exp_save_path, 'config_items.txt')

    config = configparser.ConfigParser()

    config.read('config.py')

    with open(save_file, 'w') as f:
        for section in config.sections():
            for key, value in config.items(section):
                f.write("{}={}\n".format(key, value))

    LOGGER.info("配置项已保存到 {}".format(save_file))

    code_save_path = os.path.join(exp_save_path, "code")
    os.makedirs(code_save_path, exist_ok=True)

    shutil.copy("config.py", code_save_path)
    shutil.copy("run.py", code_save_path)
    shutil.copy("model.py", code_save_path)

    LOGGER.info("关键代码已保存到 {}".format(code_save_path))

    return exp_save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
