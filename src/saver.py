import torch
import wandb
import numpy as np
from time import time
from pathlib import Path
from typing import Union
from datetime import datetime
from os.path import getmtime
import matplotlib.pyplot as plt

class Saver(object):
    """
    Saver allows for saving and restore networks.
    """
    def __init__(self, base_output_dir : Path, tag=''):

        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
        #if isinstance(tag, str) and len(tag) > 0:
            # Append tag
        #    timestamp_str += f"_{tag}"
        self.path = base_output_dir/f'{tag}_{timestamp_str}'
        self.path.mkdir(parents=True, exist_ok=True)

        # Create checkpoint sub-directory
        self.ckpt_path = self.path#/'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Create a buffer for metrics
        self.buffer = {}

    @staticmethod
    def init_wandb():
        return

    def watch_model(self, model):
        wandb.watch(model, log='all', log_freq=500)

    def save_configuration(self, config: dict):
        drop_keys = ['loaders', 'samplers', 'scheduler', 'saver']
        for key in drop_keys:
            if key in config:
                del config[key]
        torch.save(config, self.ckpt_path/f"config.pth")

    @staticmethod
    def load_configuration(model_path: Union[str, Path]):
        return torch.load(model_path/f"config.pth")

    def log_configuration(self):
        return

    def save_model(self, net: torch.nn.Module, name: str, epoch: int, model_name = None):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        # Save
        if model_name is None:
            torch.save(state_dict, self.ckpt_path/f"{name}_{epoch:05d}.pth")
        else:
            torch.save(state_dict, self.ckpt_path/f"{model_name}.pth")

    def save_checkpoint(self, net: torch.nn.Module, optim: torch.optim.Optimizer, config: dict, stats: dict, name: str, epoch: int):
        """
        Save model parameters and stats in the checkpoint directory.
        """
        # Get state dict
        net_state_dict = net.state_dict()
        optim_state_dict = optim.state_dict()

        # Copy to CPU
        for k, v in net_state_dict.items():
            net_state_dict[k] = v.cpu()
        for k, v in optim_state_dict.items():
            optim_state_dict[k] = v.cpu()

        # Save
        torch.save({
            'net_state_dict': net_state_dict,
            'optim_state_dict': optim_state_dict,
            'config': config,
            'stats':stats}, 
            self.ckpt_path/f"{name}_{epoch:05d}.pth")

    def log(self):
        """
        Empty the buffer and log all elements
        """
        wandb.log(self.buffer)
        self.buffer = {}

    def add_scalar(self, name: str, value: float, iter_n: int, iter_name='epoch'):
        """
        Add a scalar to buffer
        """
        self.buffer[name] = value
        self.buffer[iter_name] = iter_n

    def add_confusion_matrix(self, name:str, labels: torch.Tensor, predictions: torch.Tensor, iter_n: int, iter_name='epoch'):
        """
        Add a confusion matrix to buffer
        """
        cm = wandb.plot.confusion_matrix(y_true=labels, preds=predictions)
        self.buffer[name] = cm
        self.buffer[iter_name] = iter_n

    def add_images(self, name: str, images_vector: torch.Tensor, iter_n: int, iter_name='epoch'):
        """
        Add a scalar to buffer
        """
        images = wandb.Image(images_vector, caption=name)
        self.buffer[name] = images
        self.buffer[iter_name] = iter_n

    def log_scalar(self, name: str, value: float, iter_n: int, iter_name='epoch'):
        '''
        Log loss to wandb
        '''
        wandb.log({name: value, iter_name: iter_n})

    def log_images(self, name: str, images_vector: torch.Tensor, iter_n: int, iter_name='epoch'):
        '''
        Log images to wandb
        image_vector.shape = (C, W, H)
        '''
        images = wandb.Image(images_vector, caption=name)
        wandb.log({name: images, iter_name: iter_n})

    def add_plot(self, name: str, fig, iter_n: int, iter_name='epoch'):
        """
        Add a plot to buffer
        """
        self.buffer[name] = fig
        self.buffer[iter_name] = iter_n