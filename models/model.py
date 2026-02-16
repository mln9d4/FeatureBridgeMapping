import os
import torch
import tqdm
import numpy as np
import copy
from .network import Network
import torch.nn as nn
import os

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class RollingStatistics():
    """
    Class for capturing the rolling statistics when doing online training. Calculates the mean and variance of the data per batch and updates the overall mean and variance using https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html formula.

    For online training only do it for one epoch after we have the overall statistics, then we can fix the statistics and do normalization and denormalization using the fixed statistics. This is because the statistics will not change much after one epoch of training, and it will be more stable to use fixed statistics for normalization and denormalization during training.

    Attributes:
        mean: The rolling mean of the data.
        var: The rolling variance of the data.
        std: The rolling standard deviation of the data.
        p_samples: The total number of observations seen so far. Used for calculating the new mean and variance when a new batch of data is observed. p_samples is the amount of samples in a batch.
    """
    def __init__(self, mean=None, var=None, std=None, p_samples=0):
        self.fixed = False
        self.mean = mean
        self.var = var
        self.std = std
        self.p_samples = p_samples

    def set_device(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.var is not None:
            self.var = self.var.to(device)
        if self.std is not None:
            self.std = self.std.to(device)

    def __str__(self):
        return f"RollingStatistics(mean={torch.mean(self.mean)}, var={torch.mean(self.var)}, std={torch.mean(self.std)}, p_samples={self.p_samples})"

    def update(self, batch_data):
        """
        Update rolling statistics
        Args:
            batch_data: The new batch of data to update the rolling statistics with. Should be a tensor of shape (batch_size, channels, width, height).
        
        """
        if self.fixed:
            return
        dims = (0, 2, 3)
        batch_mean = torch.mean(batch_data, dim=dims, keepdim=True) 
        batch_var = torch.var(batch_data, dim=dims, keepdim=True)
        q_samples = batch_data.shape[0] # Assuming batch_data is of shape (batch_size, features)

        if self.p_samples == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.std = torch.sqrt(self.var)
            self.p_samples = q_samples
        else:
            new_mean = self.p_samples * self.mean / (self.p_samples + q_samples) + q_samples * batch_mean / (self.p_samples + q_samples)

            new_var = self.p_samples * self.var / (self.p_samples + q_samples) + q_samples * batch_var / (self.p_samples + q_samples) + (self.p_samples * q_samples) * torch.pow(self.mean - batch_mean, 2) / (self.p_samples + q_samples)**2
            
            self.mean = new_mean
            self.var = new_var
            self.std = torch.sqrt(self.var)
            self.p_samples += q_samples

    def get_stats(self):
        return {'mean': self.mean, 'var': self.var, 'std': self.std, 'p_samples': self.p_samples}
    

    def normalize(self, sample):
        if self.mean is None or self.std is None:
            raise ValueError("Rolling statistics have not been updated with any data yet.")
        return (sample - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, sample):
        if self.mean is None or self.std is None:
            raise ValueError("Rolling statistics have not been updated with any data yet.")
        return sample * (self.std + 1e-8) + self.mean

class Palette():
    def __init__(self, network_config=None, beta_schedule=None, rolling_statistics_cfg=None, ema_scheduler=None, opts=None, loss_fn=None, phase='train'):
        """
        Class that handles two networks, one is the current network and the other is the ema network. The current network is used for training and the ema network is used for validation. The ema network is updated using the EMA class after each training step. The class also handles the rolling statistics for normalization and denormalization of the data during training and validation. 

        Class handles Loading and Saving of networks.   
        """
        self.phase = phase
        self.opts = opts
        self.network_config = network_config
        self.beta_schedule = beta_schedule
        self.rolling_stats_cfg = rolling_statistics_cfg

        if opts['resume'] is True:
            print("Loading checkpoint for training / inference.")
            self.load_network()
            if ema_scheduler is not None:
                self.ema_scheduler = ema_scheduler
                self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
                print("++++++++++++ Loaded EMA network from checkpoint. +++++++++++++")

        else:
            print("Starting training from scratch.")
            self.network = Network(self.network_config, self.beta_schedule)
            
            self.epoch = 0

            if rolling_statistics_cfg is not None:
                self.rolling_statistics_X = RollingStatistics(**rolling_statistics_cfg['X'])
                self.rolling_statistics_y = RollingStatistics(**rolling_statistics_cfg['y'])
            else:
                self.rolling_statistics_X = RollingStatistics()
                self.rolling_statistics_y = RollingStatistics()

        
            if ema_scheduler is not None:
                self.ema_scheduler = ema_scheduler
                self.ema_network = copy.deepcopy(self.network)
                self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
            else:
                self.ema_network = None

        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.network.set_loss(self.loss_fn)
        self.network.set_new_noise_schedule(phase=self.phase)
        self.ema_network.set_new_noise_schedule(phase=self.phase) if self.ema_network is not None else None


    def set_device(self, device):
        self.network.to(device)
        if self.ema_network is not None:
            self.ema_network.to(device)

    def train_step(self, y_cond, y_0):
        # y_cond is the condition aka camera features, y_0 is the original point cloud features, t is the time step
        # y_cond is also known as X and y_0 is also known as y

        self.network.train()
        loss = self.network(y_0=y_0, y_cond=y_cond)
        
        if self.ema_network is not None:
            self.EMA.update_model_average(self.ema_network, self.network)
        return loss
    
    def val_step(self, y_cond, n_steps=10, use_tqdm=True):
        """
        Do validation step by using DPM-Solver to sample from the model. If use_ema is True, use the ema_network for sampling, otherwise use the current network. y_cond is the condition aka camera features, n_steps is the number of steps for DPM-Solver sampling, use_tqdm is whether to use tqdm for progress bar.

        Args:
            y_cond: The condition aka camera features for sampling.
            n_steps: The number of steps for DPM-Solver sampling.
            use_tqdm: Whether to use tqdm for progress bar.
            use_ema: Whether to use the ema_network for sampling, if False, use the current network.
        Returns:
            y_sampled, ret_arr: y_sampled is the sampled point cloud features, ret_arr is the array of intermediate results during sampling.
        """
        with torch.no_grad():
            self.ema_network.eval()
    
            output, ret_arr = self.ema_network.restoration(y_cond=y_cond)
        print(f"Validation step completed. Output shape: {output.shape}, ret_arr shape: {ret_arr.shape}")
        return output, ret_arr
    

    def test_step(self, y_cond, n_steps=10, use_tqdm=True):
        with torch.no_grad():
            self.network.eval()

            output, ret_arr = self.network.dpm_solver_sampling(y_cond=y_cond, n_steps=n_steps, use_tqdm=use_tqdm)

        return output, ret_arr
    
    def save_network(self):
        if not os.path.exists(self.opts['path']['save_dir']):
            os.makedirs(self.opts['path']['save_dir'])

        state_dict = self.network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        ema_state_dict = self.ema_network.state_dict() if self.ema_network is not None else None
        for key, param in ema_state_dict.items() if ema_state_dict is not None else []:
            ema_state_dict[key] = param.cpu()

        checkpoint = {
            'model_config': self.network_config,
            'beta_schedule': self.beta_schedule,
            'rolling_stats_cfg': {
                'X': self.rolling_statistics_X.get_stats(),
                'y': self.rolling_statistics_y.get_stats()
            },
            'network_state_dict': state_dict,
            'ema_network_state_dict': self.ema_network.state_dict() if self.ema_network is not None else None
        }
        model_name = f"{self.opts['architecture']}_model_epoch_{self.epoch}.pth"
        save_path = os.path.join(self.opts['path']['save_dir'], model_name)
        torch.save(checkpoint, save_path)
        print(f"Network saved to {save_path}")

    def load_network(self):
        if self.opts['path']['checkpoint'] and os.path.exists(self.opts['path']['checkpoint']):
            checkpoint = torch.load(self.opts['path']['checkpoint'], map_location='cpu')
            self.network_config = checkpoint['model_config']
            self.beta_schedule = checkpoint['beta_schedule']
            self.network = Network(self.network_config, self.beta_schedule)
            self.network.set_new_noise_schedule(phase=self.phase)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            if checkpoint['ema_network_state_dict'] is not None:
                self.ema_network = copy.deepcopy(self.network)
                self.ema_network.set_new_noise_schedule(phase=self.phase)
                self.ema_network.load_state_dict(checkpoint['ema_network_state_dict'])
            
            if 'rolling_stats_cfg' in checkpoint:
                self.rolling_stats_cfg = checkpoint['rolling_stats_cfg']
                self.rolling_statistics_X = RollingStatistics(**self.rolling_stats_cfg['X'])
                self.rolling_statistics_y = RollingStatistics(**self.rolling_stats_cfg['y'])
                self.rolling_statistics_X.set_device(self.opts['device'])
                self.rolling_statistics_y.set_device(self.opts['device'])


            print(f"Network loaded from {self.opts['path']['checkpoint']}")
        else:
            print(f"No checkpoint found at {self.opts['path']['checkpoint']}. Starting with a new network.")

    # def save_model(self, epoch):
    #     save_file_name = f"{self.opts['save_dir']}/network_epoch_{epoch}.pth"
    #     state_dict = self.network.state_dict()
    #     for key, param in state_dict.items():
    #         state_dict[key] = param.cpu()
    #     torch.save(state_dict, save_file_name)