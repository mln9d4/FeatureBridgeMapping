import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from data.dataset import BEVFeaturesDataset, PaddDataset
from torch.utils.data import DataLoader, Dataset
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import DataLoader, Subset
import math
import torch
from torch import nn
from numpy import mean, var
from functools import partial
from inspect import isfunction
from mmdet.models import HEADS
from mmdet3d.unibev_plugin.models.dense_heads import diffusion_model


beta_schedule = dict(
    train=dict(
        schedule='linear',
        n_timestep=2000,
        linear_start=1e-4,
        linear_end=2e-2,
    ),
    test=dict(
        schedule='linear',
        n_timestep=1000,
        linear_start=1e-5,
        linear_end=1e-1,
    )
)

model_config = dict(
    in_channel=2,
    out_channel=1,
    inner_channel=64,
    norm_groups=32,
    channel_mults=(1, 2, 4, 8),
    attn_res=(25,),
    res_blocks=2,
    dropout=0,
    with_noise_level_emb=True,
    image_size=200,
    eps=1e-5
)

hyperparameters = dict(
    model_config=model_config,
    beta_schedule=beta_schedule,
    batch_size=12,
)

ModelClass = HEADS.get('DenoiseDiffusion')
diffusion = ModelClass(model_config, beta_schedule)



def load_data():
    # Load the saved data
    dataset = BEVFeaturesDataset(root_dir='/home/mingdayang/FeatureBridgeMapping/data/bev_features', transform=None)

    return dataset

def create_splits(dataset, train_split=0.8):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_split * len(dataset)), len(dataset) - int(train_split * len(dataset))])

    return train_dataset, test_dataset

def make_loader(batch_size, dataset):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Let's check out what we've created

    return dataloader

dataset = load_data()
train_dataset, test_dataset = create_splits(dataset)
train_loader = make_loader(hyperparameters['batch_size'], train_dataset)
test_loader = make_loader(hyperparameters['batch_size'], test_dataset)
single_loader = DataLoader(Subset(dataset, [0]), batch_size=hyperparameters['batch_size'], shuffle=False)

def save_model(model, model_config, beta_schedule):
    save_temp = {}
    model.to('cpu')
    save_temp['model_state_dict'] = model.eps_model.state_dict()
    save_temp['model_config'] = model_config
    save_temp['beta_schedule'] = beta_schedule
    save_temp['rolling_stats_img'] = model.rolling_stats_img.get_stats()
    save_temp['rolling_stats_pts'] = model.rolling_stats_pts.get_stats()
    torch.save(save_temp, 'checkpoints/latest_mini_nusc_one_sample_one_channel.pth')



# simple training loop using the existing `tensor` as toy data
device = 'cuda:1'
diffusion.to(device)
diffusion.set_new_noise_schedule(phase='train', device=device)
diffusion.train()

optimizer = torch.optim.Adam(diffusion.eps_model.parameters(), lr=1e-5)

epochs = 100000
val_interval = 100
save_interval = 100

try:
    with wandb.init(project="diffusion_test", config=hyperparameters) as run:
        run.watch(diffusion, log="all", log_freq=10)
        for epoch in range(epochs):
            diffusion.set_new_noise_schedule(phase='train', device=device)
            train_loss = 0
            for batch in tqdm(single_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                X, y = batch['img_bev_embed'], batch['pts_bev_embed']
                X = X.to(device)
                y = y.to(device)

                if epoch > 1:
                    diffusion.rolling_stats_img.fixed = True
                    diffusion.rolling_stats_pts.fixed = True
                
                diffusion.rolling_stats_img.update(X)
                diffusion.rolling_stats_pts.update(y)

                X = diffusion.rolling_stats_img.normalize(X)
                y = diffusion.rolling_stats_pts.normalize(y)

                loss = diffusion.forward(y0=y, y_cond=X)
                nn.utils.clip_grad_norm_(diffusion.eps_model.parameters(), max_norm=5.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                print(f"Batch Loss: {loss.item()}")
            
            avg_train_loss = train_loss / len(single_loader)
            run.log({'train_loss': avg_train_loss}, step=epoch)

            if (epoch + 1) % val_interval == 0:
                diffusion.set_new_noise_schedule(phase='test', device=device)
                val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(single_loader, desc=f'Validation at Epoch {epoch+1}/{epochs}'):
                        X, y = batch['img_bev_embed'], batch['pts_bev_embed']
                        X = X.to(device)
                        y = y.to(device)

                        X = diffusion.rolling_stats_img.normalize(X)
                        y = diffusion.rolling_stats_pts.normalize(y)

                        loss = diffusioI mean n.forward(y0=y, y_cond=X)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(single_loader)
                run.log({'val_loss': avg_val_loss}, step=epoch)


            if save_interval is not None and (epoch + 1) % save_interval == 0:
                save_model(diffusion, model_config, beta_schedule)
                diffusion.to(device)
            if avg_train_loss < 0.001:
                print(f"Early stopping at epoch {epoch+1} with average training loss {avg_train_loss}")
                break

        run.finish()
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    save_model(diffusion, model_config, beta_schedule)
    run.finish()