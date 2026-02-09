import torch
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.sr3_modules.unet import UNet
from models.network import Network, make_beta_schedule

class BEVFeaturesDataset(Dataset):
    def __init__(self, img_path, pts_path):
        self.img_path = img_path
        self.pts_path = pts_path

        self.img_bev_embed = torch.load(self.img_path)['img_bev_embed']
        self.pts_bev_embed = torch.load(self.pts_path)['pts_bev_embed']

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'image': self.img_bev_embed[idx], 'points': self.pts_bev_embed[idx]}



model = dict(
    in_channel=512,
    out_channel=256,
    inner_channel=32,
    norm_groups=32,
    channel_mults=(1, 2, 4, 8, 8),
    attn_res=[8],
    res_blocks=3,
    dropout=0,
    with_noise_level_emb=True,
    image_size=208
)

beta_schedule = dict(
    train=dict(
        schedule='linear',
        linear_start=1e-6,
        linear_end=0.01,
        n_timestep=2000
    ),
    test=dict(
        schedule='linear',
        linear_start=1e-4,
        linear_end=0.09,
        n_timestep=1000
    )
)

network = Network(
    unet = model,
    beta_schedule = beta_schedule,
    module_name='sr3',)
network.set_new_noise_schedule(device=torch.device('cpu'), phase='train')
network.set_loss(torch.nn.MSELoss())

# summary(network, input_size=[(1, 256, 208, 208), (1, 256, 208, 208)], device='cpu')