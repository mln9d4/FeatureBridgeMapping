"""
Custom dataset implementation
"""
from logging import root
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from einops import rearrange

class PaddDataset(object):
    """
    Padd transform to make sure our input data has zero padding
    """
    def __init__(self, pad_size=8):
        self.pad_size = pad_size
        self.value = self.pad_size //2

    def __call__(self, sample):
        img_bev_embed, pts_bev_embed = sample['img_bev_embed'], sample['pts_bev_embed']
        img_bev_embed = torch.nn.functional.pad(img_bev_embed, (self.value, self.value, self.value, self.value), mode='constant', value=0)
        pts_bev_embed = torch.nn.functional.pad(pts_bev_embed, (self.value, self.value, self.value, self.value), mode='constant', value=0)
        return {'img_bev_embed': img_bev_embed, 'pts_bev_embed': pts_bev_embed}

class BEVFeaturesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data. Assuming in one file contains both the image and point cloud features.
        """
        self.root_dir = root_dir
        
        assert os.path.exists(self.root_dir), f"The folder '{self.root_dir}' does not exist."

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]
        self.num_files = len(self.data_files)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_file = torch.load(os.path.join(self.root_dir, self.data_files[idx]),
                               map_location='cpu')
        sample = {'img_bev_embed': rearrange(data_file['img_bev_embed'], '1 (w h) c -> c w h', w=200),
                   'pts_bev_embed': rearrange(data_file['pts_bev_embed'], '1 (w h) c -> c w h', w=200)}

        if self.transform:
            sample = self.transform(sample)
        sample['img_bev_embed'] = sample['img_bev_embed'][0:1, :, :]
        sample['pts_bev_embed'] = sample['pts_bev_embed'][0:1, :, :]
        return sample

    def __len__(self):
        return self.num_files
    
    def uncrop(self, singular_sample):
        singular_sample = singular_sample[ :, self.transform.value:-self.transform.value, self.transform.value:-self.transform.value]
        return singular_sample
    


if __name__ == "__main__":
    # Test the dataset
    padd = PaddDataset(pad_size=8)
    dataset=BEVFeaturesDataset(root_dir='/home/mingdayang/FeatureBridgeMapping/data/bev_features', transform=padd)

    for i, sample in enumerate(dataset):
        print(i, sample['img_bev_embed'].shape, sample['pts_bev_embed'].shape)
        uncropped_img = dataset.uncrop(sample['img_bev_embed'])
        print(uncropped_img.shape)