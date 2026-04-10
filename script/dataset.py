from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import argparse
import os
import PIL
from PIL import Image
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as tvtrans
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

class NSD_Dataset(Dataset):
    def __init__(self, fmri, clip):
        self.fmri = fmri
        self.clip = clip
        
        print(f'Dataset samples: fmri-{len(self.fmri)} clip-{len(self.clip)}')

    def __getitem__(self, idx):
        fmri = self.fmri[idx]
        clip = self.clip[idx]
        
        return fmri, clip

    def __len__(self):
        return len(self.fmri)


def train_nsd_dataloader(args):
    mode = 'train'
    subj = args.subject
    if args.zscore:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_zscore_sub{subj}.npy')
    # else:
    #     path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_scale_sub{subj}.npy')
    fmri = np.load(path).astype(np.float32)
    print(np.min(fmri), np.max(fmri), np.mean(fmri), np.std(fmri))
    
    b, c, v = fmri.shape
    fmri = fmri.reshape(b*c, v)
    fmri = fmri[:750*args.hour]
    print(f'Train fMRI Sub{subj}: {fmri.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
    clip = np.load(path).astype(np.float32)
    print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))
    
    clip = np.array([feat for feat in clip for _ in range(3)])
    clip = clip[:750*args.hour]
    print(f'Train Clip Sub{subj}: {clip.shape}')
    
    print(f'Train batch size: {args.batch_size}')
    dataset = NSD_Dataset(fmri, clip)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    
    return dataloader

def val_nsd_dataloader(args):
    mode = 'test'
    subj = args.subject
    if args.zscore:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_zscore_sub{subj}.npy')
    # else:
    #     path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_scale_sub{subj}.npy')
    fmri = np.load(path).astype(np.float32)
    print(np.min(fmri), np.max(fmri), np.mean(fmri), np.std(fmri))

    fmri = fmri.mean(axis=1)
    print(f'Val fMRI Sub{subj}: {fmri.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
    clip = np.load(path).astype(np.float32) 
    print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))
    print(f'Val Clip Sub{subj}: {clip.shape}')
    
    print(f'Val batch size: {args.test_batch_size}')
    dataset = NSD_Dataset(fmri, clip)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return dataloader