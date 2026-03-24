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
    def __init__(self, fmri, clip, pool):
        self.fmri = fmri
        self.clip = clip
        self.pool = pool
        
        print(f'Dataset samples: fmri-{len(self.fmri)} clip-{len(self.clip)}')

    def __getitem__(self, idx):
        fmri = self.fmri[idx]
        clip = self.clip[idx]
        pool = self.pool[idx]
        
        return fmri, clip, pool

    def __len__(self):
        return len(self.fmri)
    
class NSD_CLIP_Dataset(Dataset):
    def __init__(self, clip):
        self.clip = clip
        
        print(f'Dataset samples: clip-{len(self.clip)}')

    def __getitem__(self, idx):
        clip = self.clip[idx]
        
        return clip

    def __len__(self):
        return len(self.clip)

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
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
    pool = np.load(path).astype(np.float32)
    print(np.min(pool), np.max(pool), np.mean(pool), np.std(pool))
    
    pool = np.array([feat for feat in pool for _ in range(3)])
    pool = pool[:750*args.hour]
    print(f'Train ClipPool Sub{subj}: {pool.shape}')
    
    print(f'Train batch size: {args.batch_size}')
    dataset = NSD_Dataset(fmri, clip, pool)
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
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
    pool = np.load(path).astype(np.float32)
    print(np.min(pool), np.max(pool), np.mean(pool), np.std(pool))
    print(f'Val ClipPool Sub{subj}: {pool.shape}')
    
    print(f'Val batch size: {args.test_batch_size}')
    dataset = NSD_Dataset(fmri, clip, pool)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return dataloader

def val_nsd_dataloader_3468(args):
    mode = 'test'
    subj = args.subject
    if args.zscore:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_zscore_avg_sub{subj}.npy')
    # else:
    #     path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_scale_sub{subj}.npy')
    fmri = np.load(path).astype(np.float32)
    print(np.min(fmri), np.max(fmri), np.mean(fmri), np.std(fmri))

    # fmri = fmri.mean(axis=1)
    print(f'Val fMRI Sub{subj}: {fmri.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
    clip = np.load(path).astype(np.float32) 
    print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))
    print(f'Val Clip Sub{subj}: {clip.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
    pool = np.load(path).astype(np.float32)
    print(np.min(pool), np.max(pool), np.mean(pool), np.std(pool))
    print(f'Val ClipPool Sub{subj}: {pool.shape}')
    
    print(f'Val batch size: {args.test_batch_size}')
    dataset = NSD_Dataset(fmri, clip, pool)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return dataloader

def val_nsd_dataloader_select(args):
    mode = 'test'
    subj = args.subject
    select = args.select
    if args.zscore:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_{mode}_fmri_zscore_sub{subj}.npy')
    fmri = np.load(path).astype(np.float32)
    print(np.min(fmri), np.max(fmri), np.mean(fmri), np.std(fmri))

    fmri = fmri.mean(axis=1)[select]
    print(f'Val fMRI Sub{subj}: {fmri.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
    clip = np.load(path).astype(np.float32)[select]
    print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))
    print(f'Val Clip Sub{subj}: {clip.shape}')
    
    path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
    pool = np.load(path).astype(np.float32)[select]
    print(np.min(pool), np.max(pool), np.mean(pool), np.std(pool))
    print(f'Val ClipPool Sub{subj}: {pool.shape}')
    
    print(f'Val batch size: {args.test_batch_size}')
    dataset = NSD_Dataset(fmri, clip, pool)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return dataloader


def train_nsd_aug_dataloader(args):
    mode = 'train'
    # subj = args.subject
    all_clip = []
    for subj in [2, 5, 7]:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
        clip = np.load(path).astype(np.float32)
        print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))

        clip = clip[:250*args.hour]
        print(f'Train Clip Sub{subj}: {clip.shape}')

        all_clip.append(clip)
        
    all_clip = np.concatenate(all_clip, axis=0)
    print(f'Train batch size: {args.batch_size}')
    dataset = NSD_CLIP_Dataset(all_clip)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return dataloader

def train_nsd_augment_dataloader(args):
    mode = 'train'
    # subj = 2
    
    path = "/home/bingxing2/ailab/group/ai4neuro/BrainVL/NeuroFlow/evals/fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-aug/sub1/single_s1_all_recon_fmri.npy"
    fmri = np.load(path).astype(np.float32)
    print(np.min(fmri), np.max(fmri), np.mean(fmri), np.std(fmri))
    
    # b, v = fmri.shape
    # fmri = fmri.reshape(b*c, v)
    fmri = fmri.squeeze(1)
    # fmri = fmri[:250*args.hour]
    print(f'Train Recon fMRI: {fmri.shape}')
    
    all_clip = []
    all_pool = []
    for subj in [2, 5, 7]:
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_{mode}_sub{subj}.npy')
        clip = np.load(path).astype(np.float32)
        print(np.min(clip), np.max(clip), np.mean(clip), np.std(clip))
        
        # clip = np.array([feat for feat in clip for _ in range(3)])
        # clip = clip[:250*args.hour]
        print(f'Train Clip Sub{subj}: {clip.shape}')
        
        path = os.path.join(args.data_path, f'nsd/subj0{subj}/nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
        pool = np.load(path).astype(np.float32)
        print(np.min(pool), np.max(pool), np.mean(pool), np.std(pool))
        
        # pool = np.array([feat for feat in pool for _ in range(3)])
        # pool = pool[:250*args.hour]
        print(f'Train ClipPool Sub{subj}: {pool.shape}')
        
        all_clip.append(clip)
        all_pool.append(pool)
    
    all_clip = np.concatenate(all_clip, axis=0)
    all_pool = np.concatenate(all_pool, axis=0)
    
    print(f'Train batch size: {args.batch_size}')
    dataset = NSD_Dataset(fmri, all_clip, all_pool)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    
    return dataloader