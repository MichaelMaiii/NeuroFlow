from sklearn.decomposition import PCA
# from umap._umap import UMAP
import os
import sys
sys.path.append("/home/maiweijian/project/NeuroFlow/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/xfm/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/vae/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/sdxl/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/sdxl/generative_models")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
import argparse
import copy
from copy import deepcopy
import logging

from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from sit import SiT
from loss import SILoss
from neurovae import NeuroVAE_V10

from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torch.distributed as dist
from dataset import *
from mind_utils import *
from utils import *

from scipy.spatial.distance import euclidean
from umap.umap_ import UMAP

from samplers import euler_sampler_cycle_reverse, euler_sampler_bwd_reverse, euler_sampler_fwd_reverse

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA

def mindeye_normalize(fmri, subj):
    # 加载保存好的标准化参数
    norm_params = np.load(f'/home/maiweijian/project/SynBrain/src/mindeye2/norm_mean_scale_sub{subj}.npz')

    norm_mean_train = norm_params['mean']
    norm_scale_train = norm_params['scale']

    # 将mean和scale转为tensor，并放到fmri的device上
    norm_mean_train = torch.tensor(norm_mean_train, dtype=torch.float32, device=fmri.device)
    norm_scale_train = torch.tensor(norm_scale_train, dtype=torch.float32, device=fmri.device)

    # 使用加载的均值和标准差进行标准化
    fmri = (fmri - norm_mean_train) / norm_scale_train
    
    return fmri

def compute_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot_XY = torch.norm(X @ Y.T) ** 2
    dot_XX = torch.norm(X @ X.T) ** 2
    dot_YY = torch.norm(Y @ Y.T) ** 2
    return (dot_XY / (torch.sqrt(dot_XX * dot_YY) + 1e-8)).item()

def evaluate_voxel_and_structural_metrics(all_recon, all_fmri):
    """
    Evaluate voxel-level (MSE, Pearson) and structural-level (CKA, Cosine Similarity) metrics.
    
    Args:
        all_recon: Tensor of shape [N, 1, D]
        all_fmri: Tensor of shape [N, 1, D]
    
    Returns:
        dict with keys: "MSE", "Pearson", "CKA", "Cosine"
    """
    N, _, D = all_recon.shape
    all_recon = all_recon.squeeze(1)  # [N, D]
    all_fmri = all_fmri.squeeze(1)  # [N, D]

    mse_vals = []
    pearson_vals = []

    for i in range(N):
        recon = all_recon[i]
        target = all_fmri[i]
        mse = torch.mean((recon - target) ** 2).item()
        p = pearsonr(recon.cpu().numpy(), target.cpu().numpy())[0]
        mse_vals.append(mse)
        pearson_vals.append(p)

    # Flatten across trials for structure-level comparison
    recon_flat = all_recon.view(-1, D)   # [N, D]
    target_flat = all_fmri.view(-1, D)   # [N, D]

    # CKA
    cka = compute_cka(recon_flat, target_flat)

    # Cosine Similarity (averaged per sample)
    recon_np = recon_flat.cpu().numpy()
    target_np = target_flat.cpu().numpy()
    cos_sim = np.mean([
        cosine_similarity(recon_np[i:i+1], target_np[i:i+1])[0, 0]
        for i in range(recon_np.shape[0])
    ])
    
    # 计算总体解释方差
    total_var = np.var(target_np)
    residual_var = np.var(target_np - recon_np)
    explained_variance_total = 1 - (residual_var / total_var) if total_var != 0 else 0.0
    print(f"总体解释方差: {explained_variance_total:.4f}")
    
    # 计算每个体素的指标
    num_voxels = target_np.shape[1]
    spearman_corr = np.zeros(num_voxels)
    explained_variance = np.zeros(num_voxels)
    
    for voxel in range(num_voxels):
        # 提取该体素在所有样本中的原始值和重建值
        x_voxel = target_np[:, voxel]
        recon_voxel = recon_np[:, voxel]
        
        # 计算Spearman相关系数
        corr, _ = spearmanr(x_voxel, recon_voxel)
        spearman_corr[voxel] = corr
        
        # 计算解释方差 (避免除以零)
        var_x = np.var(x_voxel)
        if var_x == 0:
            explained_variance[voxel] = 0.0  # 无变异的体素无法被解释
        else:
            var_residual = np.var(x_voxel - recon_voxel)
            explained_variance[voxel] = 1 - (var_residual / var_x)
    
    # 计算体素级Spearman相关系数的统计量
    median_corr = np.median(spearman_corr)
    p90_corr = np.percentile(spearman_corr, 90)
    p95_corr = np.percentile(spearman_corr, 95)
    p99_corr = np.percentile(spearman_corr, 99)
    
    print(f"\n体素级Spearman相关系数统计:")
    print(f"中位数 (Median): {median_corr:.4f}")
    print(f"90% 分位数: {p90_corr:.4f}")
    print(f"95% 分位数: {p95_corr:.4f}")
    print(f"99% 分位数: {p99_corr:.4f}")
    
    # # 计算体素级解释方差的统计量
    median_ev = np.median(explained_variance)
    p90_ev = np.percentile(explained_variance, 90)
    p95_ev = np.percentile(explained_variance, 95)
    p99_ev = np.percentile(explained_variance, 99)
    
    print(f"\n体素级解释方差统计:")
    print(f"中位数 (Median): {median_ev:.4f}")
    print(f"90% 分位数: {p90_ev:.4f}")
    print(f"95% 分位数: {p95_ev:.4f}")
    print(f"99% 分位数: {p99_ev:.4f}")
    
    return {
        "MSE": np.mean(mse_vals),
        "Pearson": np.mean(pearson_vals),
        "CKA": cka,
        "Cosine": cos_sim,
        "Spearman": median_corr,
        "EV": median_ev
    }


def compute_retrieval(x_fmri, target, device):
    clip_voxels_norm = nn.functional.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}   Backward top1: {bwd_percent_correct}")
    
def compute_retrieval_fwd(x_fmri, target, device):
    clip_voxels_norm = nn.functional.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}")
    
def compute_retrieval_bwd(x_clip, target, device):
    clip_voxels_norm = nn.functional.normalize(x_clip.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Backward top1: {bwd_percent_correct}")


def plot_umap(clip_target, aligned_clip_voxels):
    print('umap plotting...')
    combined = np.concatenate((clip_target.flatten(1).detach().cpu().numpy(),
                                aligned_clip_voxels.flatten(1).detach().cpu().numpy()), axis=0)
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined)

    batch = int(len(embedding) // 2)
    umap_distance = [euclidean(point1, point2) for point1, point2 in zip(embedding[:batch], embedding[batch:])]
    avg_umap_distance = np.mean(umap_distance)
    print(f"Average UMAP Euclidean Distance: {avg_umap_distance}")

    colors = np.array([[0, 0, 1, .5] for i in range(len(clip_target))])
    colors = np.concatenate((colors, np.array([[0, 1, 0, .5] for i in range(len(aligned_clip_voxels))])))

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors)
    plt.title(f"Avg.Euclidean Distance = {avg_umap_distance:.4f}")
    
    return fig

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_brain_autoencoder_mlp(args):
    from brain_autoencoder import BrainEncoder, BrainDecoder, BrainAutoEncoder
    
    encoder = BrainEncoder(in_dim=15724, out_dim=256*1664, clip_size=1664, h=2048)
    decoder = BrainDecoder(in_dim=256*1664, out_dim=15724, h=2048)
    
    model = BrainAutoEncoder(encoder=encoder, decoder=decoder, clip_weight=1000)
    
    model_path = f'/home/bingxing2/ailab/group/ai4neuro/BrainVL/train_logs/{args.model_path}/last.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def load_brain_autoencoder(args):
    from brain_autoencoder import BrainAutoencoderKL2D_joint
    
    config = load_config("/home/bingxing2/ailab/maiweijian/NeuroFlow/configs/autoencoder_kl_64x64x256.yaml")
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = BrainAutoencoderKL2D_joint(ddconfig=ddconfig,
                        clip_weight=1000,
                        hidden_dim=4096,
                        cycle=False,
                        )
    
    model_path = f'/home/bingxing2/ailab/group/ai4neuro/BrainVL/train_logs/{args.brain_path}/last.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def load_brain_vae(args):
    
    config = load_config(f"/home/maiweijian/project/NeuroFlow/configs/neurovae{args.hidden_dim}_V10.yaml")
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = NeuroVAE_V10(ddconfig=ddconfig)
    
    model_path = f'/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow/train_logs/{args.vae_path}/last.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load BrainVAE Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    
    device = "cuda"
    set_seed(args.seed)
    
    subj = f"sub{args.subject}"
    # save_path = os.path.join(args.save_path, "evals", args.model_name, subj)
    save_path = os.path.join(args.save_path, "evals", args.save_name, subj)
    os.makedirs(save_path, exist_ok=True)
    
    blurry_save_path = os.path.join(save_path, "blurry")
    os.makedirs(blurry_save_path, exist_ok=True)
    
    #! Load MindEye2 for reconstructed fMRI decoding
    args.mindeye_ckpt = f"final_subj0{args.subject}_pretrained_40sess_24bs"
    voxel_dict = {1:15724, 2:14278, 5:13039, 7:12682}
    args.num_voxels = voxel_dict[args.subject]
    mindeyev2 = load_mindeye2(args)
    requires_grad(mindeyev2, False)
    print("params of mindeyev2:")
    count_params(mindeyev2)
    
    # #! Load SDXL UnClip decoder using CPU, parameter: 4.5B
    # diffusion_engine, vector_suffix = load_pretrained_sdxl_unclip()
    # requires_grad(diffusion_engine, False)
    # print("params of sdxl:")
    # count_params(diffusion_engine)
    
    #! Load test dataset
    # train_dataloader = train_nsd_dataloader(args)
    test_dataloader = val_nsd_dataloader(args)

    all_recon_blurry = None
    count_f2i = 0
    for x_fmri, z_clip, sub in test_dataloader:  # x is vae features, y is labels
        with torch.no_grad():
            x_fmri = x_fmri.float().unsqueeze(1).to(device)
            x_length = x_fmri.shape[-1]
            z_clip = z_clip.float().to(device)
            
            recon_blurry = mindeyev2_blurry(mindeyev2, x_fmri)
            
            if all_recon_blurry is None:
                all_recon_blurry = recon_blurry.cpu()
            else:
                all_recon_blurry = torch.vstack((all_recon_blurry, recon_blurry.cpu()))

                
            for i in range(len(recon_blurry)):
                count_f2i += 1
                recon_blurry_resized = transforms.Resize((256, 256))(transforms.ToPILImage()(recon_blurry[i]))
                recon_blurry_resized.save(f"{blurry_save_path}/{count_f2i}.png")
                print(f"Generating {count_f2i}/1000 images......")
    
    # resize
    imsize = 256
    all_recon_blurry = transforms.Resize((imsize,imsize))(all_recon_blurry).float()

    # saving
    print(all_recon_blurry.shape)
    setting_name = args.setting_name
    torch.save(all_recon_blurry,f"{save_path}/{setting_name}_all_recon_blurry.pt")
    print(f"saved {args.model_name} outputs!")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--ckpt-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow/train_logs")
    parser.add_argument("--save-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow")
    parser.add_argument("--encoder", type=str, default="vae", choices=["mlp", "conv", "vae"])
    parser.add_argument("--vae-path", type=str, default="neurovae-nsd-s7-vs7-bs64-d1664-zscore-v10-cycle")
    parser.add_argument("--setting-name", type=str, default="single_s7")
    parser.add_argument("--subject", type=int, default=7) #!记得改
    parser.add_argument("--hidden-dim", type=int, default=1664) #!记得改

    # nohup python generate_neuroflow_V10_reverse.py > logs/ms_d12_h13_s1_cos_reverse_woclip.log 2>&1 &
    parser.add_argument("--prediction", type=str, default="v")
    parser.add_argument("--model-name", type=str, default="fm-s7-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse")
    parser.add_argument("--save-name", type=str, default="fm-s7-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse")
    parser.add_argument("--num-step", type=int, default=20)
    parser.add_argument("--sample",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zscore",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-depth", type=int, default=12)
    parser.add_argument("--model-head", type=int, default=13)
    parser.add_argument("--save_img", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--data-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data/")

    # # precision
    # parser.add_argument("--allow-tf32", action="store_true")
    # parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # seed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-log", action=argparse.BooleanOptionalAction, default=False)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
