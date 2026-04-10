import os
import sys

sys.path.append("/home/maiweijian/project/NeuroFlow/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/xfm/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/vae/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/sdxl/")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/sdxl/generative_models")
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate.utils import set_seed

from sit import SiT
from dataset import val_nsd_dataloader
from samplers import *
from utils import *
from mind_utils import batchwise_cosine_similarity, topk

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def compute_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot_xy = torch.norm(X @ Y.T, p="fro", dim=None) ** 2
    dot_xx = torch.norm(X @ X.T, p="fro", dim=None) ** 2
    dot_yy = torch.norm(Y @ Y.T, p="fro", dim=None) ** 2
    return (dot_xy / (torch.sqrt(dot_xx * dot_yy) + 1e-8)).item()

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
    
    print("\n体素级Spearman相关系数统计:")
    print(f"中位数 (Median): {median_corr:.4f}")
    print(f"90% 分位数: {p90_corr:.4f}")
    print(f"95% 分位数: {p95_corr:.4f}")
    print(f"99% 分位数: {p99_corr:.4f}")
    
    # # 计算体素级解释方差的统计量
    median_ev = np.median(explained_variance)
    p90_ev = np.percentile(explained_variance, 90)
    p95_ev = np.percentile(explained_variance, 95)
    p99_ev = np.percentile(explained_variance, 99)
    
    print("\n体素级解释方差统计:")
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
    clip_voxels_norm = F.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = F.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}   Backward top1: {bwd_percent_correct}")
    
def compute_retrieval_fwd(x_fmri, target, device):
    clip_voxels_norm = F.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = F.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}")
    
def compute_retrieval_bwd(x_clip, target, device):
    clip_voxels_norm = F.normalize(x_clip.flatten(1), dim=-1)
    clip_target_norm = F.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Backward top1: {bwd_percent_correct}")


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    
    device = "cuda"
    set_seed(args.seed)
    
    subj = f"sub{args.subject}"
    save_path = os.path.join(args.save_path, "evals", args.save_name, subj)
    os.makedirs(save_path, exist_ok=True)
    
    img_save_path = os.path.join(save_path, "img")
    os.makedirs(img_save_path, exist_ok=True)
    
    f2i_save_path = os.path.join(save_path, "f2i")
    os.makedirs(f2i_save_path, exist_ok=True)
    
    #! Load SiT
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    ema = SiT(        
            num_patches=256,
            embed_size=1664,
            hidden_size=1664,
            depth=args.model_depth,
            num_heads=args.model_head,
            **block_kwargs)
    
    ckpt_name = 'last.pt'
    ckpt = torch.load(f'{os.path.join(args.ckpt_path, args.model_name)}/{ckpt_name}', map_location='cpu', weights_only=False)
    ema.load_state_dict(ckpt['ema'])
    global_step = ckpt['steps']
    print(f'Load Checkpoint from {global_step} steps......')
        
    ema.to(device).eval()
    requires_grad(ema, False)
    print("params of SiT:")
    count_params(ema)
    
    # #! Load SDXL UnClip decoder using CPU, parameter: 4.5B
    diffusion_engine, vector_suffix = load_pretrained_sdxl_unclip()
    requires_grad(diffusion_engine, False)
    print("params of sdxl:")
    count_params(diffusion_engine)
    
    #! Load Brain Autoencoder
    brain_enc = load_neurovae_v10_proj(args, checkpoint_root=args.ckpt_path).to(device).eval()
    requires_grad(brain_enc, False)
    print("params of brain encoder:")
    count_params(brain_enc)
    
    #! Load test dataset
    test_dataloader = val_nsd_dataloader(args)

    all_recon_fmri = None
    all_recon_f2i = None
    all_recon_i2f = None
    all_recon_img = None
    all_clipvoxels = None
    all_sample_fmri = None
    all_sample_clip = None
    all_clipvoxels_recon = None
    all_mse_disc = []
    count_img = 0
    count_f2i = 0
    for x_fmri, z_clip in test_dataloader:  
        with torch.no_grad():
            x_fmri = x_fmri.float().unsqueeze(1).to(device)
            z_clip = z_clip.float().to(device)

            z_fmri, z_fmri_clip = brain_enc.encode(x_fmri, sample=False)
                
            # #! retrieval -> 300 batch sizes
            compute_retrieval(z_fmri_clip.clone(), z_clip.clone(), device)
            if all_clipvoxels is None:
                all_clipvoxels = z_fmri_clip.cpu()
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, z_fmri_clip.cpu()))

            #! cycle sampling
            print('Using Euler Multi-step sampling..................')
            sample_clip, sample_fmri = euler_sampler_cycle(
                            ema,
                            z_fmri,
                            z_clip,
                            num_steps=args.num_step,
                            heun=args.heun,
                        )
        
            # compute_retrieval_bwd
            compute_retrieval_bwd(sample_fmri.clone(), z_fmri.clone(), device)
            if all_sample_fmri is None:
                all_sample_fmri = sample_fmri.clone().cpu()
            else:
                all_sample_fmri = torch.vstack((all_sample_fmri, sample_fmri.clone().cpu()))
                
            compute_retrieval_fwd(sample_clip.clone(), z_clip.clone(), device)
            if all_sample_clip is None:
                all_sample_clip = sample_clip.clone().cpu()
            else:
                all_sample_clip = torch.vstack((all_sample_clip, sample_clip.clone().cpu()))
            
            recon_fmri = brain_enc.generate(sample_fmri)
            if all_recon_fmri is None:
                all_recon_fmri = recon_fmri.cpu()
            else:
                all_recon_fmri = torch.vstack((all_recon_fmri, recon_fmri.cpu()))
                
            evaluate_voxel_and_structural_metrics(recon_fmri, x_fmri)
            
            #! recon fmri retrieval
            z_recon_fmri, z_recon_fmri_clip = brain_enc.encode(recon_fmri, sample=False)
                
            compute_retrieval(z_recon_fmri_clip.clone(), z_clip.clone(), device)
            if all_clipvoxels_recon is None:
                all_clipvoxels_recon = z_recon_fmri_clip.cpu()
            else:
                all_clipvoxels_recon = torch.vstack((all_clipvoxels_recon, z_recon_fmri_clip.cpu()))
                
            #! compute mse betweeen original and generated fmri
            mse_disc = F.mse_loss(x_fmri, recon_fmri)
            all_mse_disc.append(mse_disc.item())
            
            sample_f2i = euler_sampler_bwd(ema, z_recon_fmri, num_steps=args.num_step, heun=args.heun)
            
            sample_i2f = euler_sampler_fwd(ema, sample_clip, num_steps=args.num_step, heun=args.heun)
            recon_i2f = brain_enc.generate(sample_i2f)
            if all_recon_i2f is None:
                all_recon_i2f = recon_i2f.cpu()
            else:
                all_recon_i2f = torch.vstack((all_recon_i2f, recon_i2f.cpu()))
            
            for i in range(len(sample_clip)):
                recon_img = unclip_recon(sample_clip[i].unsqueeze(0).to(device),
                            diffusion_engine,
                            vector_suffix,
                            num_samples=1)
                
                if all_recon_img is None:
                    all_recon_img = recon_img.cpu()
                else:
                    all_recon_img = torch.vstack((all_recon_img, recon_img.cpu()))
                
                count_img += 1
                if args.save_img:
                    recon_img_resized = transforms.Resize((256, 256))(transforms.ToPILImage()(recon_img[0]))
                    recon_img_resized.save(f"{img_save_path}/{count_img}.png")
                    print(f"Generating {count_img}/1000 images......")
                
            for i in range(len(sample_f2i)):
                recon_f2i = unclip_recon(sample_f2i[i].unsqueeze(0).to(device),
                            diffusion_engine,
                            vector_suffix,
                            num_samples=1)
                
                if all_recon_f2i is None:
                    all_recon_f2i = recon_f2i.cpu()
                else:
                    all_recon_f2i = torch.vstack((all_recon_f2i, recon_f2i.cpu()))
                
                count_f2i += 1
                if args.save_img:
                    recon_f2i_resized = transforms.Resize((256, 256))(transforms.ToPILImage()(recon_f2i[0]))
                    recon_f2i_resized.save(f"{f2i_save_path}/{count_f2i}.png")
                    print(f"Generating {count_f2i}/1000 images......")
                    
    avg_mse_disc = np.mean(all_mse_disc)
    print(f"Average MSE distance between original and reconstructed fmri: {avg_mse_disc}")
    
    # resize
    imsize = 256
    all_recon_f2i = transforms.Resize((imsize,imsize))(all_recon_f2i).float()
    all_recon_img = transforms.Resize((imsize,imsize))(all_recon_img).float()

    # saving
    print(all_recon_f2i.shape)
    setting_name = args.setting_name
    torch.save(all_recon_f2i,f"{save_path}/{setting_name}_all_recon_f2i.pt")
    torch.save(all_recon_img,f"{save_path}/{setting_name}_all_recon_img.pt")
    torch.save(all_recon_fmri,f"{save_path}/{setting_name}_all_recon_fmri.pt")
    torch.save(all_recon_i2f,f"{save_path}/{setting_name}_all_recon_i2f.pt")
    torch.save(all_clipvoxels,f"{save_path}/{setting_name}_all_zfmri_raw.pt")
    torch.save(all_clipvoxels_recon,f"{save_path}/{setting_name}_all_zfmri_syn.pt")
    print(f"saved {args.model_name} outputs!")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--ckpt-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow/train_logs")
    parser.add_argument("--save-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow")
    parser.add_argument("--encoder", type=str, default="vae", choices=["mlp", "conv", "vae"])
    parser.add_argument("--vae-path", type=str, default="neurovae-nsd-s1-vs1-bs64-d1664-zscore-v10-cycle-proj")
    parser.add_argument("--setting-name", type=str, default="single_s1")
    parser.add_argument("--subject", type=int, default=1) #!记得改
    parser.add_argument("--hidden-dim", type=int, default=1664) #!记得改

    # nohup python generate.py > logs/gen_proj_s1_d12.log 2>&1 &
    parser.add_argument("--prediction", type=str, default="v")
    parser.add_argument("--model-name", type=str, default="fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj")
    parser.add_argument("--save-name", type=str, default="fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj")
    parser.add_argument("--num-step", type=int, default=20)
    parser.add_argument("--heun",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zscore",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-depth", type=int, default=12)
    parser.add_argument("--model-head", type=int, default=13)
    parser.add_argument("--save_img", action=argparse.BooleanOptionalAction, default=False)  #! change to False
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--data-path", type=str, default="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data/")

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
