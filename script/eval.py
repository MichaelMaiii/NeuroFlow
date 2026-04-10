import argparse
import logging
import os
import sys
sys.path.append("/home/maiweijian/project/NeuroFlow")
sys.path.append("/home/maiweijian/project/NeuroFlow/script")
sys.path.append("/home/maiweijian/project/NeuroFlow/script/eval")

import clip
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import pearsonr
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights,
                                Inception_V3_Weights, alexnet, efficientnet_b1,
                                inception_v3)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from mind_utils import batchwise_cosine_similarity, topk

def get_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Configuration")
    
    parser.add_argument("--model_name", type=str, default="demo")
    parser.add_argument("--subj", type=int, default=1)
    
    return parser.parse_args()


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def calculate_retrival_percent_correct(all_image_voxels, all_clip_voxels):
    
    fwd_percent_correct = []
    bwd_percent_correct = []
    rng = np.random.default_rng(0)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for _ in tqdm(range(30)):
            random_samps = rng.choice(np.arange(len(all_image_voxels)), size=300, replace=False)
            i_emb = all_image_voxels[random_samps].to(device).float()  # CLIP-image
            b_emb = all_clip_voxels[random_samps].to(device).float()                  # CLIP-brain

            # flatten if necessary
            i_emb = i_emb.reshape(len(i_emb), -1)
            b_emb = b_emb.reshape(len(b_emb), -1)

            # l2norm
            i_emb = nn.functional.normalize(i_emb, dim=-1)
            b_emb = nn.functional.normalize(b_emb, dim=-1)

            labels = torch.arange(len(i_emb)).to(device)
            fwd_sim = batchwise_cosine_similarity(b_emb, i_emb)  # brain, clip
            bwd_sim = batchwise_cosine_similarity(i_emb, b_emb)  # clip, brain

            fwd_percent_correct.append(topk(fwd_sim, labels, k=1).item())
            bwd_percent_correct.append(topk(bwd_sim, labels, k=1).item())

    mean_fwd_percent_correct = np.mean(fwd_percent_correct)
    mean_bwd_percent_correct = np.mean(bwd_percent_correct)

    fwd_sd = np.std(fwd_percent_correct) / np.sqrt(len(fwd_percent_correct))
    fwd_ci = stats.norm.interval(0.95, loc=mean_fwd_percent_correct, scale=fwd_sd)

    bwd_sd = np.std(bwd_percent_correct) / np.sqrt(len(bwd_percent_correct))
    bwd_ci = stats.norm.interval(0.95, loc=mean_bwd_percent_correct, scale=bwd_sd)

    print(f"fwd percent_correct: {mean_fwd_percent_correct:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
    print(f"bwd percent_correct: {mean_bwd_percent_correct:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")

    return mean_fwd_percent_correct, mean_bwd_percent_correct

    
def evaluate_voxel_metrics(all_recon, all_fmri):
    """
    Evaluate voxel-level (MSE, Pearson) and structural-level (CKA, Cosine Similarity) metrics.
    
    Args:
        all_recon: Tensor of shape [N, 1, D]
        all_fmri: Tensor of shape [N, 1, D]
    
    Returns:
        dict with keys: "MSE", "Pearson", "CKA", "Cosine"
    """
    all_recon = all_recon.squeeze(1)  # [N, D]
    N, D = all_recon.shape

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

    # Cosine Similarity (averaged per sample)
    recon_np = recon_flat.cpu().numpy()
    target_np = target_flat.cpu().numpy()
    cos_sim = np.mean([
        cosine_similarity(recon_np[i:i+1], target_np[i:i+1])[0, 0]
        for i in range(recon_np.shape[0])
    ])
    
    return {
        "MSE": np.mean(mse_vals),
        "Pearson": np.mean(pearson_vals),
        "Cosine": cos_sim,
    }


def calculate_pixcorr(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # flatten images while keeping the batch dimension
    gt_flat = preprocess(gt).reshape(len(gt), -1).cpu()
    pd_flat = preprocess(pd).reshape(len(pd), -1).cpu()
    logger.debug(f"gt_flat shape: {gt_flat.shape}")
    logger.debug(f"pd_flat shape: {pd_flat.shape}")

    print("image flattened, now calculating pixcorr...")
    pixcorr_score = []
    for gt, pd in tqdm(zip(gt_flat, pd_flat), total=len(gt_flat)):
        pixcorr_score.append(np.corrcoef(gt, pd)[0,1])
    
    return np.mean(pixcorr_score)


# see https://github.com/zijin-gu/meshconv-decoding/issues/3
def calculate_ssim(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    gt_gray = rgb2gray(preprocess(gt).permute((0,2,3,1)).cpu())
    pd_gray = rgb2gray(preprocess(pd).permute((0,2,3,1)).cpu())
    logger.debug(f"gt_gray shape: {gt_gray.shape}")
    logger.debug(f"pd_gray shape: {pd_gray.shape}")
    
    print("image converted to grayscale, now calculating ssim...")
    ssim_score = []
    for gt, pd in tqdm(zip(gt_gray, pd_gray), total=len(gt_gray)):
        ssim_score.append(ssim(gt, pd, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    return np.mean(ssim_score)


def get_model(model_name):
    if model_name == 'Incep':
        weights = Inception_V3_Weights.DEFAULT
        model = create_feature_extractor(inception_v3(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'CLIP':
        # 默认下载地址 ~/.cache/clip
        model, _ = clip.load("ViT-L/14", device=device)
        return model.encode_image
    elif model_name == 'Eff':
        weights = EfficientNet_B1_Weights.DEFAULT
        model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'SwAV':
        model = torch.hub.load('/home/maiweijian/.cache/torch/hub/facebookresearch-swav-06b1b7c', 
                               'resnet50', source='local')
        model = create_feature_extractor(model, return_nodes=['avgpool'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device).eval().requires_grad_(False)


def get_preprocess(model_name):
    if model_name == 'Incep':
        return transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'CLIP':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    elif model_name == 'Eff':
        return transforms.Compose([
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'SwAV':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def process_in_batches(images, model, preprocess, layer, batch_size):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = preprocess(batch).to(device)
        with torch.no_grad():
            if layer is None:
                feat = model(batch).float().flatten(1)
            else:
                feat = model(batch)[layer].float().flatten(1)
            feats.append(feat)
    return torch.cat(feats, dim=0).cpu().numpy()


def two_way_identification(gt, pd, return_avg=True):
    num_samples = len(gt)
    corr_mat = np.corrcoef(gt, pd)                   # compute correlation matrix
    corr_mat = corr_mat[:num_samples, num_samples:]  # extract relevant quadrant of correlation matrix
    
    congruent = np.diag(corr_mat)
    success = corr_mat < congruent
    success_cnt = np.sum(success, axis=0)

    if return_avg:
        return np.mean(success_cnt) / (num_samples - 1)
    else:
        return success_cnt, num_samples - 1


def calculate_metric(model_name, gt, pd, model, preprocess, layer, batch_size):
    gt = process_in_batches(gt, model, preprocess, layer, batch_size)
    pd = process_in_batches(pd, model, preprocess, layer, batch_size)

    if model_name in ['Alex', 'Incep', 'CLIP']:
        return two_way_identification(gt, pd)
    elif model_name in ['Eff', 'SwAV']:
        return np.array([sp.spatial.distance.correlation(gt[i], pd[i]) for i in range(len(gt))]).mean()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def print_results(results):
    """
    Print formatted results with proper alignment
    """
    print(f"{'PixCorr:':<20} {results['PixCorr']:>10.4f}")
    print(f"{'SSIM:':<20} {results['SSIM']:>10.4f}")
    # 百分比格式输出（×100 + 保留2位小数 + 追加%符号）
    print(f"{'Incep:':<20} {results['Incep_avgpool'] * 100:>10.2f}% (2-way percent correct)")
    print(f"{'CLIP:':<20} {results['CLIP_None'] * 100:>10.2f}% (2-way percent correct)")
    print(f"{'Eff:':<20} {results['Eff_avgpool']:>10.4f}")
    print(f"{'SwAV:':<20} {results['SwAV_avgpool']:>10.4f}")


def print_retrieval(results):
    print(f"{'fwd_percent_correct:':<20} {results['fwd_percent_correct'] * 100:>10.2f}")
    print(f"{'fwd_percent_correct_recon:':<20} {results['fwd_percent_correct_recon'] * 100:>10.2f}")


def _resize_tensors(size, *tensors):
    resize_transform = transforms.Resize((size, size))
    return tuple(resize_transform(tensor).float() for tensor in tensors)

def compute(all_images, all_recons):
    results = {}
    
    imsize = 256
    all_images, all_recons = _resize_tensors(imsize, all_images, all_recons)

    results['PixCorr'] = calculate_pixcorr(all_images, all_recons)
    print(f"PixCorr: {results['PixCorr']:.6f}\n")
    
    results['SSIM'] = calculate_ssim(all_images, all_recons)
    print(f"SSIM: {results['SSIM']:.6f}\n")

    net_list = [
        ('Incep', 'avgpool'),
        ('CLIP', None),  # final layer
        ('Eff', 'avgpool'),
        ('SwAV', 'avgpool'),
    ]

    batch_size = 32
    for model_name, layer in net_list:
        logger.info(f"calculating {model_name} with layer {layer}...")
        
        model = get_model(model_name)
        preprocess = get_preprocess(model_name)

        results[f"{model_name}_{layer}"] = calculate_metric(model_name, all_images, all_recons, model, preprocess, layer, batch_size)
        logger.info(f"{model_name}({layer}): {results[f'{model_name}_{layer}']:.6f}")

        del model
        torch.cuda.empty_cache()
    
    return results


logger = setup_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = get_args() 

def main():
    sub = 1
    args.gt_path = "/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/BrainSyn/evals"
    args.eval_path = "/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow/evals"

    args.model_name = f"fm-s{sub}-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub{sub}"
    args.setting_name = f"single_s{sub}"
    
    print(args)

    all_images = torch.load("/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data/nsd/subj01/all_images.pt")
    all_recon_img = torch.load(f"{args.eval_path}/{args.model_name}/{args.setting_name}_all_recon_img.pt")
    all_recon_f2i = torch.load(f"{args.eval_path}/{args.model_name}/{args.setting_name}_all_recon_f2i.pt")
    all_recon_fmri = torch.load(f"{args.eval_path}/{args.model_name}/{args.setting_name}_all_recon_fmri.pt")
    
    all_image_voxels = torch.load(f"{args.gt_path}/all_clipvoxels.pt")
    all_zfmri_raw = torch.load(f"{args.eval_path}/{args.model_name}/{args.setting_name}_all_zfmri_raw.pt")
    all_zfmri_recon = torch.load(f"{args.eval_path}/{args.model_name}/{args.setting_name}_all_zfmri_syn.pt")
    
    all_recon_blurry = torch.load(f"{args.eval_path}/fm-s{sub}-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse/sub{sub}/single_s{sub}_all_recon_blurry.pt")

    all_recon_merge = all_recon_img * 0.65 + all_recon_blurry * 0.35
    
    data_path = "/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data/" 
    all_fmri = np.load(os.path.join(data_path, f'nsd/subj0{sub}/nsd_train_fmri_scale_sub{sub}.npy')).astype(np.float32)
    all_fmri = all_fmri.mean(axis=1)
    print(np.mean(all_fmri), np.std(all_fmri))
    all_fmri = torch.tensor(all_fmri, dtype=torch.float32).squeeze(1)

    voxel_mean = all_fmri.mean(dim=0)   # [D]
    voxel_std  = all_fmri.std(dim=0)    # [D]
    all_recon_scale = all_recon_fmri.squeeze(1) * voxel_std + voxel_mean
    
    all_test_fmri = np.load(os.path.join(data_path, f'nsd/subj0{sub}/nsd_test_fmri_scale_sub{sub}.npy')).astype(np.float32)
    all_test_fmri = all_test_fmri.mean(axis=1)
    all_test_fmri = torch.tensor(all_test_fmri, dtype=torch.float32).squeeze(1)
    
    metrics = evaluate_voxel_metrics(all_recon_scale, all_test_fmri)
    
    results_dec = compute(all_images, all_recon_merge)
    results_enc = compute(all_images, all_recon_f2i)
    
    results = {}
    fwd_percent_correct, _ = calculate_retrival_percent_correct(all_image_voxels, all_zfmri_raw)
    results['fwd_percent_correct'] = fwd_percent_correct
    
    fwd_percent_correct_recon, _ = calculate_retrival_percent_correct(all_image_voxels, all_zfmri_recon)
    results['fwd_percent_correct_recon'] = fwd_percent_correct_recon
    
    print("************************* Decoding *********************************")
    print_results(results_dec)

    print("**************************** Encoding -> Decoding ******************************")
    print_results(results_enc)
    
    print("**************************** Retrieval ******************************")
    print_retrieval(results)
    
    print("******************************* Encoding ***************************************")
    print(metrics)


if __name__ == "__main__":
    main()
