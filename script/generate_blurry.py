import os
import sys
sys.path.append("/home/maiweijian/project/NeuroFlow/script/")
import argparse
import numpy as np
import torch
from torchvision import transforms
from accelerate.utils import set_seed

from dataset import val_nsd_dataloader
from utils import load_mindeye2, mindeyev2_blurry, count_params, requires_grad

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    
    device = "cuda"
    set_seed(args.seed)
    
    subj = f"sub{args.subject}"
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

    
    # Load test dataset
    # train_dataloader = train_nsd_dataloader(args)
    test_dataloader = val_nsd_dataloader(args)

    all_recon_blurry = None
    count_f2i = 0
    for x_fmri, z_clip in test_dataloader:
        with torch.no_grad():
            x_fmri = x_fmri.float().unsqueeze(1).to(device)
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
