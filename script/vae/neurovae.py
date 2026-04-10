import sys
sys.path.append('/home/bingxing2/ailab/maiweijian/NeuroFlow/script/vae')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from vae_module import NeuroEncoder, NeuroDecoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class NeuroVAE_P(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 clip_size=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)
        targs = F.normalize(targs.flatten(1), dim=-1)
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss(self, posterior, length):
        loss = posterior.kl() 
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x_ = self.encoder(x)    #[b, 256, 1664]
        
        h_clip = self.clip_proj(x_) #! retrieval submodule
        
        x = self.pre_proj1(x_)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, h_clip, z, post
        else:
            return h, h_clip
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x_ = self.encoder(x)    #[b, 256, 1664]
        h_clip = self.clip_proj(x_)
        
        x = self.pre_proj1(x_)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h, h_clip

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, h_clip, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs) + self.soft_clip_loss(h_clip, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon, h_clip_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs) + self.soft_clip_loss(h_clip_recon, zs)
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_clip, h_recon, h_clip_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss