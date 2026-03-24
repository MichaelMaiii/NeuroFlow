import sys
sys.path.append('/home/bingxing2/ailab/maiweijian/NeuroFlow-ICLR/script/vae')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from vae_module import NeuroEncoder_V1, NeuroDecoder_V1
from vae_module import *
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D

import torch
import torch.nn as nn
import torch.nn.functional as F

class LRsSeparate(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(LRsSeparate, self).__init__()
        # z0 -> z1
        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Linear(latent_dim, output_dim)
        )
        # z1 -> z0
        self.lin2 = nn.Sequential(
            nn.Linear(output_dim, latent_dim),
            nn.Linear(latent_dim, input_dim)
        )
        
    def sample_bwd(self, z1):
        b, c, d = z1.shape
        pred_z0 = self.lin2(z1.flatten(1))
        
        return pred_z0.view(b,c,d)
        
    def sample(self, z0, z1):
        b, c, d = z0.shape
        
        pred_z1 = self.lin1(z0.flatten(1))
        pred_z0 = self.lin2(z1.flatten(1))

        return pred_z0.view(b,c,d), pred_z1.view(b,c,d)

    def forward_lin1(self, z0):
        b, c, d = z0.shape
        pred_z1 = self.lin1(z0.flatten(1))
        return pred_z1.view(b, c, d)

    def forward_lin2(self, z1):
        b, c, d = z1.shape
        pred_z0 = self.lin2(z1.flatten(1))
        return pred_z0.view(b, c, d)

class LRs(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(LRs, self).__init__()
        # z0 -> z1
        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, latent_dim),  # First linear layer: image → latent
            nn.Linear(latent_dim, output_dim)        # Second linear layer: latent → voxel
        )
        # z1 -> z0
        self.lin2 = nn.Sequential(
            nn.Linear(output_dim, latent_dim),  # First linear layer: image → latent
            nn.Linear(latent_dim, input_dim)        # Second linear layer: latent → voxel
        )

    def sample_bwd(self, z1):
        b, c, d = z1.shape
        pred_z0 = self.lin2(z1.flatten(1))
        
        return pred_z0.view(b,c,d)
        
    def sample(self, z0, z1):
        b, c, d = z0.shape
        
        pred_z1 = self.lin1(z0.flatten(1))
        pred_z0 = self.lin2(z1.flatten(1))

        return pred_z0.view(b,c,d), pred_z1.view(b,c,d)

    def forward(self, z0, z1):
        b, c, d = z0.shape
        # 预测
        pred_z1 = self.lin1(z0.flatten(1))
        pred_z0 = self.lin2(z1.flatten(1))

        # 计算loss
        loss_z1 = F.mse_loss(pred_z1.view(b,c,d), z1)
        loss_z0 = F.mse_loss(pred_z0.view(b,c,d), z0)

        # 总loss
        loss = loss_z1 + loss_z0
        return loss

class GNet(nn.Module):
    def __init__(self, encoder, fwrf):
        super().__init__()
        self.encoder = encoder
        self.fwrf = fwrf
        
    def mse_loss(self, pred, target):
        loss = nn.functional.mse_loss(pred, target, reduction='sum') / target.shape[0]
        return loss

    def forward(self, images, fmri):
        rec, fmaps, h = self.encoder(images)
        pred = self.fwrf(fmaps)
        
        loss = self.mse_loss(pred, fmri)
        return pred, loss
    
class LinearEncodingModel(nn.Module):
    def __init__(self, input_dim=256*1664, latent_dim=2048, voxel_dim=15724):
        super(LinearEncodingModel, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, latent_dim),  # First linear layer: image → latent
            nn.Linear(latent_dim, voxel_dim)        # Second linear layer: latent → voxel
        )
    
    def mse_loss(self, pred, target):
        loss = nn.functional.mse_loss(pred, target, reduction='sum') / target.shape[0]
        return loss
        
    def forward(self, fmri, x):
        # x shape: (batch_size, 257, 768)
        x = x.flatten(1)  # flatten: (batch_size, 257*768)
        pred = self.model(x)        # output shape: (batch_size, voxel_dim)
        
        loss = self.mse_loss(pred, fmri)
        
        return pred, loss
    
class LinearEncodingModel_CLS(nn.Module):
    def __init__(self, input_dim=1280, voxel_dim=15724):
        super(LinearEncodingModel_CLS, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Linear(self.input_dim, voxel_dim)
    
    def mse_loss(self, pred, target):
        loss = nn.functional.mse_loss(pred, target, reduction='sum') / target.shape[0]
        return loss
        
    def forward(self, fmri, x):
        # x shape: (batch_size, 257, 768)
        x = x.flatten(1)  # flatten: (batch_size, 257*768)
        pred = self.model(x)        # output shape: (batch_size, voxel_dim)
        
        loss = self.mse_loss(pred, fmri)
        
        return pred, loss

class BrainNetwork_MLP(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=256*1664, seq_len=1, n_blocks=4, drop=.15, clip_size=1664):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.subj_linear = nn.Linear(in_dim, h, bias=True)
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
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
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def forward(self, x, clip_):
        
        x = self.subj_linear(x).unsqueeze(1)  # h
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        c = self.clip_proj(backbone)
        
        loss = self.soft_clip_loss(c, clip_)
        
        return c, loss
    
class BrainNetwork_Encoder(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=256*1664, seq_len=1, n_blocks=4, drop=.15, clip_size=1664):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.subj_linear = nn.Linear(in_dim, h, bias=True)
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
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
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
    
    def forward(self, x):
        
        x = self.subj_linear(x).unsqueeze(1)  # h
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        c = self.clip_proj(backbone)
        
        return c
    
class BrainNetwork_Decoder(nn.Module):
    def __init__(
        self,
        h=4096,
        in_dim=15724,
        out_dim=256*1664,
        clip_size=1664,
        seq_len=1,
        n_blocks=4,
        drop=0.15
    ):
        super().__init__()

        self.h = h
        self.seq_len = seq_len
        self.clip_size = clip_size
        self.out_dim = out_dim

        # inverse of clip projector
        self.clip_deproj = self.projector(
            in_dim=clip_size,
            out_dim=clip_size,
            h=clip_size
        )

        # inverse of backbone_linear
        self.backbone_deproj = nn.Linear(
            out_dim,
            h * seq_len,
            bias=True
        )

        # Mixer blocks (same structure, shared inductive bias)
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])

        # inverse of subj_linear
        self.subj_deproj = nn.Linear(h, in_dim, bias=True)

    def projector(self, in_dim, out_dim, h):
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

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop)
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)
        )

    def forward(self, z):
        """
        z: (B, 256, 1664)
        """

        B = z.size(0)

        # undo clip projection
        z = self.clip_deproj(z)  # (B, 256, 1664)

        # flatten tokens
        z = z.reshape(B, -1)

        # undo backbone linear
        x = self.backbone_deproj(z)  # (B, h)

        x = x.reshape(B, self.seq_len, self.h)

        # Mixer blocks (reverse order is optional but recommended)
        residual1 = x
        residual2 = x.permute(0, 2, 1)

        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)

            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)

        x = x.squeeze(1)  # (B, h)

        # reconstruct fMRI
        x_hat = self.subj_deproj(x)  # (B, in_dim)

        return x_hat
    

class BrainNetwork_(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=256, n_blocks=4, drop=.15, clip_size=1664):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
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
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c,b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c, b

class BrainEncoder(nn.Module):
    def __init__(self, in_dim=15724, out_dim=256*1664, clip_size=1664, h=2048, n_blocks=2, norm_type='ln', act_first=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size

    def forward(self, x):
        x = self.lin0(x.flatten(1))  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # [b, 256, 1664]
        return x.reshape(len(x), -1, self.clip_size)
    
class BrainDecoder(nn.Module):
    def __init__(self, in_dim=256*1664, out_dim=15724, clip_size=1664, h=2048, n_blocks=2, norm_type='ln', act_first=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.15),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim)

    def forward(self, x):
        x = self.lin0(x.flatten(1))             # [B, h]
        residual = x
        for res_block in range(len(self.mlp)):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = self.lin1(x)             # [B, 15724]
        return x
    
class BrainAutoEncoder(nn.Module):
    def __init__(self, encoder: BrainEncoder, decoder: BrainDecoder, clip_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.clip_weight = clip_weight
        self.clip_weight_flag = False
    
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
    
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, clip_target):
        # 输入 x: [B, 15724]
        clip_latent = self.encode(x)      # [B, 256, 1664]
        recon_fmri = self.decode(clip_latent)  # [B, 15724]
        
        clip_loss = self.soft_clip_loss(clip_latent, clip_target)
        recon_loss = self.mse_loss(recon_fmri, x)
            
        loss = recon_loss + clip_loss * self.clip_weight
        
        return clip_latent, recon_fmri, recon_loss, clip_loss, loss
    
class Encoder(nn.Module):
    def __init__(self, in_dim=15724, h=2048, latent_dim=1280, n_blocks=2):
        super().__init__()

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(0.5)
            )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, latent_dim, bias=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        x = self.lin0(x.flatten(1))  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = self.lin1(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_dim=15724, h=2048, latent_dim=1280, n_blocks=2):
        super().__init__()

        self.lin0 = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(0.15)
            )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        x = self.lin0(x.flatten(1))
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = self.lin1(x)
        return x

# class BrainAE(nn.Module):
#     def __init__(self, encoder, decoder, clip_weight=1000):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.clip_weight = clip_weight
        
#         # self.conv = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
#         # self.lin = nn.Linear(1280, 1664, bias=True)
#         # self.proj= MLP(in_features=1280, hidden_features=2048, out_features=1664)
    
#     def soft_clip_loss(self, preds, targs, temp=0.125):
        
#         preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
#         targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
#         clip_clip = (targs @ targs.T)/temp
#         brain_clip = (preds @ targs.T)/temp
        
#         loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#         loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
#         loss = (loss1 + loss2)/2
#         return loss
    
#     def mse_loss(self, reconstructions, inputs):
#         assert reconstructions.shape == inputs.shape
#         loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
#         return loss
    
#     def encode(self, x):
#         return self.encoder(x)
    
#     def decode(self, x):
#         return self.decoder(x)

#     def forward(self, x, zs, zvf):
#         self.target_length = x.shape[-1]
        
#         z = self.encoder(x) #[b, 1280]
#         recon = self.decoder(z)
    
#         recon_loss = self.mse_loss(recon, x)
#         clip_loss_1 = self.soft_clip_loss(z, zs)
        
#         zp = self.conv(z.unsqueeze(1)) #[b, 256, 1280]
#         zp = self.lin(zp)  #[b, 256, 1664]
#         clip_loss_2 = self.soft_clip_loss(zp, zvf)
        
#         clip_loss = clip_loss_1 + clip_loss_2
        
#         loss = recon_loss + clip_loss * self.clip_weight
        
#         return z, zp, recon, recon_loss, clip_loss, loss
    
class BrainAE(nn.Module):
    def __init__(self, encoder, decoder, clip_weight=1000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.clip_weight = clip_weight
    
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
    
    def mse_loss(self, reconstructions, inputs):
        assert reconstructions.shape == inputs.shape
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, zs):
        self.target_length = x.shape[-1]
        
        z = self.encode(x)
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)

        clip_loss = self.soft_clip_loss(z, zs)
        
        loss = recon_loss + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, clip_loss, loss

# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.norm = nn.LayerNorm(hidden_features)
#         self.gelu = nn.GELU()
#         self.fc2 = nn.Linear(hidden_features, out_features)

#     def forward(self, x):
#         # Assume x is of shape [b, num_channels, sequence_length]
#         b, c, l = x.size()
#         x = x.view(b * c, l)  # Flatten for MLP
#         x = self.fc1(x)
#         x = self.norm(x)
#         x = self.gelu(x)
#         x = self.fc2(x)
#         x = x.view(b, c, -1)  # Reshape back
#         return x

def MLP(input_dim, linear_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, output_dim),
            )



class BrainAE_New(nn.Module):
    def __init__(self, ddconfig, clip_weight=1000, kl_weight=0.001, cycle_weight=1, hidden_dim=1024, linear_dim=512, embed_dim=1664):
        super().__init__()
        self.encoder = Encoder(in_dim=15724, h=2048, latent_dim=4096, n_blocks=2)
        self.decoder = Decoder(out_dim=15724, h=2048, latent_dim=4096, n_blocks=2)
        self.clip_weight = clip_weight
        self.kl_weight = kl_weight
        self.cycle_weight = cycle_weight
        
        self.down_proj = DownProjector(**ddconfig)
        self.up_proj = UpProjector(**ddconfig)
        
        self.kl_proj = MLP(4096, 2048, 4096*2)
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
    
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
    
    def mse_loss(self, reconstructions, inputs):
        assert reconstructions.shape == inputs.shape
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def KLloss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        z = self.encoder(x.flatten(1))  #[b, h]
        x = self.up_proj(z.unsqueeze(1))  #[b, 256, 1024]
        x = self.pre_projector(x) #[b, 256, 1664]
        return x
    
    def decode(self, x):
        x = self.post_projector(x) #[b, 256, 1024]
        x = self.down_proj(x)  #[b, h]
        x = self.decoder(x.squeeze(1)) #[b, 15724]
        return x.unsqueeze(1)

    def forward(self, x, zs):
        self.target_length = x.shape[-1]
        
        #! BrainAE
        z = self.encoder(x) #[b, h]
        moments = self.kl_proj(z)
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        z = posterior.sample()
        
        recon = self.decoder(z)
        recon_loss = self.mse_loss(recon, x)
        
        kl_loss = self.KLloss(posterior, x)
        
        #! Add Projection AE
        z_clip = self.up_proj(z.unsqueeze(1))  #[b, 256, 1024]   
        z_clip = self.pre_projector(z_clip)  #[b, 256, 1664]
        clip_loss = self.soft_clip_loss(z_clip, zs)
        
        z_ = self.post_projector(z_clip)  #[b, 256, 1024]
        z_ = self.down_proj(z_).squeeze(1)  #[b, h]
        proj_loss = self.mse_loss(z_, z)
        
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_clip_recon = self.encode(recon)  #[b, 256, 1664]
            cycle_loss = self.soft_clip_loss(z_clip_recon, zs)
        
        loss = recon_loss + clip_loss * self.clip_weight + proj_loss + cycle_loss * self.cycle_weight + kl_loss * self.kl_weight
        
        return z_clip, recon, z_clip_recon, recon_loss, kl_loss, clip_loss, proj_loss, cycle_loss, loss
    
class BrainVAE(nn.Module):
    def __init__(self, encoder, decoder, clip_weight=1000, kl_weight=0.001, hidden_dim=1664, linear_dim=512, embed_dim=1664):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.clip_weight = clip_weight
        self.kl_weight = kl_weight
        
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
    
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
    
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def KLloss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.encoder(x)  #[b, 256, 1664]
        x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 256, 1664]
        # x = self.decoder(x, target_length) 
        x = self.decoder(x) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[-1]
        
        posterior = self.encode(x)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        if self.kl_weight > 0:
            kl_loss = self.KLloss(posterior, x)
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        clip_loss = self.soft_clip_loss(z, zs)
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, kl_loss, clip_loss, loss
    
    
class BrainVAE_MLP(nn.Module):
    def __init__(self, encoder, decoder, clip_weight=1000, kl_weight=0.001, hidden_dim=1664, linear_dim=512, embed_dim=1664):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.clip_weight = clip_weight
        self.kl_weight = kl_weight
        
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
    
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
    
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def KLloss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.encoder(x)  #[b, 256, 1664]
        x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 256, 1664]
        # x = self.decoder(x, target_length) 
        x = self.decoder(x) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[-1]
        
        posterior = self.encode(x)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        if self.kl_weight > 0:
            kl_loss = self.KLloss(posterior, x)
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        clip_loss = self.soft_clip_loss(z, zs)
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, kl_loss, clip_loss, loss


class NeuroVAE_New(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        # self.pre_projector = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.post_projector = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        self.prior_net = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss(self, posterior, prior, length):
        loss = posterior.kl(prior)  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    # def klloss(self, posterior, length):
    #     loss = posterior.kl()  #与正态分布计算KL损失
    #     loss = loss.sum() / length
    #     return loss
    
    def prior(self, x):
        x = self.prior_net(x)
        x_mean = self.prior_mean(x)
        x_logvar = self.prior_logvar(x)
        moments = torch.cat((x_mean, x_logvar), dim=1)
        prior_ = DiagonalGaussianDistribution_2D(moments)
        return prior_
    
    def encode(self, x, sample=False):
        h, x = self.encoder(x)  #h=[b, 256, 1664], x=[b, 1, 1664]
        if sample:
            # moments = self.pre_projector(x)  #[b, 2, 1664]
            x_mean = self.pre_projector_mean(x)  #[b, 1, 1280]
            x_logvar = self.pre_projector_logvar(x)
            moments = torch.cat((x_mean, x_logvar), dim=1)
            posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
            return h, posterior
        else:
            return h
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 1, 1280] -> [b, 1, 1664]
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        #! input x: [b, 256, 1664] -> output: [b, 1, 15724]
        moments = self.pre_projector(x)  #[b, 2, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        if sample:
            z = posterior.sample()
        else:
            z = posterior.mode()  # [b, 1, 1664]
        z = self.post_projector(z)
        z = self.decoder(z)
        return z 

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, posterior = self.encode(x, sample=True)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            prior = self.prior(zs)
            kl_loss = self.klloss(posterior, prior, zs.shape[0])
            # kl_loss = self.klloss(posterior, zs.shape[0]) # prior distribution p(z)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_recon = self.encode(recon).mode()  #[b, 256, 1664]
            # cycle_mse_loss = self.mse_loss(z_recon, z)
            cycle_mse_loss = torch.tensor(0.0, device=x.device)
            cycle_clip_loss = self.soft_clip_loss(z_recon, zs)
            cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        
        # return z, recon, z_recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss

    
class NeuroVAE(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=4096,
                 linear_dim=2048,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def kl_loss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.encoder(x)  #[b, 256, 4096]
        x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x):
        x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior = self.encode(x)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        # recon = self.decode(z, self.target_length)
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.kl_loss(posterior, x)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_recon = self.encode(recon).mode()  #[b, 256, 1664]
            cycle_loss = self.soft_clip_loss(z_recon, zs)
            # # cycle_mse_loss = self.mse_loss(z_recon, z)
            # cycle_mse_loss = torch.tensor(0.0, device=x.device)
            # cycle_clip_loss = self.soft_clip_loss(z_recon, zs)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        
        # return z, recon, z_recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        return z, z_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_V5(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                #  hidden_dim=4096,
                #  linear_dim=2048,
                #  embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def kl_loss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)  #[b, 256, 4096]
        # x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        # x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        # moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(x)  # 对角高斯采样
        if sample:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        if train:
            return posterior, z
        else:
            return z
    
    def decode(self, x):
        # x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        # x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior, z = self.encode(x, sample_posterior, train=True)  #[b, 256, 1664]
            
        # recon = self.decode(z, self.target_length)
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.kl_loss(posterior, x)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_recon = self.encode(recon).mode()  #[b, 256, 1664]
            cycle_loss = self.soft_clip_loss(z_recon, zs)
            # # cycle_mse_loss = self.mse_loss(z_recon, z)
            # cycle_mse_loss = torch.tensor(0.0, device=x.device)
            # cycle_clip_loss = self.soft_clip_loss(z_recon, zs)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        
        if self.cycle_weight == 0:
            return z, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        else:
            return z, z_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
    
class NeuroVAE_R(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 hidden_dim=4096,
                 linear_dim=2048,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        
        self.clip_weight = clip_weight
        self.pre_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, linear_dim),
                nn.LayerNorm(linear_dim),
                nn.GELU(),
                nn.Linear(linear_dim, linear_dim),
                nn.LayerNorm(linear_dim),
                nn.GELU(),
                nn.Linear(linear_dim, 2*embed_dim)
            )
        
        self.post_projector = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, linear_dim),
                nn.LayerNorm(linear_dim),
                nn.GELU(),
                nn.Linear(linear_dim, linear_dim),
                nn.LayerNorm(linear_dim),
                nn.GELU(),
                nn.Linear(linear_dim, hidden_dim)
            )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def kl_loss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.encoder(x)  #[b, 256, 4096]
        moments = self.pre_projector(x)
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x, target_length) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior = self.encode(x)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.kl_loss(posterior, x)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, kl_loss, clip_loss, loss
    
    
class NeuroVAE_Revise(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 hidden_dim=4096,
                 linear_dim=2048,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.enc_subproj = MLP(15724, 1024, 8192)
        self.dec_subproj = MLP(8192, 1024, 15724)
        
        self.encoder = NeuroEncoder_Revise(**ddconfig)
        self.decoder = NeuroDecoder_Revise(**ddconfig)
        
        self.kl_weight = kl_weight
        
        self.clip_weight = clip_weight
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def kl_loss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.enc_subproj(x)
        x = self.encoder(x)  #[b, 256, 4096]
        x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x, target_length)
        x = self.dec_subproj(x) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]

        posterior = self.encode(x)  #[b, 256, 1664]
        
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.kl_loss(posterior, x)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)
            
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, kl_loss, clip_loss, loss
    
    
class NeuroVAE_V3(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss2(self, posterior, prior, length):
        loss = posterior.kl(prior)  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def klloss1(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def prior(self, x):
        x = self.prior_net(x)
        prior_ = DiagonalGaussianDistribution_2D(x)
        return prior_
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post1 = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h1 = post1.sample() #[b, 256, 1664]
        else:
            h1 = post1.mode()
        
        x = self.pre_proj2(h1)  #[b, 2, 1664]
        post2 = DiagonalGaussianDistribution_2D(x)
        if sample:
            h2 = post2.sample() #[b, 1, 1664]
        else:
            h2 = post2.mode()
            
        if train:
            return h1, h2, post1, post2
        else:
            return h1
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        #! input x: [b, 256, 1664] -> output: [b, 1, 15724]
        # x = self.pre_proj1(x)  #[b, 512, 1664]
        # post1 = DiagonalGaussianDistribution_2D(x) 
        # if sample:
        #     h1 = post1.sample() #[b, 256, 1664]
        # else:
        #     h1 = post1.mode()
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        # x = self.pre_proj2(x)  #[b, 2, 1664]
        # post2 = DiagonalGaussianDistribution_2D(x)
        # if sample:
        #     h2 = post2.sample() #[b, 1, 1664]
        # else:
        #     h2 = post2.mode()
        
        x = self.pre_proj2(x)  #[b, 2, 1664]
        h2, _ = torch.chunk(x, 2, dim=1)       
        x = self.post_proj(h2)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post1 = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h1 = post1.sample() #[b, 256, 1664]
        else:
            h1 = post1.mode()
        return h1

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h1, h2, posterior1, posterior2 = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(h2)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss1 = self.klloss1(posterior1, zs.shape[0]) # prior distribution p(z)
            
            prior = self.prior(zs)
            kl_loss2 = self.klloss2(posterior2, prior, zs.shape[0])
            
            kl_loss = (kl_loss1 + kl_loss2) / 2

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h1, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h1_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h1_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h1, h1_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        return h1, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
#! only use z as [256, 1664]
class NeuroVAE_V0(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = post.sample() #[b, 256, 1664]
        else:
            z = post.mode()
            
        if train:
            return z, post
        else:
            return z
    
    def decode(self, x):
        # x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        # x = self.pre_proj2(x)  #[b, 1, 1664]
        # x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = post.sample() #[b, 256, 1664]
        else:
            z = post.mode()
        return z

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(z_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return z, z_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return z, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
    
#! remove kl loss from z, only used for zc [256, 1664]
class NeuroVAE_V10(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h

    def encode_ft(self, x, sample=False, train=False):
        x = torch.nn.functional.interpolate(x, size=15724, mode='linear', align_corners=True)
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def decode_ft(self, x, voxel_dim):
        x = self.post_proj(x)
        x = self.decoder(x) 
        x = torch.nn.AdaptiveMaxPool1d(voxel_dim)(x)
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x

    def generate_ft(self, x, voxel_dim):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        x = torch.nn.AdaptiveMaxPool1d(voxel_dim)(x)
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
    def forward_ft(self, x, zs, target_length, sample_posterior):
        
        h = torch.nn.functional.interpolate(x, size=target_length, mode='linear', align_corners=True)
        
        h, z, posterior = self.encode(h, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
        
        kl_loss = self.klloss(posterior, zs.shape[0]) 
        clip_loss = self.soft_clip_loss(h, zs)
        h_recon = self.cycle(recon)
        cycle_loss = self.soft_clip_loss(h_recon, zs)
        
        recon = torch.nn.AdaptiveMaxPool1d(x.shape[2])(recon)
        recon_loss = self.mse_loss(recon, x)

        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_V10_Proj(nn.Module):
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
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
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
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
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

    def encode_ft(self, x, sample=False, train=False):
        x = torch.nn.functional.interpolate(x, size=15724, mode='linear', align_corners=True)
        x_ = self.encoder(x)    #[b, 256, 1664]
        
        h_clip = self.clip_proj(x_)
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
    
    def decode_ft(self, x, voxel_dim):
        x = self.post_proj(x)
        x = self.decoder(x) 
        x = torch.nn.AdaptiveMaxPool1d(voxel_dim)(x)
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x

    def generate_ft(self, x, voxel_dim):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        x = torch.nn.AdaptiveMaxPool1d(voxel_dim)(x)
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
    
    def forward_ft(self, x, zs, target_length, sample_posterior):
        
        h = torch.nn.functional.interpolate(x, size=target_length, mode='linear', align_corners=True)
        
        h, h_clip, z, posterior = self.encode(h, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
        
        kl_loss = self.klloss(posterior, zs.shape[0]) 
        clip_loss = self.soft_clip_loss(h, zs) + self.soft_clip_loss(h_clip, zs)
        h_recon, h_clip_recon = self.cycle(recon)
        cycle_loss = self.soft_clip_loss(h_recon, zs) + self.soft_clip_loss(h_clip_recon, zs)
        
        recon = torch.nn.AdaptiveMaxPool1d(x.shape[2])(recon)
        recon_loss = self.mse_loss(recon, x)

        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        return h, h_clip, h_recon, h_clip_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
    
class NeuroEncoder_V10(nn.Module):
    def __init__(self, ddconfig):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss

    def forward(self, x, zs):
        
        z = self.encoder(x)  #[b, 256, 1664]
        loss = self.soft_clip_loss(z, zs)
            
        return z, loss
    
    
class NeuroVAE_newsub(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_newsub(**ddconfig)
        self.decoder = NeuroDecoder_newsub(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
#! remove kl loss from z, only used for zc [256, 1664]
class NeuroVAE_V10_MSE(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            mse_loss = self.mse_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + 0.001 * mse_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, mse_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, mse_loss, cycle_loss, loss
    
class NeuroVAE_V10_woKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        z = self.pre_proj(x)  #[b, 1, 1664]
            
        if train:
            return x, z
        else:
            return x
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        x = self.pre_proj(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        return x

    def forward(self, x, zs, sample_posterior=False):
        self.target_length = x.shape[2]
        
        h, z = self.encode(x, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        kl_loss = torch.tensor(0.0, device=x.device)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            loss = recon_loss + kl_loss+ clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
         
        loss = recon_loss + kl_loss + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
#! change MLP to Linear
class NeuroVAE_V12(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V12(**ddconfig)
        self.decoder = NeuroDecoder_V12(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    

class NeuroVAE_Mindeye(nn.Module):
    def __init__(self,
                 hidden_dim=2048,
                 voxel_dim=15724,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 ):
        super().__init__()
        
        self.encoder = BrainNetwork_Encoder(h=hidden_dim, in_dim=voxel_dim, out_dim=256*1664, 
                         seq_len=1, n_blocks=4, drop=.15, clip_size=1664)
        self.decoder = BrainNetwork_Decoder(h=hidden_dim, in_dim=voxel_dim, out_dim=256*1664, 
                         seq_len=1, n_blocks=4, drop=.15, clip_size=1664)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        assert reconstructions.shape == inputs.shape
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        # self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_Mindeye2(nn.Module):
    def __init__(self,
                 hidden_dim=2048,
                 voxel_dim=15724,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 ):
        super().__init__()
        
        self.encoder = BrainNetwork_Encoder(h=hidden_dim, in_dim=voxel_dim, out_dim=256*1664, 
                         seq_len=1, n_blocks=4, drop=.15, clip_size=1664)
        self.decoder = BrainNetwork_Decoder(h=hidden_dim, in_dim=voxel_dim, out_dim=256*1664, 
                         seq_len=1, n_blocks=4, drop=.15, clip_size=1664)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        assert reconstructions.shape == inputs.shape
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
            
        if train:
            return h, post
        else:
            return h
    
    def decode(self, x):
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        # self.target_length = x.shape[2]
        
        h, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(h)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
#! use sub_proj before conv
class NeuroVAE_V11(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V11(**ddconfig)
        self.decoder = NeuroDecoder_V11(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
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
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        
        z = self.pre_proj2(h)  #[b, 1, 1664]
            
        if train:
            return h, z, post
        else:
            return h
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x):
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        x = self.pre_proj2(x)  #[b, 1, 1664]
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h = post.sample() #[b, 256, 1664]
        else:
            h = post.mode()
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h, z, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss(posterior, zs.shape[0]) 

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    

class NeuroVAE_V6(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        self.up_net = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.down_net = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        
        # self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.pre_proj2 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    # def klloss2(self, posterior, prior, length):
    #     loss = posterior.kl(prior)  #与正态分布计算KL损失
    #     loss = loss.sum() / length
    #     return loss
    
    def klloss1(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    # def prior(self, x):
    #     x = self.prior_net(x)
    #     prior_ = DiagonalGaussianDistribution_2D(x)
    #     return prior_
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 2, 1664]
        posterior = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = posterior.sample() #[b, 256, 1664]
        else:
            z = posterior.mode()
            
        h = self.up_net(z)
            
        if train:
            return posterior, z, h
        else:
            return h
    
    def decode(self, x):
        # x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        x = self.down_net(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = post.sample() #[b, 256, 1664]
        else:
            z = post.mode()
        h = self.up_net(z)  #[b, 512, 1664]
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior, z, h = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        z_ = self.down_net(h)
        h_loss = self.mse_loss(z_, z)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss1(posterior, zs.shape[0]) # prior distribution p(z)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h1_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h1_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h1_recon, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_V7(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        self.up_net = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.down_net = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        
        # self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.pre_proj2 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    # def klloss2(self, posterior, prior, length):
    #     loss = posterior.kl(prior)  #与正态分布计算KL损失
    #     loss = loss.sum() / length
    #     return loss
    
    def klloss1(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    # def prior(self, x):
    #     x = self.prior_net(x)
    #     prior_ = DiagonalGaussianDistribution_2D(x)
    #     return prior_
    
    def encode(self, x, sample=False, train=False):
        c, x = self.encoder(x)    #[b, 2, 1664]
        posterior = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = posterior.sample() #[b, 256, 1664]
        else:
            z = posterior.mode()
            
        h = self.up_net(z)
            
        if train:
            return posterior, c, z, h
        else:
            return h
    
    def decode(self, x):
        # x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        x = self.down_net(x)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = post.sample() #[b, 256, 1664]
        else:
            z = post.mode()
        h = self.up_net(z)  #[b, 512, 1664]
        return h

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior, c, z, h = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        z_ = self.down_net(h)
        h_loss = self.mse_loss(z_, z)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss1(posterior, zs.shape[0]) # prior distribution p(z)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs) + self.soft_clip_loss(c, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h1_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h1_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h1_recon, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_V8(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1280,
                 linear_dim=1024,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        self.up_net = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.up_linear = nn.Linear(hidden_dim, embed_dim)
        
        self.down_net = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.down_linear = nn.Linear(embed_dim, hidden_dim)
        
        # self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.pre_proj2 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    # def klloss2(self, posterior, prior, length):
    #     loss = posterior.kl(prior)  #与正态分布计算KL损失
    #     loss = loss.sum() / length
    #     return loss
    
    def klloss1(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    # def prior(self, x):
    #     x = self.prior_net(x)
    #     prior_ = DiagonalGaussianDistribution_2D(x)
    #     return prior_
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 2, 1664]
        posterior = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = posterior.sample() #[b, 256, 1664]
        else:
            z = posterior.mode()
            
        h = self.up_net(z) #[256, 1280]
        h = self.up_linear(h) #[256, 1664]
            
        if train:
            return posterior, z, h
        else:
            return h
    
    def decode(self, x):
        # x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        x = self.down_linear(x) #[256, 1280]
        x = self.down_net(x) #[1, 1280]
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        post = DiagonalGaussianDistribution_2D(x) 
        if sample:
            z = post.sample() #[b, 256, 1664]
        else:
            z = post.mode()
        h = self.up_net(z)  #[b, 512, 1664]
        return h

    def forward(self, x, zs, zp, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior, z, h = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(z)
    
        recon_loss = self.mse_loss(recon, x)
        
        z_ = self.down_linear(h)
        z_ = self.down_net(z_)
        h_loss = self.mse_loss(z_, z)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss = self.klloss1(posterior, zs.shape[0]) # prior distribution p(z)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h, zs) + self.soft_clip_loss(z, zp)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h1_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h1_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h, h1_recon, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
        
        loss = recon_loss + h_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        return h, recon, recon_loss, h_loss, kl_loss, clip_loss, cycle_loss, loss
    
    
class NeuroVAE_V4(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V1(**ddconfig)
        self.decoder = NeuroDecoder_V1(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        # self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
        self.pre_proj1 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pre_proj2 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        self.post_proj = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        
        # self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        # self.prior_mean = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        # self.prior_net = nn.Sequential(
        #     nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0),
        #     MLP(hidden_dim, linear_dim, embed_dim)
        #     )
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss2(self, posterior, prior, length):
        loss = posterior.kl(prior)  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def klloss1(self, posterior, length):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def prior(self, x):
        x = self.prior_net(x)
        prior_ = DiagonalGaussianDistribution_2D(x)
        return prior_
    
    def encode(self, x, sample=False, train=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post1 = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h1 = post1.sample() #[b, 256, 1664]
        else:
            h1 = post1.mode()
        
        x = self.pre_proj2(h1)  #[b, 2, 1664]
        post2 = DiagonalGaussianDistribution_2D(x)
        if sample:
            h2 = post2.sample() #[b, 1, 1664]
        else:
            h2 = post2.mode()
            
        if train:
            return h1, h2, post1, post2
        else:
            return h1
    
    def decode(self, x):
        x = self.post_proj(x)
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        #! input x: [b, 256, 1664] -> output: [b, 1, 15724]
        # x = self.pre_proj1(x)  #[b, 512, 1664]
        # post1 = DiagonalGaussianDistribution_2D(x) 
        # if sample:
        #     h1 = post1.sample() #[b, 256, 1664]
        # else:
        #     h1 = post1.mode()
        #! 这里x是sample后的[256, 1664],不需要再进行sample
        # x = self.pre_proj2(x)  #[b, 2, 1664]
        # post2 = DiagonalGaussianDistribution_2D(x)
        # if sample:
        #     h2 = post2.sample() #[b, 1, 1664]
        # else:
        #     h2 = post2.mode()
        
        x = self.pre_proj2(x)  #[b, 2, 1664]
        h2, _ = torch.chunk(x, 2, dim=1)       
        x = self.post_proj(h2)
        x = self.decoder(x) 
        return x
    
    def cycle(self, x, sample=False):
        x = self.encoder(x)    #[b, 256, 1664]
        x = self.pre_proj1(x)  #[b, 512, 1664]
        post1 = DiagonalGaussianDistribution_2D(x) 
        if sample:
            h1 = post1.sample() #[b, 256, 1664]
        else:
            h1 = post1.mode()
        return h1

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        h1, h2, posterior1, posterior2 = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(h2)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            kl_loss1 = self.klloss1(posterior1, zs.shape[0]) # prior distribution p(z)
            
            prior = self.prior(zs)
            kl_loss2 = self.klloss2(posterior2, prior, zs.shape[0])
            
            kl_loss = (kl_loss1 + kl_loss2) / 2

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h1, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            h1_recon = self.cycle(recon)
            cycle_loss = self.soft_clip_loss(h1_recon, zs)
            # cycle_loss = self.mse_loss(h1_recon, h1)
            # cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
            loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
            return h1, h1_recon, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        
        return h1, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
    
# class NeuroVAE_V1(nn.Module):
#     def __init__(self,
#                  ddconfig,
#                  clip_weight=1,
#                  kl_weight=1,
#                  cycle_weight=1,
#                  hidden_dim=1664,
#                  linear_dim=1024,
#                  embed_dim=1280,
#                  ):
#         super().__init__()
        
#         self.encoder = NeuroEncoder_V1(**ddconfig)
#         self.decoder = NeuroDecoder_V1(**ddconfig)
        
#         self.kl_weight = kl_weight
#         self.clip_weight = clip_weight
#         self.cycle_weight = cycle_weight
        
#         self.pre_projector = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
#         self.post_projector = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
#         self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
            
#     def soft_clip_loss(self, preds, targs, temp=0.125):
        
#         preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
#         targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
#         clip_clip = (targs @ targs.T)/temp
#         brain_clip = (preds @ targs.T)/temp
        
#         loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#         loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
#         loss = (loss1 + loss2)/2
#         return loss
        
#     def mse_loss(self, reconstructions, inputs):
#         loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
#         return loss
    
#     def klloss(self, posterior, prior, length):
#         loss = posterior.kl(prior)  #与正态分布计算KL损失
#         loss = loss.sum() / length
#         return loss
    
#     def prior(self, x):
#         x = self.prior_net(x)
#         prior_ = DiagonalGaussianDistribution_2D(x)
#         return prior_
    
#     def encode(self, x, sample=False, train=False):
#         h1 = self.encoder(x)  # h=[b, 256, 1664], 
#         x = self.pre_projector(h1) # x=[b, 2, 1664]
#         posterior = DiagonalGaussianDistribution_2D(x)  # 对角高斯采样
#         if sample:
#             x = posterior.sample()
#         else:
#             x = posterior.mode()
            
#         h2 = self.post_projector(x)  # h2=[b, 256, 1664]
        
#         if train:
#             return h1, h2, posterior
#         else:
#             return h2
    
#     def decode(self, x):
#         x = self.decoder(x) 
#         return x
    
#     def generate(self, x, sample=False):
#         #! input x: [b, 256, 1664] -> output: [b, 1, 15724]
#         z = self.decoder(x) #使用decoder部分的256x1664作为fm的分布，只需经过decoder
#         return z 

#     def forward(self, x, zs, sample_posterior):
        
#         h1, h2, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
#         recon = self.decode(h2)
    
#         recon_loss = self.mse_loss(recon, x)
        
#         if self.kl_weight == 0:
#             kl_loss = torch.tensor(0.0, device=x.device)
#         else:
#             prior = self.prior(zs)
#             kl_loss = self.klloss(posterior, prior, zs.shape[0])
#             # kl_loss = self.klloss(posterior, zs.shape[0]) # prior distribution p(z)

#         if self.clip_weight == 0:
#             clip_loss = torch.tensor(0.0, device=x.device)
#         else:
#             clip_loss = self.soft_clip_loss(h1, zs) + self.soft_clip_loss(h2, zs)
            
#         if self.cycle_weight == 0:
#             cycle_loss = torch.tensor(0.0, device=x.device)
#         else:
#             z_recon = self.encode(recon).mode()  #[b, 256, 1664]
#             # cycle_mse_loss = self.mse_loss(z_recon, z)
#             cycle_mse_loss = torch.tensor(0.0, device=x.device)
#             cycle_clip_loss = self.soft_clip_loss(z_recon, zs)
#             cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
        
#         loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        
#         # return z, recon, z_recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
#         return h1, h2, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
    
class NeuroVAE_V2(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 cycle_weight=1,
                 hidden_dim=1664,
                 linear_dim=1024,
                 embed_dim=1280,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder_V2(**ddconfig)
        self.decoder = NeuroDecoder_V2(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.cycle_weight = cycle_weight
        
        self.pre_projector = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
        self.post_projector = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.prior_net = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def klloss(self, posterior, prior, length):
        loss = posterior.kl(prior)  #与正态分布计算KL损失
        loss = loss.sum() / length
        return loss
    
    def prior(self, x):
        x = self.prior_net(x)
        prior_ = DiagonalGaussianDistribution_2D(x)
        return prior_
    
    def encode(self, x, sample=False, train=False):
        h1 = self.encoder(x)  # h=[b, 256, 1664], 
        x = self.pre_projector(h1) # x=[b, 2, 1664]
        posterior = DiagonalGaussianDistribution_2D(x)  # 对角高斯采样
        if sample:
            x = posterior.sample()
        else:
            x = posterior.mode()
            
        h2 = self.post_projector(x)  # h2=[b, 256, 1664]
        
        if train:
            return h1, h2, posterior
        else:
            return h2
    
    def decode(self, x):
        x = self.decoder(x) 
        return x
    
    def generate(self, x, sample=False):
        #! input x: [b, 256, 1664] -> output: [b, 1, 15724]
        z = self.decoder(x) #使用decoder部分的256x1664作为fm的分布，只需经过decoder
        return z 

    def forward(self, x, zs, sample_posterior):
        
        h1, h2, posterior = self.encode(x, sample=sample_posterior, train=True)  #[b, 256, 1664]
            
        recon = self.decode(h2)
    
        recon_loss = self.mse_loss(recon, x)
        
        if self.kl_weight == 0:
            kl_loss = torch.tensor(0.0, device=x.device)
        else:
            prior = self.prior(zs)
            kl_loss = self.klloss(posterior, prior, zs.shape[0])
            # kl_loss = self.klloss(posterior, zs.shape[0]) # prior distribution p(z)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(h1, zs) + self.soft_clip_loss(h2, zs)
            
        if self.cycle_weight == 0:
            cycle_loss = torch.tensor(0.0, device=x.device)
        else:
            z_recon = self.encode(recon).mode()  #[b, 256, 1664]
            # cycle_mse_loss = self.mse_loss(z_recon, z)
            cycle_mse_loss = torch.tensor(0.0, device=x.device)
            cycle_clip_loss = self.soft_clip_loss(z_recon, zs)
            cycle_loss = cycle_mse_loss * 0.01 + cycle_clip_loss * self.clip_weight
        
        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight + cycle_loss * self.cycle_weight
        
        # return z, recon, z_recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss
        return h1, h2, recon, recon_loss, kl_loss, clip_loss, cycle_loss, loss