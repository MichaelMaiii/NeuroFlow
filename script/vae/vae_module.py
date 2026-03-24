import sys
import torch
# import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import math
import torch.nn as nn
import numpy as np
from einops import rearrange

# from ldm.modules.attention import LinearAttention

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    if in_channels >= 32:
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
    
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # 使用 Conv1d 代替 Conv2d
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3)

    def forward(self, x):
        if self.with_conv:
            # 在 Conv1d 中，我们只需要对时间维度 l 进行填充
            # pad = (1, 0)  # 仅在左侧填充
            # x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 使用 avg_pool1d 代替 avg_pool2d
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x

    
# class Upsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             self.conv = torch.nn.Conv1d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="linear")
#         # x = torch.nn.functional.interpolate(x, size=self.target_size, mode='linear', align_corners=True)
#         if self.with_conv:
#             x = self.conv(x)
#         return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        # 使用 ConvTranspose1d 反卷积
        self.trans_conv = torch.nn.ConvTranspose1d(in_channels,
                                                   in_channels,
                                                   kernel_size=7,
                                                   stride=2,
                                                   padding=3,
                                                   output_padding=1)

    def forward(self, x):
        # 反卷积操作，恢复尺寸
        x = self.trans_conv(x)
        return x
    
class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:   #处理temporal embedding
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:  #处理维度不一致，残差相加
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

# 修改LinAttnBlock
class LinAttnBlock(nn.Module):
    """A block implementing linear attention."""
    def __init__(self, in_channels):
        super().__init__()
        self.heads = 1
        dim_head = in_channels
        hidden_dim = dim_head * self.heads
        self.to_qkv = nn.Conv1d(in_channels, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, in_channels, 1)

    def forward(self, x):
        # 输入 x 的形状为 [b, c, l]
        b, c, l = x.shape

        # 计算 q, k, v
        qkv = self.to_qkv(x)  # 形状为 [b, 3 * heads * dim_head, l]
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', heads=self.heads, qkv=3)

        # 对 k 进行 softmax
        k = k.softmax(dim=-1)

        # 使用爱因斯坦求和计算 context = k @ v
        context = torch.einsum('bhdn,bhen->bhde', k, v)

        # 使用爱因斯坦求和计算 out = context @ q
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        # 重塑回原始维度
        out = rearrange(out, 'b heads c l -> b (heads c) l')

        # 通过输出卷积层
        return self.to_out(out)


# 修改注意力Block
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 输入 x 的形状为 [b, c, l]
        h_ = self.norm(x)

        # 计算 q, k, v，形状均为 [b, c, l]
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取维度
        b, c, l = q.shape

        # 变换 q 的形状为 [b, l, c]，方便与 k 进行矩阵乘法
        q = q.permute(0, 2, 1)  # [b, l, c]
        # k 的形状保持为 [b, c, l]
        # 计算注意力矩阵 w_，形状为 [b, l, l]
        w_ = torch.bmm(q, k)  # [b, l, l]
        w_ = w_ * (c ** -0.5)  # 进行缩放
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 在最后一个维度上进行 softmax

        # 将注意力矩阵 w_ 应用到 v 上
        h_ = torch.bmm(v, w_.permute(0, 2, 1))  # [b, c, l]

        # 通过输出投影层
        h_ = self.proj_out(h_)

        # 残差连接
        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
    
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, ch=128, out_channels=1, target_size=8192):
        super(UNet1D, self).__init__()
        
        # Downsampling layers
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, ch // 4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch // 4, ch // 2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch // 2, ch, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )
        
        # Adaptive MaxPooling to map to a fixed size (e.g., [b, ch, 8192])
        self.ada_maxpool = nn.AdaptiveMaxPool1d(target_size)
        
        # Upsampling layers
        self.conv_out = nn.Sequential(
            nn.Conv1d(ch, ch // 2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch // 2, ch // 4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch // 4, out_channels, kernel_size=7, stride=1, padding=3),
        )
    
    def forward(self, x):
        # Downsampling
        x = self.conv_in(x)
        
        # Adaptive MaxPooling
        x = self.ada_maxpool(x)
        
        # Upsampling
        x = self.conv_out(x)
        
        return x
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # Assume x is of shape [b, num_channels, sequence_length]
        b, c, l = x.size()
        x = x.view(b * c, l)  # Flatten for MLP
        x = self.fc1(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = x.view(b, c, -1)  # Reshape back
        return x

def mlp_proj(input_dim, linear_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, output_dim),
            )

class UNet1DWithAttention(nn.Module):
    def __init__(self, in_channels=1, ch=128, out_channels=1, target_size=8192, num_heads=8):
        super(UNet1DWithAttention, self).__init__()
        
        self.resnet_in = nn.Sequential(
            ResNetBlock(in_channels, ch // 2),
            ResNetBlock(ch // 2, ch)
        )
        # Adaptive MaxPooling to map to a fixed size (e.g., [b, ch, 8192])
        self.ada_maxpool = nn.AdaptiveMaxPool1d(target_size)
        self.resnet_out = nn.Sequential(
            ResNetBlock(ch, 256),
            ResNetBlock(256, 197),
        )
        self.mlp = MLP(in_features=8192, hidden_features=4096, out_features=768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, batch_first=True) #[b, 197, 8192]
        # self.attention = AttnBlock(197)
    
    def forward(self, x):
        x = self.resnet_in(x)
        x = self.ada_maxpool(x)
        x = self.resnet_out(x)
        x = self.mlp(x)
        # x = x.permute(0, 2, 1) # Apply attention (requires permuting dimensions to [b, seq_len, embed_dim])
        x, _ = self.attention(x, x, x)
        return x

    
class NeuroEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", 
                 output_tokens=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        self.output_tokens = output_tokens
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        self.subj_proj= MLP(in_features=15724, hidden_features=linear_dim, out_features=latent_dim)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        out = self.conv_out(h)
        
        if self.output_tokens:
            return h, out
        else:
            return out


class NeuroDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False, output_tokens=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks
        self.output_tokens = output_tokens

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.subj_proj= MLP(in_features=latent_dim, hidden_features=linear_dim, out_features=15724)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        # if self.output_tokens:
        #     h = z
        # else:
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        # h = self.resnet_out(h)  ##! add resblock layer here
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    

class NeuroEncoder_Revise(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), ada_length, num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        # self.subj_proj= MLP(in_features=15724, hidden_features=1024, out_features=8192)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        # x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)   # [b, 4*2, 1024]
        return h


class NeuroDecoder_Revise(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # self.subj_proj= MLP(in_features=8192, hidden_features=1024, out_features=15724)

    def forward(self, z, target_length):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        # h = self.resnet_out(h)  ##! add resblock layer here
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        # h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    

#! from [1, 4096] to [256, 1664]
class UpProjector(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), ada_length, num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0) #[b, 128, h]
        # self.subj_proj= MLP(in_features=15724, hidden_features=linear_dim, out_features=8192)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        # x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)   # [b, 256, h//num_down_blocks]
        return h


class DownProjector(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # self.subj_proj= MLP(in_features=8192, hidden_features=linear_dim, out_features=15724)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        # h = self.resnet_out(h)  ##! add resblock layer here
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        # h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! [b, 1, h]
        
        return h
    
class NeuroEncoder_New(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), ada_length, num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        self.subj_proj= MLP(in_features=15724, hidden_features=linear_dim, out_features=8192)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)   # [b, 4*2, 1024]
        return h


class NeuroDecoder_New(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.subj_proj= MLP(in_features=8192, hidden_features=linear_dim, out_features=15724)

    def forward(self, z, target_length):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        # h = self.resnet_out(h)  ##! add resblock layer here
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    
    
class NeuroEncoder_V1(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", 
                 output_tokens=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        self.output_tokens = output_tokens
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        self.subj_proj= MLP(in_features=voxel_dim, hidden_features=linear_dim, out_features=latent_dim)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv1d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        # out = self.conv_out(h)
        
        return h


class NeuroDecoder_V1(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False, output_tokens=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks
        self.output_tokens = output_tokens

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv1d(z_channels,
        #                                block_in,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.subj_proj= MLP(in_features=latent_dim, hidden_features=linear_dim, out_features=voxel_dim)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        h = z #![b, 256, 1664]

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    
class NeuroEncoder_newsub(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", 
                 output_tokens=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ada_length = 1664*4 
        self.num_down_blocks = num_down_blocks
        self.output_tokens = output_tokens
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        # self.subj_proj= MLP(in_features=voxel_dim, hidden_features=linear_dim, out_features=latent_dim)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv1d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        # x = self.subj_proj(x)
        x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        # out = self.conv_out(h)
        
        return h


class NeuroDecoder_newsub(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False, output_tokens=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks
        self.output_tokens = output_tokens
        self.voxel_dim = voxel_dim

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv1d(z_channels,
        #                                block_in,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # self.subj_proj= MLP(in_features=latent_dim, hidden_features=linear_dim, out_features=voxel_dim)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        h = z #![b, 256, 1664]

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = self.subj_proj(h)
        h = torch.nn.functional.interpolate(h, size=self.voxel_dim, mode='linear', align_corners=True)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    
class NeuroEncoder_V12(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", 
                 output_tokens=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        self.output_tokens = output_tokens
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        # self.subj_proj= MLP(in_features=voxel_dim, hidden_features=linear_dim, out_features=latent_dim)
        self.subj_proj= nn.Linear(voxel_dim, latent_dim)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv1d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.conv_in(x)
        x = self.subj_proj(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        # out = self.conv_out(h)
        
        return h


class NeuroDecoder_V12(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, voxel_dim, linear_dim, latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False, output_tokens=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks
        self.output_tokens = output_tokens

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv1d(z_channels,
        #                                block_in,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # self.subj_proj= MLP(in_features=latent_dim, hidden_features=linear_dim, out_features=voxel_dim)
        self.subj_proj= nn.Linear(latent_dim, voxel_dim)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        h = z #![b, 256, 1664]

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        h = self.subj_proj(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        
        return h
    

class NeuroEncoder_V11(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_down_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", 
                 output_tokens=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # self.ada_length = ada_length
        self.num_down_blocks = num_down_blocks
        self.output_tokens = output_tokens
        
        self.conv_in = nn.Conv1d(in_channels,self.ch,kernel_size=1,stride=1,padding=0)
        # self.ada_maxpool = torch.nn.AdaptiveMaxPool1d(self.ada_length)
        self.subj_proj= MLP(in_features=15724, hidden_features=linear_dim, out_features=latent_dim)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_down_blocks:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv1d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 padding=0)

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        x = self.subj_proj(x)
        x = self.conv_in(x)
        # x = self.ada_maxpool(x)
        hs = [x]   #[b, ch, 8192]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  #! add block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) #! add attention
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))  #! add downsample layer

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)  #temb is none
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        # out = self.conv_out(h)
        
        return h


class NeuroDecoder_V11(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, num_up_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, linear_dim, latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False, output_tokens=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks
        self.output_tokens = output_tokens

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv1d(z_channels,
        #                                block_in,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions-self.num_up_blocks:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.subj_proj= MLP(in_features=latent_dim, hidden_features=linear_dim, out_features=15724)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        h = z #![b, 256, 1664]

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions-self.num_up_blocks:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:  #是否返回特征图
            return h
        
        # h = torch.nn.functional.interpolate(h, size=target_length, mode='linear', align_corners=True)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  ##! add conv layer here
        h = self.subj_proj(h)
        
        return h