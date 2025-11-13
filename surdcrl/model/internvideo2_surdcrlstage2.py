import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from typing import Dict, List, Optional, Union
import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange
from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .utils import ObjectsCrops
# from flash_attention_class import FlashAttention
# from flash_attn.modules.mlp import FusedMLP
# from flash_attn.ops.rms_norm import DropoutAddRMSNorm
from collections import OrderedDict
from mmengine.model import BaseModule, ModuleList
from mmengine.logging import MMLogger
from .internvideo2context import internvideo2_base_patch14_224_context
from itertools import combinations
import itertools
import numpy as np
from torch.nn import Parameter,Dropout
from .ops import extend_neg_masks
logger = MMLogger.get_current_instance()
token_objects = [
    [],              # idx 0: cls_token
    [0,1], [0,2], [0,3],         # idx 1~3: 二阶，包含对象0
    [1,2], [1,3], [2,3],         # idx 4~6: 二阶，其余组合
    [0,1,2], [0,1,3], [0,2,3],   # idx 7~9: 三阶，包含对象0
    [1,2,3],                    # idx 10: 三阶，其余组合
    [0,1,2,3]                   # idx 11: 四阶（所有对象）
]

from .Bert_backbone import BertAttention

import json
from types import SimpleNamespace

with open("/mnt/mymodel/work2/exp3-2/model/reverie_GOAT_model_config.json", "r") as f:
    config_dict = json.load(f)

config = SimpleNamespace(**config_dict)

class FrontDoorEncoder(nn.Module):
    def __init__(self, config, mode='batch_shared_group', num_prototypes=8):
        """
        :param mode: 'batch_shared_group' | 'point2group' | 'group2group'
        :param num_prototypes: 每个样本采样的 prototype 数（用于 group2group 或 batch_shared_group）
        """
        super().__init__()
        self.ll_self_attn = BertAttention(config)
        self.lg_cross_attn = BertAttention(config)

        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.config = config

        self.aug_linear = nn.Linear(config.hidden_size, 1)
        self.ori_linear = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.mode = mode
        self.num_prototypes = num_prototypes

    def forward(self, local_feats, global_feat_dict, local_feats_masks=None):
        """
        :param local_feats: [bs, L, d]
        :param global_feat_dict: [N, d] — prototype pool
        """
        bs, L, d = local_feats.shape
        N = global_feat_dict.shape[0]
        # ----- 前门采样 -----
        if self.mode == 'batch_shared_group':
            # 全 batch 共享一组 prototypes
            sample_indices = np.random.choice(N, self.num_prototypes, replace=False)
            logger.debug(f"FrontDoor sample indices: {sample_indices}")
            sampled = global_feat_dict[sample_indices]      # [num_prototypes, d]
            global_feats = sampled.unsqueeze(0)             # [1, P, d]，自动 broadcast 到 [bs, P, d]
            
        elif self.mode == 'point2group':
            # 每个样本一个 prototype
            sample_indices = np.random.choice(N, bs, replace=True)
            global_feats = global_feat_dict[sample_indices].unsqueeze(1)  # [bs, 1, d]

        elif self.mode == 'group2group':
            # 每个样本多个 prototypes
            sample_indices = np.random.choice(N, bs * self.num_prototypes, replace=True)
            sample_indices = sample_indices.reshape(bs, self.num_prototypes)  # [bs, P]
            global_feats = torch.stack([
                global_feat_dict[idx] for idx in sample_indices
            ], dim=0)  # [bs, P, d]

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'.")

        # ----- 前门注意力 -----
        if local_feats_masks is not None and len(local_feats_masks.shape) != 4:
            local_feats_masks = extend_neg_masks(local_feats_masks)
        ll_feats = self.ll_self_attn(local_feats, attention_mask=local_feats_masks)[0]
        lg_feats = self.lg_cross_attn(hidden_states=local_feats, encoder_hidden_states=global_feats)[0]
        out_feats = self.ln(ll_feats + lg_feats)

        # ----- 门控融合 -----
        aug_weight = self.sigmoid(self.aug_linear(out_feats) + self.ori_linear(local_feats))
        # print(f'out_feats: {aug_weight.mean():.4f}', f'local_feats: {(1 - aug_weight).mean():.4f}')
        out_feats = aug_weight * out_feats + (1 - aug_weight) * local_feats

        return out_feats



#=============================================================CCIM因果模块=================================================================
class CCIM(nn.Module):
    def __init__(self, num_joint_feature, num_gz, strategy, feature_dim: int = 768): #num_joint_feature d=1536 num_gz;768
        super(CCIM, self).__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz
        self.feature_dim = feature_dim
        if strategy == 'dp_cause':
              self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature, feature_dim=feature_dim)  #768,1536
        elif strategy == 'ad_cause':
              self.causal_intervention = additive_intervention(num_gz, num_joint_feature, feature_dim=feature_dim)
        else:
              raise ValueError("Do Not Exist This Strategy.")
                
        # self.w_h = Parameter(torch.Tensor(self.num_joint_feature, 768)) 
        # self.w_g = Parameter(torch.Tensor(self.num_gz, 768)) 
        self.aug_linear = nn.Linear(self.feature_dim, self.feature_dim)
        # para
        self.norm = nn.LayerNorm(self.feature_dim)
        self.balance = nn.Parameter(torch.zeros((self.feature_dim)))
        self.sigmoid = nn.Sigmoid()
        # gate
        # self.aug_linear = nn.Linear(768, 1)
        # self.ori_linear = nn.Linear(768, 1)
        # self.sigmoid = nn.Sigmoid()
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_normal_(self.w_h)
    #     nn.init.xavier_normal_(self.w_g)

    def forward(self, joint_feature, confounder_dictionary, prior):  #20,768   #20*1
#         ipdb.set_trace()
        joint_feature = joint_feature.squeeze(1)
        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)  #bs,768
        do_x = joint_feature + g_z
        # joint_feature = self.aug_linear(joint_feature)
        # proj_h = torch.matmul(joint_feature, self.w_h)  #1536，768=》768
        # proj_g_z = torch.matmul(g_z.to(torch.float32), self.w_g) 
        # do_x = proj_h + proj_g_z

        # para
        weight = self.sigmoid(self.balance)
        # print(f'g_z: {weight.mean():.4f}', f'joint_feature: {(1 - weight).mean():.4f}')
        do_x = self.norm((1 - weight) * joint_feature + weight * g_z)
        # gate
        # aug_weight = self.sigmoid(self.aug_linear(g_z) + self.ori_linear(joint_feature))
        # print(f'g_z: {aug_weight.mean():.4f}', f'joint_feature: {(1 - aug_weight).mean():.4f}')
        # do_x = aug_weight * g_z + (1 - aug_weight) * joint_feature


        do_x = do_x.unsqueeze(1)
        return do_x

        
# class dot_product_intervention(nn.Module):
#     def __init__(self, con_size, fuse_size):
#         super(dot_product_intervention, self).__init__()
#         self.con_size = con_size 
#         self.fuse_size = fuse_size  
#         self.query = nn.Linear(self.fuse_size, 768, bias= False) 
#         self.key = nn.Linear(self.con_size, 768, bias= False) 

#     def forward(self, confounder_set, fuse_rep, probabilities):
#         query = self.query(fuse_rep) 
#         key = self.key(confounder_set)  
#         mid = torch.matmul(query, key.transpose(0,1)) / math.sqrt(self.con_size)  
#         attention = F.softmax(mid, dim=-1)  # 1,26,20
#         # attention = attention.unsqueeze(2)  
        
#         batch_size = fuse_rep.size(0)  # 假设为 32
#         weighted_confounders = confounder_set * probabilities  # [20, 768]
#         weighted_confounders = weighted_confounders.unsqueeze(0).repeat(batch_size, 1, 1)  # [32, 20, 768]
#         fin = torch.bmm(attention, weighted_confounders)  # [32, 26, 768]


#         # fin = (attention*confounder_set*probabilities).sum(1)  

#         return fin
class dot_product_intervention(nn.Module):
    def __init__(self, con_size, fuse_size, feature_dim: int = 768):
        super(dot_product_intervention, self).__init__()
        self.con_size = con_size 
        self.fuse_size = fuse_size  
        self.feature_dim = feature_dim
        self.query = nn.Linear(self.fuse_size, self.feature_dim, bias= False) 
        self.key = nn.Linear(self.con_size, self.feature_dim, bias= False) 

    def forward(self, confounder_set, fuse_rep, probabilities):
#         ipdb.set_trace()
        query = self.query(fuse_rep) 
        key = self.key(confounder_set)  
        mid = torch.matmul(query, key.transpose(0,1)) / math.sqrt(self.con_size)  
        attention = F.softmax(mid, dim=-1) 
        attention = attention.unsqueeze(2)  
        fin = (attention*confounder_set*probabilities).sum(1)  

        return fin


class additive_intervention(nn.Module):
    def __init__(self, con_size, fuse_size, feature_dim: int = 768):
        super(additive_intervention,self).__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.Tan = nn.Tanh()
        self.feature_dim = feature_dim
        self.query = nn.Linear(self.fuse_size, self.feature_dim, bias = False)
        self.key = nn.Linear(self.con_size, self.feature_dim, bias = False)
        self.w_t = nn.Linear(self.feature_dim, 1, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):
#         ipdb.set_trace()

        query = self.query(fuse_rep) 

        key = self.key(confounder_set)  

        query_expand = query.unsqueeze(1)  
        fuse = query_expand + key 
        fuse = self.Tan(fuse)
        attention = self.w_t(fuse) 
        attention = F.softmax(attention, dim=1)
        fin = (attention*confounder_set*probabilities).sum(1)  

        return fin
        
class QuickGELU(BaseModule):
    """Quick GELU function. Forked from https://github.com/openai/CLIP/blob/d50
    d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py.

    Args:
        x (torch.Tensor): The input features of shape :math:`(B, N, C)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        if drop_path > 0.:
            logger.debug(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)
        
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # ipdb.set_trace()
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 1 2049 768
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)
        
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs
    
    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def forward(self, x, residual=None):
        
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x
        
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
            num_frames=8, tubelet_size=1, norm_layer=None, num_objects: int = 3
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_objects = num_objects
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ) # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.crop_layer = ObjectsCrops()
        self.box_categories = nn.Parameter(torch.zeros(self.grid_size[0], num_objects, embed_dim))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4,  embed_dim// 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim, bias=False),
            nn.ReLU()
        )
    def forward(self, x,metadata):
        box_tensors = metadata
        box_tensors = box_tensors.permute(0,2, 1, 3)

        box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
        BS = box_tensors_normalized.shape[0]
        box_emb = self.c_coord_to_feature(box_tensors_normalized)
        shape = (8,3, 768)
        box_categories = self.box_categories

        box_emb = box_categories.unsqueeze(0).expand(BS, -1, -1, -1) + box_emb # [BS, T, O, d]


        x = self.proj(x) # 1,768,8,16,16
        x = self.crop_layer(x, box_tensors)  # [BS, O, T, d, H, W]
        N,O,T,C,H,W = x.shape
        # 维度对齐：将box_emb匹配到特征图空间维度
        box_emb_aligned = box_emb.permute(0, 2, 1, 3)  # [BS, O, T, d]
        box_emb_aligned = box_emb_aligned.reshape(N*O, T, C, 1, 1)  # [N*O, T, C, 1, 1]
        box_emb_aligned = box_emb_aligned.permute(0, 2, 1, 3, 4)  # [N*O, C, T, 1, 1]

        x = x.reshape(N*O,T,C,H,W).permute(0,2,1,3,4)
        x = x + box_emb_aligned  # [N*O, C, T, H, W]


        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x
class VideoFrameChangeModule(nn.Module):
    def __init__(self, mlp_hidden_dim, output_dim, distance_type='diff', feature_dim: int = 768):
        super(VideoFrameChangeModule, self).__init__()
        self.distance_type = distance_type
        self.feature_dim = feature_dim
        
        # 初始化不同距离计算需要的组件
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.mahalanobis_estimator = None  # 用于存储马氏距离的协方差逆矩阵
        # 可学习的位置编码方式
        # scale = 768**-0.5
        # self.time_pos_embed = nn.Parameter(torch.zeros(1, 4, 8, 1, 768))
        # 增强的MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )
        # self.mlp = ModuleList([nn.Sequential(
        #     nn.Linear(768, mlp_hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_hidden_dim, output_dim)
        # )for i in range(7)])
        # self.ln = nn.LayerNorm(768)
        # 唯一 buffer 名称 + 安全注册
        self.register_buffer('_mahalanobis_inv_cov', None)  # 关键修复
    def _compute_distance(self, prev, curr):
        """根据配置计算不同距离"""
        if self.distance_type == 'diff':
            # 直接元素差值 [..., 768]
            return prev - curr
            
        elif self.distance_type == 'manhattan':
            # 曼哈顿距离 [..., 1]
            distance = torch.sum(torch.abs(prev - curr), dim=-1, keepdim=True)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'euclidean':
            # 欧式距离 [..., 1]
            distance = torch.norm(prev - curr, p=2, dim=-1, keepdim=True)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'cosine':
            # 余弦距离 [..., 1]
            prevclone = prev.clone()
            currclone = curr.clone()
            cos_sim = torch.cosine_similarity(prevclone, currclone, dim=-1, eps=1e-6)
            distance = 1 - cos_sim.unsqueeze(-1)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'mahalanobis':
            if self._mahalanobis_inv_cov is None:
                self._init_mahalanobis(prev)
            
            diff = (prev - curr).reshape(-1, 768)
            VI = self._mahalanobis_inv_cov
            mahal = torch.sqrt(torch.einsum('bi,ij,bj->b', diff, VI, diff))
            distance = mahal.view(*prev.shape[:-1], 1)
            return self._expand_distance(distance)
            
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def _expand_distance(self, distance):
        """将标量距离扩展为768维"""
        if distance.shape[-1] == 1 and self.distance_type != 'diff':
            # 复制扩展策略（可替换为可学习的MLP扩展）
            return distance.expand(*distance.shape[:-1], self.feature_dim)
        return distance
    
    def _init_mahalanobis(self, features):
        with torch.no_grad():
            flat_features = features.reshape(-1, self.feature_dim).float().to(features.device)
            cov = torch.cov(flat_features.T, correction=0)
            cov_reg = cov + 1e-6 * torch.eye(self.feature_dim, device=features.device)
            self._mahalanobis_inv_cov = torch.linalg.pinv(cov_reg)  # 直接赋值

    def forward(self, xo):
        """处理输入特征并计算帧间变化
        Args:
            x: (batch_size, num_targets, num_frames, L, feature_dim)
        """
        x = xo.clone()
        # 应用时间位置编码
        # x = x + self.time_pos_embed
        # x = self.ln(x)
        # 修正形状解包（原代码可能存在维度不匹配）
        batch_size, num_targets, num_frames, L, feature_dim = x.shape
        
        # 存储所有变化特征
        change_features = []

        # 遍历时间步计算变化
        for t in range(1, num_frames):
            prev_frame = x[:, :, t-1]  # [B, N, L, D]
            curr_frame = x[:, :, t]    # [B, N, L, D]
            
            # 计算帧间变化（根据配置类型）
            change = self._compute_distance(prev_frame, curr_frame)
            
            # 通过MLP处理变化特征
            change_feature = self.mlp(change)
            
            # 残差连接更新当前帧
            x[:, :, t] = curr_frame + change_feature
            
            # 保存变化特征
            change_features.append(change_feature.unsqueeze(2))
        
        # 合并时间维度变化特征
        final_change_features = torch.cat(change_features, dim=2)
        return x, final_change_features
class Extractor(BaseModule):
    """Global UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        drop_out (float): Stochastic dropout rate.
            Defaults to 0.0.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_factor: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))

        
        aff = aff.softmax(dim=-1)
        out = aff @ v
        # ipdb.set_trace()
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
class InternVideo2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            head_drop_path_rate: float = 0.,
            num_heads: int = 16,
            mlp_ratio: float = 4.3637,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = False,
            use_fused_rmsnorm: bool = False,
            use_fused_mlp: bool = False,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            fc_drop_rate: float = 0., 
            num_classes: int = 1000, 
            init_scale: float = 0.001,
            num_objects: int = 3,
            objagg_num_heads: int = 12,
        ):
        super().__init__()
        
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        logger.debug(f"MLP ratio: {mlp_ratio}")
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.num_objects = num_objects
        
        # if use_fused_rmsnorm: 
        #     norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        # else:
        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size, num_objects=num_objects
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            logger.debug("Use separable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            logger.debug("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.debug(f"Droppath rate: {dpr}")
        logger.debug(f"Checkpoint list: {with_cp_list}")
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=head_drop_path_rate, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim
        )
        
        self.fc_norm = nn.LayerNorm(clip_embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        
        # self.head = nn.Linear(clip_embed_dim, num_classes)
        self.vfcm = VideoFrameChangeModule(embed_dim * 4, embed_dim,"manhattan", feature_dim=embed_dim)
        self.objagg =  Extractor(
                embed_dim,
                objagg_num_heads,
                mlp_factor=4,
                dropout=0.0,
                drop_path=0.0,
            )
        self.obj_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ln = nn.LayerNorm(embed_dim)
        
        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def init_pos_embed(self):
        logger.debug("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            # trunc_normal_(self.pos_embed_spatial, std=.02)
            # trunc_normal_(self.pos_embed_temporal, std=.02)
            # trunc_normal_(self.pos_embed_cls, std=.02)
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[0], # t_size
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 
            'pos_embed_spatial', 
            'pos_embed_temporal', 
            'pos_embed_cls',
            'cls_token'
        }
    
    def forward(self, x,box):
        x = self.patch_embed(x.type(self.dtype),box)
        B, T, L, C = x.shape  # T: temporal; L: spatial
        O = self.num_objects
        BS = B // O
        # x = x.view([B, T * L, C])
        x = x.reshape([B, T * L, C])
        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add pos_embed
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.grid_size[1] * self.grid_size[2],
                dim=1,
            )
            pos_embed = torch.cat(
                [
                    self.pos_embed_cls.expand(pos_embed.shape[0], -1, -1),
                    pos_embed,
                ],
                1,
            )
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed # [1,2049,768]

        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
         # 1,2049,768
        resx = x
        resx = self.clip_projector(resx).reshape(BS,O,C) # 1,768

        objx = x[:,1:,:].reshape(BS,O,T,L,C)

        ym,yt = self.vfcm(objx) # 1,4,7,197,768
        yt = yt.reshape(L*(T-1), BS*O, C)  # L, N, T, C
        cls_token = self.obj_cls_token.repeat(1, BS*O, 1)
        yt = self.objagg(cls_token,yt)[0, :, :].reshape(BS,O,768)
        
        obj = resx + self.ln(yt) # 2,
        # obj = resx + yt # 2,
        feature = self.fc_norm(obj)

        # x = self.head(self.fc_dropout(feature))
        return feature,resx

def get_updateModel(model, path):
    # pretrained_dict = torch.load(path, map_location='cpu')['module'] # 自己训练的模型
    pretrained_dict = torch.load(path, map_location='cpu') # 自己训练的模型
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    # for key, value in pretrained_dict.items():
    #         print("{}: {}".format(key, value.shape))

    # for key, value in model_dict.items():
    #     print("{}: {}".format(key, value.shape))
    # ipdb.set_trace()
    for k, v in pretrained_dict.items():
        new_state_dict[k] = v

    shared_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(shared_dict)

    # 下面两行，test和训练注释掉
    # del model_dict['head.weight']  # 改变分类数量
    # del model_dict['head.bias']
    logger.info("Re-initialize fc layer")
    logger.info("Checkpoint matched keys: {}".format(len(shared_dict.keys())))
    model.load_state_dict(model_dict, strict=False)
    return model





@register_model
def internvideo2_small_patch14_224(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=384, 
        depth=12, num_heads=6, mlp_ratio=4, 
        attn_pool_num_heads=16, clip_embed_dim=768,
        **kwargs
    )
    return model


@register_model
def internvideo2_base_patch14_224(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4, 
        attn_pool_num_heads=16, clip_embed_dim=768,
        **kwargs
    )
    if pretrained:
        model = get_updateModel(model,"/mnt/fuxiancode/internvideo2/pretrained/k400_base.bin")
    return model

    
@register_model
def internvideo2_large_patch14_224(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=1024, 
        depth=24, num_heads=16, mlp_ratio=4, 
        attn_pool_num_heads=16, clip_embed_dim=768,
        **kwargs
    )
    if pretrained:
        model = get_updateModel(model,"/mnt/fuxiancode/internvideo2/pretrained/k400_large.bin")
    return model


@register_model
def internvideo2_1B_patch14_224(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=1408, 
        depth=40, num_heads=16, mlp_ratio=48/11, 
        attn_pool_num_heads=16, clip_embed_dim=768,
        **kwargs
    )
    if pretrained:
        model = get_updateModel(model,"/mnt/fuxiancode/internvideo2/pretrained/1B_ft_k710_ft_k700_f8.pth")
    return model


@register_model
def internvideo2_6B_patch14_224(pretrained=False, **kwargs):
    model = InternVideo2(
        img_size=224, patch_size=14, embed_dim=3200, 
        depth=48, num_heads=25, mlp_ratio=4, 
        attn_pool_num_heads=16, clip_embed_dim=768,
        **kwargs
    )
    return model


class multiAgg(BaseModule):
    """Global UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        drop_out (float): Stochastic dropout rate.
            Defaults to 0.0.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_factor: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')
        # self.fc = nn.Sequential(
        #     nn.Linear(768*2, 768),
        #     nn.ELU()
        # )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))
        
        aff = aff.softmax(dim=-1)
        out = aff @ v
        # ipdb.set_trace()
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
class CrossAttention1(BaseModule):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # spatial
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)

    # def attention(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.attn(x, x, x, need_weights=True, attn_mask=None)
    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x: [seq_len, bs, d_model]
        attn_mask: [seq_len, seq_len] or [bs, seq_len, seq_len], with dtype=bool
                   True 表示该位置被 mask（不可见）
        """
        B = x.shape[1]
        seq_len = x.shape[0]
        d_model = x.shape[2]
        head_dim = d_model // self.n_head
    
        # Linear projections: [seq_len, bs, d_model] -> [bs, n_head, seq_len, head_dim]
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(seq_len, B, self.n_head, head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(seq_len, B, self.n_head, head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(seq_len, B, self.n_head, head_dim).transpose(0, 1).transpose(1, 2)
    
        ##-----temp------##
        # r = torch.matmul(q, k.transpose(-2, -1)) * 1e4 # [bs, n_head, seq, seq]
        # # Apply attn_mask
        # if attn_mask is not None:
        #     # Ensure mask shape is [seq, seq] → broadcast to [bs, n_head, seq, seq]
        #     if attn_mask.dim() == 2:
        #         attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        #     r = r.masked_fill(attn_mask.to(r.device), float('-inf'))
        # r_attn_weights = F.softmax(r, dim=-1)  # [bs, n_head, seq, seq]
        ##------temp-----##

        # Scaled dot-product attention
        # attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # [bs, n_head, seq, seq]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * 1e4 # [bs, n_head, seq, seq]
        # Apply attn_mask
        if attn_mask is not None:
            # Ensure mask shape is [seq, seq] → broadcast to [bs, n_head, seq, seq]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            attn_logits = attn_logits.masked_fill(attn_mask.to(attn_logits.device), float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)  # [bs, n_head, seq, seq]
    
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [bs, n_head, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, d_model).transpose(0, 1)
    
        # Output projection
        output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)  # [seq_len, bs, d_model]
    
        return output, attn_weights.mean(1)  # mean over heads → [bs, seq, seq]

    def forward(self, x: torch.Tensor,attn_mask: torch.Tensor = None) -> torch.Tensor:
        out, w = self.drop_path(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + out  # Residual
        x = x + self.drop_path(self.mlp(self.ln_2(x)))  # FFN
        return x, w




class synandred(nn.Module):
    def __init__(self,
                 use_pos_embedding=False, 
                 use_mean_pooling=True, 
                 shared_relation_fc=True, 
                 use_structured_attn_mask=True,
                 used_orders=[2, 3, 4, 5],
                 num_samples = 10,attnLayer = 4,num_prototypes=4,
                 feature_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        self.use_mean_pooling = use_mean_pooling
        self.shared_relation_fc = shared_relation_fc
        self.use_structured_attn_mask = use_structured_attn_mask
        self.used_orders = used_orders
        self.num_samples = num_samples        
        self.feature_dim = feature_dim
        if self.shared_relation_fc:
            self.shared_fc2 = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.ELU(),
                # nn.Linear(768, 768)
            )
            self.shared_fc3 = nn.Sequential(
                nn.Linear(self.feature_dim * 3, self.feature_dim),
                nn.ELU(),
                # nn.Linear(768, 768)
            )
            self.shared_fc4 = nn.Sequential(
                nn.Linear(self.feature_dim * 4, self.feature_dim),
                nn.ELU(),
                # nn.Linear(768, 768)
            )
            self.shared_fc5 = nn.Sequential(
                nn.Linear(self.feature_dim * 5, self.feature_dim),
                nn.ELU(),
                # nn.Linear(768, 768)
            )
        else:
            self.fc2_list = nn.ModuleList([
                nn.Sequential(nn.Linear(self.feature_dim * 2, self.feature_dim), nn.ELU())
                for _ in range(10)
            ])
            self.fc3_list = nn.ModuleList([
                nn.Sequential(nn.Linear(self.feature_dim * 3, self.feature_dim), nn.ELU())
                for _ in range(10)
            ])
            self.fc4_list = nn.ModuleList([       
                nn.Sequential(nn.Linear(self.feature_dim * 4, self.feature_dim), nn.ELU())
                for _ in range(5)
            ])

        self.crossAttention = nn.ModuleList([CrossAttention1(self.feature_dim, num_heads) for _ in range(attnLayer)])
        if use_mean_pooling == False:
            self.symodule = nn.ModuleList([multiAgg(self.feature_dim, num_heads, mlp_factor=4, dropout=0.0, drop_path=0.0) for _ in range(1)])
            self.remodule = nn.ModuleList([multiAgg(self.feature_dim, num_heads, mlp_factor=4, dropout=0.0, drop_path=0.0) for _ in range(1)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.syclstoken = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.reclstoken = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(11+1, 1, self.feature_dim))  # [cls] + 11
        
        self.bd = CCIM(self.feature_dim, self.feature_dim, strategy = 'dp_cause', feature_dim=self.feature_dim)
        self.fd = FrontDoorEncoder(config,'batch_shared_group',num_prototypes)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
            for p in m.parameters():
                nn.init.normal_(p, std=0.02)

    def get_combinations(self, x, n):
        bs, N, d = x.shape
        combos = list(itertools.combinations(range(N), n))
        feats = torch.stack([
            x[:, list(c), :].reshape(bs, -1) for c in combos
        ], dim=1)
        return feats, combos



    
    def build_structured_attn_mask(self, token_objects, total_tokens=27):
        """
        构建结构感知的 attention mask：
        - 高阶组合可以看到它的所有低阶子组合；
        - 低阶组合也可以看到它所包含的所有高阶组合；
        - 同阶可以看到存在交集的同阶组合
        - cls token 可以看所有；
        - token 自己可以看自己；
        
        Args:
            token_objects: list of list[int], 编码对象组成（[0] 是 cls_token）
            total_tokens: int, 默认 27（1 cls + 26 组合）
    
        Returns:
            attn_mask: BoolTensor, shape [total_tokens, total_tokens]
        """
        combo2idx = {
            tuple(sorted(combo)): idx
            for idx, combo in enumerate(token_objects[1:], start=1)
        }
    
        # 初始化：全为 True（全部 masked）
        attn_mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)
    
        # cls_token <-> 所有组合
        attn_mask[0, 1:] = False  # cls 看所有组合
        attn_mask[1:, 0] = False  # 所有组合可以看 cls
    
        # 正向连接：高阶组合看所有子组合（r=2 ~ L-1）
        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            L = len(current_combo)
            for r in range(2, L):  # 只看二阶以上子集
                for subset in combinations(current_combo, r):
                    subset = tuple(sorted(subset))
                    if subset in combo2idx:
                        j = combo2idx[subset]
                        attn_mask[i, j] = False
            attn_mask[i, i] = False  # 自己看自己
    
        # 反向连接：低阶组合看所有包含它的高阶组合
        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            current_key = tuple(sorted(current_combo))
            for key, j in combo2idx.items():
                if set(current_key).issubset(set(key)) and key != current_key:
                    attn_mask[i, j] = False  # i 可以看 j（高阶包含当前组合）
                    
        # 同阶 token 可以互相可见
        # for j in range(1, total_tokens):
        #     if j == i:
        #         continue
        #     other_combo = token_objects[j]
        #     if len(current_combo) == len(other_combo):  # 同阶
        #         attn_mask[i, j] = False

        # 同阶 token 之间只有在存在交集时才可见
        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            for j in range(1, total_tokens):
                if i == j:
                    continue
                other_combo = token_objects[j]
                if len(current_combo) == len(other_combo):
                    if set(current_combo) & set(other_combo):  # 有交集才可见
                        attn_mask[i, j] = False

        return attn_mask


    def soft_inverse_transform_sampling(self, cdf, num_samples,p,STE=False):
        bs, T = cdf.shape
        device = cdf.device
        positions = (torch.arange(num_samples, device=device).float() + 0.5) / num_samples
        positions = positions.unsqueeze(0).expand(bs, -1)
        padded_cdf = F.pad(cdf, (1, 0), mode='constant', value=0.0)
        bin_idx = torch.searchsorted(cdf, positions, right=False)
        bin_idx = torch.clamp(bin_idx, 0, T - 1)

        logger.debug(f"Sampled bin indices: {bin_idx}")
        if STE == True:
            # -- One-hot mask --
            hard_mask = torch.zeros_like(p).scatter(1, bin_idx, 1.0)  # [bs, 10], 多个 1
            # -- 使用 STE：前向使用硬mask，反向用 soft --
            soft_mask = (hard_mask - p).detach() + p  # STE 技巧，p 有梯度
        else:
            # 采集idx
            left_cdf = torch.gather(padded_cdf, 1, bin_idx)
            right_cdf = torch.gather(padded_cdf, 1, bin_idx + 1)
            denom = (right_cdf - left_cdf + 1e-6)
            alpha = (positions - left_cdf) / denom
            soft_mask = torch.zeros(bs, T, device=device)
            for i in range(num_samples):
                b = torch.arange(bs, device=device)
                left_idx = bin_idx[:, i]
                right_idx = torch.clamp(bin_idx[:, i] + 1, max=T - 1)
                soft_mask[b, left_idx] += (1.0 - alpha[:, i])
                soft_mask[b, right_idx] += alpha[:, i]
            soft_mask = soft_mask / (soft_mask.sum(dim=1, keepdim=True) + 1e-6)
        return soft_mask,bin_idx

    def encode_relation_feats(self, feats, fc_list_or_shared_fc, order):
        if self.shared_relation_fc:
            fc = getattr(self, f'shared_fc{order}')
            return fc(feats)
        else:
            return torch.stack([
                fc_list_or_shared_fc[i](feats[:, i, :]) for i in range(feats.size(1))
            ], dim=1)

    def forward(self, obj, context,global_feat_dict,bg_confounder_dictionary,bg_prior):

        # # # # ----------后门干预，减少关系中背景占据主导的影响-----------
        context = self.bd(context, bg_confounder_dictionary, bg_prior)
        # # -----前门干预------
        obj = self.fd(obj,global_feat_dict)
        x = torch.cat([context, obj], dim=1)  # [bs, 5, 768]

        
        rel_feats = []

        if 2 in self.used_orders:
            feats2, _ = self.get_combinations(x, 2)
            rel2 = self.encode_relation_feats(feats2, self.fc2_list if not self.shared_relation_fc else None, order=2)
            rel_feats.append(rel2)
        
        if 3 in self.used_orders:
            feats3, _ = self.get_combinations(x, 3)
            rel3 = self.encode_relation_feats(feats3, self.fc3_list if not self.shared_relation_fc else None, order=3)
            rel_feats.append(rel3)
        
        if 4 in self.used_orders:
            feats4, _ = self.get_combinations(x, 4)
            rel4 = self.encode_relation_feats(feats4, self.fc4_list if not self.shared_relation_fc else None, order=4)
            rel_feats.append(rel4)
        
        if 5 in self.used_orders:
            feats5 = x.reshape(x.size(0), 1, -1)
            rel5 = self.shared_fc5(feats5) if self.shared_relation_fc else self.fc5_list[0](feats5)
            rel_feats.append(rel5)


        
        all_rel_feats = torch.cat(rel_feats, dim=1)  # [bs, N_rel, 768]
        bs = all_rel_feats.size(0)
        
        
        cls_token = self.cls_token.expand(bs, -1, -1)
        rel_input = torch.cat([cls_token, all_rel_feats], dim=1).permute(1, 0, 2)
        if self.use_pos_embedding:
            pos_embedding = self.pos_embedding.repeat(1, bs, 1)
            rel_input = rel_input + pos_embedding
        if self.use_structured_attn_mask:
            attn_mask = self.build_structured_attn_mask(token_objects,total_tokens=12)  # [27, 27]
        else:
            attn_mask = torch.zeros(len(token_objects), len(token_objects), dtype=torch.bool)

        
        # rel_input = self.fd(rel_input,global_feat_dict)

        # ipdb.set_trace()
        #计算组合得分
        for attn in self.crossAttention:
            rel_input, attn_weights = attn(rel_input,attn_mask=attn_mask) # [27,bs,768],[2,27,27]
            # 前门调整
            # special_token = rel_input[0:1, :, :] 
            # tokens_to_adjust = rel_input[1:, :, :]
            # rel_input  = self.fd(rel_input,global_feat_dict)
            # rel_input = torch.cat([special_token, adjusted], dim=0)  
            # ipdb.set_trace()
        
        w = attn_weights[:, 0, 1:]
        cdf = torch.cumsum(w, dim=1)
        soft_mask,bin_idx = self.soft_inverse_transform_sampling(cdf, num_samples=self.num_samples,p=w,STE=False)
        # print(soft_mask)
        relation_ = rel_input.permute(1, 0, 2)[:, 1:, :]
        # print(soft_mask)
        
        if self.use_mean_pooling:
            synergistic = torch.bmm(soft_mask.unsqueeze(1), relation_).squeeze(1)
            redundant = torch.bmm((1 - soft_mask).unsqueeze(1), relation_).squeeze(1)
        else:
            sy_token = self.syclstoken.repeat(1, bs, 1)
            re_token = self.reclstoken.repeat(1, bs, 1)
            for symodule in self.symodule:
                sy_token = symodule(sy_token, (soft_mask.unsqueeze(-1) * relation_).permute(1, 0, 2))
            synergistic = sy_token.squeeze(0)
            for remodule in self.remodule:
                re_token = remodule(re_token, ((1 - soft_mask).unsqueeze(-1) * relation_).permute(1, 0, 2))
            redundant = re_token.squeeze(0)
        return synergistic, redundant, bin_idx
class orcausal(nn.Module):
    def __init__(self, num_class=20,num_samples=10,attnlayer=8,num_prototypes=8,
                 feature_dim: int = 768, num_objects: int = 3, relation_num_heads: int = 12):
        super().__init__()
        self.moduleobj = internvideo2_base_patch14_224(pretrained=True,num_classes=num_class, num_objects=num_objects)
        self.modulecontext = internvideo2_base_patch14_224_context(pretrained=True,num_classes=num_class)
        self.uniquefc = nn.Sequential(
            nn.Linear(feature_dim * (num_objects + 1), feature_dim),
            nn.ELU()
        )
        self.cgfc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ELU()
        )
        self.synandred = synandred(use_pos_embedding=True,
                                   use_mean_pooling=True, 
                                   shared_relation_fc=True,
                                   use_structured_attn_mask=True,
                                   used_orders=[2,3,4],
                                   num_samples=num_samples,
                                   attnLayer=attnlayer,
                                   num_prototypes=num_prototypes,
                                   feature_dim=feature_dim,
                                   num_heads=relation_num_heads)
        self.classifer = nn.Linear(feature_dim,num_class)
        self.synhead = nn.Linear(feature_dim,num_class)

    
        # self.get_updateModel("/mnt/mymodel/work2/exp3-2/model/checkpoint/exp4/msrvtt-uniformerv2-2-1e-05-25-0.4-0.05-2-L1_noln-63.14.pth")
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     else:
    #        for p in m.parameters(): nn.init.normal_(p, std=0.02)
    def get_updateModel(self, path): 
        # pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_dict = torch.load(path, map_location ='cpu') # 自己训练的模型
        model_dict = self.state_dict() 
        new_state_dict = OrderedDict()
        # for key, value in pretrained_dict.items():
        #     print("{}: {}".format(key, value.shape))
        # # ipdb.set_trace()
        # for key, value in model_dict.items():
        #     print("{}: {}".format(key, value.shape))
        # ipdb.set_trace()
        for k, v in pretrained_dict.items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
            if 'module' in k:
                name = k[7:]
                new_state_dict[name] = v 
#         ipdb.set_trace()

        shared_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(shared_dict)

        # 下面两行，test和训练注释掉
        # del model_dict['classifer.weight']  # 改变分类数量
        # del model_dict['classifer.bias']
        print("re-initialize fc layer")
        print("ckpt key lens{}".format(len(shared_dict.keys())))
        self.load_state_dict(model_dict, strict=False)
        return self 
    def forward(self,x,metadata: torch.Tensor,back,global_feat_dict,bg_confounder_dictionary,bg_prior):
        obj,objres = self.moduleobj(x,metadata)
        context,contextres = self.modulecontext(back)
        context = context.unsqueeze(1)
        BS = obj.shape[0]
        unique = torch.cat([obj,context],dim=1).view(BS,-1)# 1,5,768
        unique = self.uniquefc(unique)
        out = self.classifer(unique)

        # objres = objres.permute(3,0,1,2,4)[0].mean(2)
        # contextres = contextres.view(197,BS,8,768)[0].mean(1).unsqueeze(1)
        synergistic,redundant,bin_idx = self.synandred(obj,context,global_feat_dict,bg_confounder_dictionary,bg_prior) # 这里可能还是要用contextres
        

        # weight = self.sigmoid(self.redw)
        # print(f'weight_red: {weight.mean():.4f}', f'weight_syn: {(1 - weight).mean():.4f}')
        # casualgraph = torch.cat([synergistic, weight * redundant], dim=1) # 

    
        casualgraph = torch.cat([synergistic,redundant], dim=1)
        casualgraph = self.cgfc(casualgraph)
        
        # # ----------后门干预，减少关系中背景占据主导的影响-----------
        # casualgraph = self.bd(casualgraph, bg_confounder_dictionary, bg_prior)
        
        out1 = self.synhead(casualgraph)

        return out + out1,synergistic,redundant,bin_idx
if __name__ == '__main__':
    import time
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np
    model = orcausal(num_class=20)
    input = torch.rand(1,3,8,224, 224)  # [batch size, channl, frames, H, W]
    box = torch.randint(low=0, high=224, size=(1, 4, 8, 4))
    back = torch.rand(1,3,8,224, 224)
    globaldict = torch.rand(512,768)
    bg_confounder_dictionary = torch.rand(20,768)
    bg_prior = torch.rand(20,1)
    
    # 使用 thop 计算 FLOPs 和参数
    gflops, params = profile(model, inputs=(input, box, back, globaldict, bg_confounder_dictionary, bg_prior))
    gflops = gflops / 1e9
    gflops, params = clever_format([gflops, params], "%.3f")
    print(f"Total FLOPs: {gflops} GFLOPs")
    print(f"Total Params: {params} params")

    # 使用 fvcore 计算每层的 FLOPs
    flops = FlopCountAnalysis(model, (input, box, back, globaldict, bg_confounder_dictionary, bg_prior))
    
    # 打印每个模块的 FLOPs 和参数
    print(flop_count_table(flops, max_depth=2))  # 控制显示层级，max_depth=2 显示更详细的层级结构
# if __name__ == '__main__':
#     import time
#     from fvcore.nn import FlopCountAnalysis
#     from fvcore.nn import flop_count_table
#     import numpy as np
#     from thop import profile, clever_format
#     seed = 4217
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     input = torch.rand(1,3,8,224, 224)  # [batch size, channl, frames, H, W]
#     box = torch.randint(low=0, high=224, size=(1, 3, 8, 4))
#     back = torch.rand(1,3,8,224, 224)
#     globaldict = torch.rand(512,768)
#     bg_confounder_dictionary = torch.rand(512,768)
#     bg_prior = torch.rand(512,1)
#     model = orcausal(num_class=20,num_samples=5,attnlayer=8,num_prototypes=8)

#     gflops, params = profile(model, inputs=(input,box,back,globaldict,bg_confounder_dictionary,bg_prior))
#     gflops = gflops / 1e9
#     gflops, params = clever_format([gflops, params], "%.3f")
#     print(gflops,params)