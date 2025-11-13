import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange
import ipdb
from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
# from flash_attention_class import FlashAttention
# from flash_attn.modules.mlp import FusedMLP
# from flash_attn.ops.rms_norm import DropoutAddRMSNorm
from collections import OrderedDict
from .utils import ObjectsCrops, box2spatial_layout, Mlp
from .attention import TrajectoryAttention, SeltAttentionBlock
# from mmaction.registry import MODELS
from .orvitdefaults import _C
class ORViT(nn.Module):

    def __init__(
            self, cfg, dim=768, dim_out=None, num_heads=12, attn_type='trajectory',
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code = False, nb_frames=8,
        ):
        super().__init__()

        self.cfg = cfg
        self.in_dim = dim
        self.dim = dim
        self.nb_frames = nb_frames

        self.with_cls_token = True 
        self.with_motion_stream = False

        # Object Tokens
        self.crop_layer = ObjectsCrops()
        self.patch_to_d = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // 2, self.dim, bias=False),
            nn.ReLU()
        )

        self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, self.cfg.ORVIT.O, self.in_dim))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

        # Attention Block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = TrajectoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop,
            proj_drop=drop,
        )


        if self.with_motion_stream:
            self.motion_stream = MotionStream(cfg, dim=dim, num_heads=num_heads, attn_type=cfg.ORVIT.MOTION_STREAM_ATTN_TYPE, 
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                                                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                                                nb_frames=self.nb_frames,
                                                )
            self.motion_mlp = Mlp(in_features=cfg.ORVIT.MOTION_STREAM_DIM if cfg.ORVIT.MOTION_STREAM_DIM > 0 else dim,
                                        hidden_features=mlp_hidden_dim, out_features=dim,
                                        act_layer=act_layer, drop=drop)

        if self.cfg.ORVIT.INIT_WEIGHTS:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
           for p in m.parameters(): nn.init.normal_(p, std=0.02)

    def forward(self,x, metadata,thw):
        # x = [1 2049 768]   metadata =  [1,4,8,4]
        box_tensors = metadata # 1,4,8,4
        box_tensors = box_tensors.permute(0,2, 1, 3)
        box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
        BS,O = box_tensors_normalized.shape[0],box_tensors_normalized.shape[2]
        assert box_tensors is not None

        if self.with_cls_token:
            # 2 197 8*768
            cls_token, patch_tokens = x[:,[0]], x[:,1:]
    
        BS, _, d = x.shape # 1,2049,768
        T,H,W = thw
        patch_tokens = patch_tokens.permute(0,2,1).reshape(BS, -1, T,H,W) # 可用conv1后的代替 2 768 8 14 14
        
        BS, d, T, H, W = patch_tokens.shape
        assert T == self.nb_frames

        # Tratio = box_tensors.shape[1]
        # box_tensors = box_tensors[:,::Tratio] # [BS, T , O, 4]
        O = box_tensors.shape[-2]
        object_tokens = self.crop_layer(patch_tokens, box_tensors)  # [BS, O,T, d, H, W]
        object_tokens = object_tokens.permute(0, 1,2,4,5,3)  # [BS, O,T, H, W, d]
        object_tokens = self.patch_to_d(object_tokens) # [BS,O,T, H, W, d]
        object_tokens =torch.amax(object_tokens, dim=(-3,-2)) # [BS, O,T, d]
        object_tokens = object_tokens.permute(0,2,1,3) # [BS, T, O, d]
        box_categories = self.box_categories.unsqueeze(0).expand(BS,-1,-1,-1)
        box_emb = self.c_coord_to_feature(box_tensors)
        # ipdb.set_trace()

        all_tokens = torch.cat([patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d), object_tokens], dim = 2).flatten(1,2) # [BS, T * (H*W+O),d]
        if self.with_cls_token:
            all_tokens =  torch.cat([cls_token, all_tokens], dim = 1) # [BS, 1 + T*N, d]

        all_tokens, thw = self.attn(
                    self.norm1(all_tokens), 
                    [T, H*W + O, 1], 
                )

        if self.with_cls_token:
            cls_token, all_tokens =  all_tokens[:, [0]], all_tokens[:, 1:]

        patch_tokens = all_tokens.reshape(BS,T,H*W+O,d)[:,:,:H*W].reshape(BS,T*H*W,d)

        if self.with_motion_stream:
            box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
            motion_emb = self.motion_stream(box_tensors_normalized,H, W) # [BS, T, H, W, d]
            motion_emb = self.motion_mlp(motion_emb) # [BS, T*H*W, d]
            patch_tokens = patch_tokens + motion_emb


        if self.with_cls_token:
            patch_tokens = torch.cat([cls_token, patch_tokens], dim = 1) # [BS, 1 + N, d]

        x = x + self.drop_path(patch_tokens) # [BS, N, d]
        # x = x + self.drop_path(self.mlp(self.norm2(x))) # [BS, N, d]

        return x, thw

class Object2Spatial(nn.Module):
    def __init__(self, cfg, _type):
        super().__init__()
        self.cfg = cfg
        self._type = _type 
    def forward(self, all_features, context, boxes, H, W, t_avg_pooling = False):
        BS, T, O, d = all_features.shape

        if self._type == 'layout':
            ret = box2spatial_layout(boxes, all_features,H,W) # [B, d, T, H, W]
            ret = ret.permute(0,2,3,4,1)
            if t_avg_pooling:
                BS, T, H, W, d = ret.size()
                Tratio = int(T / self.cfg.MF.TEMPORAL_RESOLUTION)
                if Tratio > 1:
                    ret = ret.reshape(BS, -1, Tratio, H, W, d).mean(2)
            ret = ret.reshape(BS*T,H*W,d)
            # ret = ret.flatten(1,3) # [BS, T*H*W, d]
        elif self._type == 'spatial_only':
            assert context is not None
            ret = context.flatten(1,-2) # [BS, T*H*W, d]
        elif self._type == 'object_pooling':
            ret = torch.amax(all_features, dim = 2) # [BS, T, d]
            ret = ret.reshape(BS, T, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        elif self._type == 'all_pooling':
            ret = torch.amax(all_features, dim = [1,2]) # [BS, T, d]
            ret = ret.reshape(BS, 1, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        else:
            raise NotImplementedError(f'{self._type}')
        return ret

class MotionStream(nn.Module):
    def __init__(self, cfg, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            nb_frames = None,
        ):
        super().__init__()


        self.cfg = cfg

        self.in_dim = cfg.ORVIT.MOTION_STREAM_DIM if cfg.ORVIT.MOTION_STREAM_DIM > 0 else dim
        self.dim = dim
        self.nb_frames = nb_frames

        if self.cfg.ORVIT.MOTION_STREAM_SEP_POS_EMB:
            self.box_categories_T = nn.Parameter(torch.zeros(self.nb_frames, 1, self.in_dim))
            self.box_categories_O = nn.Parameter(torch.zeros(1, self.cfg.ORVIT.O, self.in_dim))
        else:
            self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, self.cfg.ORVIT.O, self.in_dim))


        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

        # Attention Block
        self.attn_type = attn_type

        if attn_type == 'joint':
            self.attn = SeltAttentionBlock(
                dim=self.in_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

        self.obj2spatial = Object2Spatial(cfg, _type = 'layout')

    def forward(self, box_tensors, H, W):
        # box_tenors: [BS, T, O, 4]
        BS = box_tensors.shape[0]

        box_emb = self.c_coord_to_feature(box_tensors)

        if self.cfg.ORVIT.MOTION_STREAM_SEP_POS_EMB:
            shape = (self.nb_frames, self.cfg.ORVIT.O, self.in_dim)
            box_categories = self.box_categories_T.expand(shape) + self.box_categories_O.expand(shape)
        else:
            box_categories = self.box_categories
        box_emb = box_categories.unsqueeze(0).expand(BS, -1, -1, -1) + box_emb # [BS, T, O, d]
        oshape = box_emb.shape
        box_emb = box_emb.flatten(1,-2)
        box_emb, _  = self.attn(box_emb, None, None) # [BS, T, O,d]
        box_emb = box_emb.reshape(oshape)

        box_emb = self.obj2spatial(box_emb, None, box_tensors, H, W, t_avg_pooling=True) # [BS, T, H, W, d]
        return box_emb

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
            print(f"Use DropPath in projector: {drop_path}")
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
            num_frames=8, tubelet_size=1, norm_layer=None
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
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
    
    def forward(self, x):
        x = self.proj(x) # 1,768,8,16,16
        
        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        # ipdb.set_trace()
        x = self.norm(x)
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
        ):
        super().__init__()
        
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        print(mlp_ratio)
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        
        # if use_fused_rmsnorm: 
        #     norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        # else:
        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        print(f"Droppath rate: {dpr}")
        print(f"Checkpoint list: {with_cp_list}")
        
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
        self.orvit = ORViT(cfg=_C)
        self.fc_norm = nn.LayerNorm(clip_embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(clip_embed_dim, num_classes)

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def init_pos_embed(self):
        print("Init pos_embed from sincos pos_embed")
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
    
    def forward(self, x,metadata):
        thw = torch.tensor([8,16,16])
        BS = x.shape[0]
        x = self.patch_embed(x.type(self.dtype))
        B, T, L, C = x.shape  # T: temporal; L: spatial
        x = x.view([B, T * L, C])
        
        
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
        
        x,_ = self.orvit(x,metadata,thw)
        
        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        # ipdb.set_trace() # 1,2049,768
        x = self.clip_projector(x) # 1,768
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x

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
        # if 'module' in k:
        #     name = k[7:]
        #     new_state_dict[name] = v 
        new_state_dict[k] = v

    shared_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(shared_dict)

    # 下面两行，test和训练注释掉
    del model_dict['head.weight']  # 改变分类数量
    del model_dict['head.bias']
    print("re-initialize fc layer")
    print("ckpt key lens{}".format(len(shared_dict.keys())))
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
        # model = get_updateModel(model,"/mnt/fuxiancode/internvideo2/checkpoint/activitynet-internvideo2-bs2-lr1e-05-lrdecay2-hlr0.001-92.306.pth")
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


if __name__ == '__main__':
    import time
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np
    model = internvideo2_base_patch14_224()
    input = torch.rand(1,3,8,224, 224)  # [batch size, channl, frames, H, W]
    box = torch.rand(1,4,8,4)
    gflops, params = profile(model, inputs=(input,box))
    gflops = gflops / 1e9
    gflops, params = clever_format([gflops, params], "%.3f")
    print(gflops,params)

