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
from torch.nn import Parameter

import numpy as np
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
    
    def forward(self, x):
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
        z = self.fc_norm(x)
        y = self.head(self.fc_dropout(z))
        return x,y

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



class Video_Classification(nn.Module):
    """Video classification model with causal intervention modules.
    
    Args:
        num_classes: Number of output classes
        feature_dim: Feature dimension for embeddings (default: 768)
        num_frames: Number of temporal frames (default: 8)
        causal_feature_multiplier: Multiplier for causal feature concatenation (default: 4)
        joint_feature_count: Number of features to concatenate for joint representation (default: 3)
    """
    def __init__(self, num_classes=20, feature_dim=768, num_frames=8,
                 causal_feature_multiplier=4, joint_feature_count=3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.causal_feature_multiplier = causal_feature_multiplier
        self.joint_feature_count = joint_feature_count
        
        # Feature extraction modules
        self.module1 = internvideo2_base_patch14_224(pretrained=True, num_classes=num_classes) 
        self.module2 = internvideo2_base_patch14_224(pretrained=True, num_classes=num_classes) 
        self.module3 = internvideo2_base_patch14_224(pretrained=True, num_classes=num_classes)
        
        # Causal intervention modules
        self.frontIntervention = LGCAM(module_dim=self.feature_dim)
        self.backIntervention = CCIM(self.feature_dim, self.feature_dim, 
                                      strategy='dp_cause', num_class=num_classes)
        
        # Temporal attention modules (commented out in original)
        # self.bgTemporal = TemporalSelfAttention_bg(self.feature_dim, self.feature_dim, self.num_frames)
        # self.objTemporal = TemporalSelfAttention_obj(self.feature_dim, self.feature_dim, self.num_frames)

        # Fusion and classification layers
        self.dense = nn.Linear(self.feature_dim * self.causal_feature_multiplier, self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        self.dense2 = nn.Linear(self.feature_dim * self.joint_feature_count, self.feature_dim)
        self.fc1 = nn.Linear(self.feature_dim, num_classes)
        self.fc2 = nn.Linear(self.feature_dim, num_classes)
        
        # self.sigmoid = nn.Sigmoid()
        
        print(f"Initialized Video_Classification with {num_classes} classes")


    def forward(self, frames,obj, bg,global_feat_dict,confounder_dictionary,prior):
        video_feature,logits1 = self.module1(frames)  #joint_feature为uniformer提取的特征，logist为预测
        local_feat = video_feature
        
        # 从global全局字典当中随机选择与batchsize大小相同的样本
        batchsize = local_feat.shape[0]
        num_samples, sample_dim = global_feat_dict.shape
        num_requested_samples = batchsize #与batchsize的大小相同
        sample_indices = np.random.choice(num_samples, num_requested_samples, replace=False)  # 从字典中随机抽取 16 个样本的索引
        global_feat = global_feat_dict[sample_indices]  # 使用随机选择的索引获取对应的样本数据
        
        #尝试1：扩充local_feature 使其从768维度变为1536
        local_feat = torch.cat([local_feat,local_feat],dim=1)
        global_feat = torch.cat([global_feat,global_feat],dim=1)
        causal_feature = self.frontIntervention(local_feat,global_feat)
        causal_feature = self.dense(causal_feature)
        lg_logits = self.fc(causal_feature)
        # 加入CCIM模块
        bg_feature_not,logits2 = self.module3(bg) 
        # bg_feature_t = self.bgTemporal(bg_feature_t)  #bg_t:bs,t,768
        # bg_feature_t = torch.mean(bg_feature_t,dim=1)
        # bg_feature_t = bg_feature_t.squeeze(dim=1) #bs*768
        bg_feature = bg_feature_not
        
        obj_feature_not,logits2 = self.module2(obj) 
        # obj_featuret = self.objTemporal(obj_featuret)  #bg_t:bs,t,768
        # obj_featuret = torch.mean(obj_featuret,dim=1)
        # obj_featuret = obj_featuret.squeeze(dim=1) #bs*768
        obj_feature = obj_feature_not
        
        h_feature = torch.cat([video_feature,bg_feature,obj_feature],dim=1) #bs,768*2
        joint_feature = self.dense2(h_feature) ##bs,768
        h_logits = self.fc1(joint_feature) ##原来两者的融合的预测
        
        #加入因果干预模块,计算gz_logits
        gz = self.backIntervention(joint_feature, confounder_dictionary, prior) #gz表示的是加权之后的特征 gz: bs,768
        gz_logits = self.fc2(gz) #上下文加权积分的得分预测
        logits = logits1
        # logits = h_logits + lg_logits + gz_logits + logits1   
        
        return logits

#=====================================背景的时间关联建模模块=============================================================================

class TemporalSelfAttention_bg(nn.Module):
    def __init__(self, input_size, hidden_size, num_frames):
        super(TemporalSelfAttention_bg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames

        # 将输入向量映射到Query、Key和Value向量
        self.query_projection = nn.Linear(input_size, hidden_size)
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.value_projection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
      
        # x的维度为 (batch_size, num_frames, input_size)

        # 计算Query、Key和Value向量
        queries = self.query_projection(x)  # (batch_size, num_frames, hidden_size)
        keys = self.key_projection(x)  # (batch_size, num_frames, hidden_size)
        values = self.value_projection(x)  # (batch_size, num_frames, hidden_size)

        # 计算Attention分数
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (
                    self.hidden_size ** 0.5)  # (batch_size, num_frames, num_frames)
        attention_weights = F.softmax(attention_scores, dim=-1)  # 在时间维度上进行softmax得到注意力权重

        # 加权求和得到加权后的帧表示
        weighted_sum = torch.bmm(attention_weights, values)  # (batch_size, num_frames, hidden_size)

        return weighted_sum


class TemporalSelfAttention_obj(nn.Module):
    def __init__(self, input_size, hidden_size, num_frames):
        super(TemporalSelfAttention_obj, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames

        # 将输入向量映射到Query、Key和Value向量
        self.query_projection = nn.Linear(input_size, hidden_size)
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.value_projection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
       
        # x的维度为 (batch_size, num_frames, input_size)

        # 计算Query、Key和Value向量
        queries = self.query_projection(x)  # (batch_size, num_frames, hidden_size)
        keys = self.key_projection(x)  # (batch_size, num_frames, hidden_size)
        values = self.value_projection(x)  # (batch_size, num_frames, hidden_size)

        # 计算Attention分数
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (
                    self.hidden_size ** 0.5)  # (batch_size, num_frames, num_frames)
        attention_weights = F.softmax(attention_scores, dim=-1)  # 在时间维度上进行softmax得到注意力权重

        # 加权求和得到加权后的帧表示
        weighted_sum = torch.bmm(attention_weights, values)  # (batch_size, num_frames, hidden_size)

        return weighted_sum

#=============================================================CCIM因果模块=================================================================
class CCIM(nn.Module):
    def __init__(self, num_joint_feature, num_gz, strategy,num_class): #num_joint_feature d=1536 num_gz;768
        super(CCIM, self).__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz
        if strategy == 'dp_cause':
              self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature )  #768,1536
        elif strategy == 'ad_cause':
              self.causal_intervention = additive_intervention(num_gz, num_joint_feature )
        else:
              raise ValueError("Do Not Exist This Strategy.")
                
        self.output_dim = self.num_gz  # Output dimension matches num_gz
        self.w_h = Parameter(torch.Tensor(self.num_joint_feature, self.output_dim)) 
        self.w_g = Parameter(torch.Tensor(self.num_gz, self.output_dim)) 
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    def forward(self, joint_feature, confounder_dictionary, prior):  #20,768   #20*1
#         ipdb.set_trace()
        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)  #bs,768
        # return g_z
        do_x = joint_feature + g_z
#         proj_h = torch.matmul(joint_feature, self.w_h)  #1536，768=》768
#         proj_g_z = torch.matmul(g_z.to(torch.float32), self.w_g) 
#         do_x = proj_h + proj_g_z
        return do_x
    
class dot_product_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super(dot_product_intervention, self).__init__()
        self.con_size = con_size 
        self.fuse_size = fuse_size  
        self.query = nn.Linear(self.fuse_size, 768, bias= False) 
        self.key = nn.Linear(self.con_size, 768, bias= False) 

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
    def __init__(self, con_size, fuse_size):
        super(additive_intervention,self).__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.Tan = nn.Tanh()
        self.query = nn.Linear(self.fuse_size, 768, bias = False)
        self.key = nn.Linear(self.con_size, 768, bias = False)
        self.w_t = nn.Linear(768, 1, bias=False)

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
        
#==================================================================LGCAM模块======================================================================
        
class LGCAM(nn.Module):
    def __init__(self, module_dim=768):
        super(LGCAM, self).__init__()
        self.feature_LL = FeatureAggregation_LL(module_dim=768)
        self.feature_LG = FeatureAggregation_LG(module_dim=768)
        
    def forward(self,local_feature,global_feature):
        
        ll_feature = self.feature_LL(local_feature)
        lg_feature = self.feature_LG(local_feature,global_feature)
        
        causal_feature = torch.cat([ll_feature,lg_feature],dim=1)
        
        return causal_feature

class FeatureAggregation_LL(nn.Module):
    def __init__(self, module_dim=768):
        super(FeatureAggregation_LL, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self,local_feature):
        # breakpoint()
#         ipdb.set_trace()
        local_feature = self.dropout(local_feature)
        q_proj = self.q_proj(local_feature)
        v_proj = self.v_proj(local_feature)

        v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * local_feature)

        return v_distill



class FeatureAggregation_LG(nn.Module):
    def __init__(self, module_dim=768):
        super(FeatureAggregation_LG, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, local_feature,global_feature):
        # breakpoint()
#         ipdb.set_trace()
        global_feature = self.dropout(global_feature)
        q_proj = self.q_proj(local_feature)
        v_proj = self.v_proj(global_feature)

        v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * global_feature)

        return v_distill








if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np
    from thop import profile, clever_format
    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    model = internvideo2_base_patch14_224(pretrained=True,num_classes=20)
    input = torch.rand(1,3,8,224, 224)  # [batch size, channl, frames, H, W]
    gflops, params = profile(model, inputs=(input,))
    gflops = gflops / 1e9
    gflops, params = clever_format([gflops, params], "%.3f")
    print(gflops,params)

