    # Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
import ipdb
from .utils import ObjectsCrops, box2spatial_layout, Mlp
from .attention import TrajectoryAttention, SeltAttentionBlock
# from mmaction.registry import MODELS
from .orvitdefaults import _C
logger = MMLogger.get_current_instance()

# MODEL_PATH = 'https://download.openmmlab.com/mmaction/v1.0/recognition'
# _MODELS = {
#     'ViT-B/16':
#     os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
#                  'vit-base-p16-res224_clip-rgb_20221219-b8a5da86.pth'),
#     'ViT-L/14':
#     os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
#                  'vit-large-p14-res224_clip-rgb_20221219-9de7543e.pth'),
#     'ViT-L/14_336':
#     os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
#                  'vit-large-p14-res336_clip-rgb_20221219-d370f9e5.pth'),
# }


class QuickGELU(BaseModule):
    """Quick GELU function. Forked from https://github.com/openai/CLIP/blob/d50
    d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py.

    Args:
        x (torch.Tensor): The input features of shape :math:`(B, N, C)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Local_MHRA(BaseModule):
    """Local MHRA.

    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        dw_reduction: float = 1.5,
        pos_kernel_size: int = 3,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x)


class ResidualAttentionBlock(BaseModule):
    """Local UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA.
            Defaults to True.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
        dw_reduction: float = 1.5,
        no_lmhra: bool = False,
        double_lmhra: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        logger.info(f'No L_MHRA: {no_lmhra}')
        logger.info(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor, T: int = 8) -> torch.Tensor:
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


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

#         score = aff.view(1,12,1,8,197).sum(dim=-1).permute(0,2,3,1).sum(dim=-1).squeeze(0).squeeze(0)*1e-3
#         score = score.softmax(dim=-1)
#         print(score)
        
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


class Transformer(BaseModule):
    """Backbone:

    Args:
        width (int): Number of input channels in local UniBlock.
        layers (int): Number of layers of local UniBlock.
        heads (int): Number of attention head in local UniBlock.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA
            in local UniBlock. Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        width: int,
        layers: int=12,
        heads: int=12,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        dw_reduction: float = 1.5,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [
            x.item()
            for x in torch.linspace(0, backbone_drop_path_rate, layers)
        ]
        self.resblocks = ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])

        # global block
        assert n_layers == len(return_list)
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.dpe = ModuleList([
            nn.Conv3d(
                n_dim,
                n_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=n_dim) for _ in range(n_layers)
        ])
        for m in self.dpe:
            nn.init.constant_(m.bias, 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dec = ModuleList([
            Extractor(
                n_dim,
                n_head,
                mlp_factor=mlp_factor,
                dropout=mlp_dropout[i],
                drop_path=dpr[i],
            ) for i in range(n_layers)
        ])
        # weight sum
        self.norm = nn.LayerNorm(n_dim)
        self.balance = nn.Parameter(torch.zeros((n_dim)))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
#         ipdb.set_trace()
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1)**0.5)
        cls_token = self.temporal_cls_token.repeat(1, N, 1)

        j = -1
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # ipdb.set_trace()
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2, 0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats.clone()).view(
                    N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x) # 1 2 768

        weight = self.sigmoid(self.balance)
        #ipdb.set_trace()
        residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C  2 768
        out = self.norm((1 - weight) * cls_token[0, :, :] + weight * residual)
        return out
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




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
        # box_tensors = metadata['orvit_bboxes']
        box_tensors = metadata
        box_tensors = box_tensors.permute(0,2, 1, 3)
        box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
        BS,O = box_tensors_normalized.shape[0],box_tensors_normalized.shape[2]
        assert box_tensors is not None
        patch_tokens = x
        if self.with_cls_token:
            # 2 197 8*768
            cls_token, patch_tokens = x[:,[0]], x[:,1:]
        d = self.dim
        # BSxT, _, d = x.shape # 
        T,H,W = thw
        # patch_tokens = patch_tokens.permute(0,2,1).reshape(BS, -1, T,H,W) # 可用conv1后的代替 2 768 8 14 14
        patch_tokens = patch_tokens.reshape(BS, T, H, W, d)  # [BS, 8, 14, 14, 768]
        patch_tokens = patch_tokens.permute(0, 4, 1, 2, 3)  # [BS, 768, 8, 14, 14]
        
        BS, d, T, H, W = patch_tokens.shape
        assert T == self.nb_frames
        
        # Tratio = box_tensors.shape[1]
        # box_tensors = box_tensors[:,::Tratio] # [BS, T , O, 4]
        # O = box_tensors.shape[-2]
        object_tokens = self.crop_layer(patch_tokens, box_tensors)  # [BS, O,T, d, H, W]
        object_tokens = object_tokens.permute(0, 1,2,4,5,3)  # [BS, O,T, H, W, d]
        object_tokens = self.patch_to_d(object_tokens) # [BS,O,T, H, W, d]
        object_tokens =torch.amax(object_tokens, dim=(-3,-2)) # [BS, O,T, d]
        object_tokens = object_tokens.permute(0,2,1,3) # [BS, T, O, d]
        
        # object_tokens = object_tokens.reshape(BS*T, O, d)
        
        box_categories = self.box_categories.unsqueeze(0).expand(BS,-1,-1,-1)
        box_emb = self.c_coord_to_feature(box_tensors)

        # all_tokens = torch.cat([patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d), object_tokens], dim = 2).flatten(1,2) # [BS, T * (H*W+O),d]
        all_tokens = torch.cat([patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d), object_tokens], dim = 2).flatten(0,1)
        # if self.with_cls_token:
        #     all_tokens =  torch.cat([cls_token, all_tokens], dim = 1) # [BS, 1 + T*N, d]

        all_tokens, thw = self.attn(
                    self.norm1(all_tokens), 
                    [T, H*W + O, 1], 
                )
        
        # if self.with_cls_token:
        #     cls_token, all_tokens =  all_tokens[:, [0]], all_tokens[:, 1:]

        patch_tokens = all_tokens.reshape(BS,T,H*W+O,d)[:,:,:H*W].reshape(BS*T,H*W,d)

        if self.with_motion_stream:
            box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
            motion_emb = self.motion_stream(box_tensors_normalized,H, W) # [BS, T, H, W, d]
            motion_emb = self.motion_mlp(motion_emb) # [BS, T*H*W, d]
            patch_tokens = patch_tokens + motion_emb

        
        if self.with_cls_token:
            patch_tokens = torch.cat([cls_token, patch_tokens], dim = 1) # [BS, 1 + N, d]
        # print(torch.mean(torch.abs(x - patch_tokens)))  # 非零，表示 token 表征内容不同
        # x = x.reshape(BS*T,H*W,d)
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

# @MODELS.register_module()
class UniFormerV2(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        num_classes = 20,
        # pretrain
        clip_pretrained: bool = False,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.t_size = t_size
        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )
        self.orvit = ORViT(cfg=_C)
        self.classifer = nn.Linear(768,num_classes)
        # self.get_updateModel("/mnt/wangyuqing/weights/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb.pth") 
        self.get_updateModel("/mnt/wangyuqing/weights/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics700-rgb_20230313-69070837.pth")
        # self.get_updateModel("/mnt/wangyuqing/weights/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20230612-63cdbad9.pth") 
        # self.get_updateModel("./uniformerv2/checkpoint/finalcheck/msrvtt-uniformerv2-8-1e-05-4.pth") 
        # self.get_updateModel("./uniformerv2/checkpoint/finalcheck/activitynet-uniformerv2-8-1e-05-4-m1.pth") 
    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d
    
    
    
    def get_updateModel(self, path): 
        
        pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
        # pretrained_dict = torch.load(path, map_location ='cpu') # 自己训练的模型
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


    

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x: torch.Tensor, metadata) -> torch.Tensor:
        #ipdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid] 2 768 8 14 14
        N, C, T, H, W = x.shape
        thw = torch.tensor([8,14,14])
        # x,_ = self.orvit(x,metadata,thw)
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)# 16 196 768

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype) # 16 197 768
        
        # 可以在这里添加权重代码！
        x = self.ln_pre(x)
        x,_ = self.orvit(x,metadata,thw)
        x = x.permute(1, 0, 2)  # NLD -> LND 197 16 768 
        out = self.transformer(x) # 2 768
        out = self.classifer(out) # 2 20
        return out
    
    
if __name__ == '__main__':
    import time
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np
    model = UniFormerV2()
    input = torch.rand(1,3,8,224, 224)  # [batch size, channl, frames, H, W]
    box = torch.rand(1,4,8,4)
    gflops, params = profile(model, inputs=(input,box))
    gflops = gflops / 1e9
    gflops, params = clever_format([gflops, params], "%.3f")
    print(gflops,params)

