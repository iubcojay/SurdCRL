"""
UniFormerV2 Common Components

This module contains shared components used across different UniFormerV2 implementations
to reduce code duplication.

Classes:
    - QuickGELU: Fast GELU activation function
    - Local_MHRA: Local Multi-Head Relation Aggregator
    - ResidualAttentionBlock: Residual attention block
    - Extractor: Feature extractor with cross-attention
    - Transformer: Transformer backbone
    - UniFormerV2context: Context feature extraction backbone
    - UniFormerV2obj: Object feature extraction backbone
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from timm.models.layers import trunc_normal_

from .utils import ObjectsCrops

logger = MMLogger.get_current_instance()

from scipy.spatial.distance import mahalanobis
class VideoFrameChangeModule(nn.Module):
    def __init__(self, mlp_hidden_dim, output_dim, distance_type='diff'):
        super(VideoFrameChangeModule, self).__init__()
        self.distance_type = distance_type
        
        # 初始化不同距离计算需要的组件
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.mahalanobis_estimator = None  # 用于存储马氏距离的协方差逆矩阵
        # 可学习的位置编码方式
        # scale = 768**-0.5
        # self.time_pos_embed = nn.Parameter(torch.zeros(1, 4, 8, 1, 768))
        # 增强的MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(768, mlp_hidden_dim),
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
            return distance.expand(*distance.shape[:-1], 768)
        return distance
    
    def _init_mahalanobis(self, features):
        with torch.no_grad():
            flat_features = features.reshape(-1, 768).float().to(features.device)
            cov = torch.cov(flat_features.T, correction=0)
            cov_reg = cov + 1e-6 * torch.eye(768, device=features.device)
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

    def forward(self, x: torch.Tensor,option) -> torch.Tensor:
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        B = N // 4
        H = W = int((L - 1)**0.5)

        if option == 'context':
            cls_token = self.temporal_cls_token.repeat(1, N, 1)
        else:
            cls_token = self.temporal_cls_token.repeat(1, B, 1)
        j = -1
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down)

            # ipdb.set_trace()
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
                
                if option == 'context':
                    tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                else:
                    
                    tmp_x = tmp_x.reshape(L,B,4,T_down,C)
                    tmp_x = tmp_x.permute(2,3, 0, 1, 4).flatten(0,2)
                # 
                # ipdb.set_trace()
                # tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x) # 1 2 768
        weight = self.sigmoid(self.balance)
        if option == 'context':
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C  2 768
        else:
            residual = x.view(L, B, 4 * T_down, C)[0].mean(1)  # L, N, T, C  2 768
        out = self.norm((1 - weight) * cls_token[0, :, :] + weight * residual)
        return out,x
# @MODELS.register_module()
class UniFormerV2context(BaseModule):
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
        mlp_dropout: List[float] = [0.4, 0.4, 0.4, 0.4],
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
        # self.get_updateModel("/mnt/wangyuqing/weights/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics700-rgb_20230313-69070837.pth")

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
        model_dict = self.state_dict() 
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
            if 'module' in k:
                name = k[7:]
                new_state_dict[name] = v 

        shared_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(shared_dict)
        # Optionally adjust classifier layer for different class counts
        logger.info("re-initialize fc layer")
        logger.info("ckpt key lens{}".format(len(shared_dict.keys())))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #ipdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid] 2 768 8 14 14
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)# 16 196 768

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype) # 16 197 768

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND 197 16 768 
        out,out1 = self.transformer(x,'context') # 2 768

        return out,out1
# @MODELS.register_module()
class UniFormerV2obj(BaseModule):
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
        mlp_dropout: List[float] = [0.4, 0.4, 0.4, 0.4],
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

        self.crop_layer = ObjectsCrops()
        self.box_categories = nn.Parameter(torch.zeros(8, 4, width))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4,  width// 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, width, bias=False),
            nn.ReLU()
        )
        self.vfcm = VideoFrameChangeModule(n_dim * 4, n_dim,"manhattan")
        self.objagg =  Extractor(
                n_dim,
                n_head,
                mlp_factor=4,
                dropout=0.0,
                drop_path=0.0,
            )
        self.obj_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.ln = nn.LayerNorm(n_dim)
        self.apply(self._init_weights)
        # self.get_updateModel("/mnt/wangyuqing/weights/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics700-rgb_20230313-69070837.pth")


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

        model_dict = self.state_dict() 
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
            if 'module' in k:
                name = k[7:]
                new_state_dict[name] = v 
        shared_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(shared_dict)
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

    def forward(self, x: torch.Tensor,metadata: torch.Tensor) -> torch.Tensor:
        box_tensors = metadata
        box_tensors = box_tensors.permute(0,2, 1, 3)

        box_tensors_normalized = box_tensors / 224.0  # 缩放到 [0, 1]
        BS = box_tensors_normalized.shape[0]
        box_emb = self.c_coord_to_feature(box_tensors_normalized)
        box_categories = self.box_categories
        box_emb = box_categories.unsqueeze(0).expand(BS, -1, -1, -1) + box_emb # [BS, T, O, d]
        

        x = self.conv1(x)  # shape = [*, width, grid, grid] 2 768 8 14 14
        """
        boxes: List[Tensor[T, N_OBJ, 4]]
        features: [BS, d, T, H=14, W=14]
        """
        x = self.crop_layer(x, box_tensors)  # [BS, O, T, d, H, W]
        N,O,T,C,H,W = x.shape

        # Align box embeddings to feature map spatial dimensions
        box_emb_aligned = box_emb.permute(0, 2, 1, 3)  # [BS, O, T, d]
        box_emb_aligned = box_emb_aligned.reshape(N*O, T, C, 1, 1)  # [N*O, T, C, 1, 1]
        box_emb_aligned = box_emb_aligned.permute(0, 2, 1, 3, 4)  # [N*O, C, T, 1, 1]
        x = x.reshape(N*O,T,C,H,W).permute(0,2,1,3,4)
        x = x + box_emb_aligned  # [N*O, C, T, H, W]
        
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)# 16 196 768
        

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],dim=1)  # shape: [*, grid**2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)  # example: [16, 197, 768]
        # Optional: apply per-token weighting here if needed
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, e.g., [197, 16, 768]
        x, y = self.transformer(x, 'obj')  # global token + features
        # Object temporal message passing: restore Transformer outputs to [B, O, T, L, C]
        L= H * W + 1
        y = y.view(L, N, T, C)
        y = y.permute(1, 2, 0, 3).view(BS, O, T, L, C) 
        ym, yt = self.vfcm(y)  # ym: updated sequence, yt: temporal-change features, e.g. [1, 4, 7, 197, 768]
        # Normalize to [T'*L, B*O, C] for the global aggregation module
        yt = yt.permute(2, 3, 0, 1, 4).reshape((T - 1) * L, BS * O, C)

        cls_token = self.obj_cls_token.repeat(1, BS*O, 1)
        yt = self.objagg(cls_token,yt)[0, :, :].view(BS,O,768)
        x = x.unsqueeze(1).expand(-1, O, -1)
        obj = x + self.ln(yt)
        return obj,y
