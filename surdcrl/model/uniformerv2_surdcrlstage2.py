    # Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union
import torch.nn.functional as F
import torch
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
import ipdb
from .utils import ObjectsCrops
from .ops import extend_neg_masks
from .uniformerv2_common import (
    QuickGELU, Local_MHRA, ResidualAttentionBlock, 
    Extractor, Transformer, UniFormerV2context, UniFormerV2obj
)
from itertools import combinations
# from mmaction.registry import MODELS
import itertools
import numpy as np
from torch.nn import Parameter,Dropout
logger = MMLogger.get_current_instance()
token_objects = [
    [],              # idx 0: cls_token
    [0,1], [0,2], [0,3], [0,4],        # idx 1~3: second-order
    [1,2], [1,3], [1,4], [2,3], [2,4], [3,4],  # idx 4~10: second-order (10)
    [0,1,2], [0,1,3], [0,1,4], [0,2,3], [0,2,4], [0,3,4], [1,2,3], [1,2,4], [1,3,4], [2,3,4],  # third-order (10)
    [0,1,2,3], [0,1,2,4], [0,1,3,4], [0,2,3,4], [1,2,3,4],  # fourth-order (5)
    [0,1,2,3,4]  # idx 26: fifth-order (all)
]


from .Bert_backbone import *

import json
from types import SimpleNamespace

with open("/mnt/mymodel/work2/exp3-2/model/reverie_GOAT_model_config.json", "r") as f:
    config_dict = json.load(f)

config = SimpleNamespace(**config_dict)



class FrontDoorEncoder(nn.Module):
    def __init__(self, config, mode='batch_shared_group', num_prototypes=8):
        """
        :param mode: 'batch_shared_group' | 'point2group' | 'group2group'
        :param num_prototypes: Number of prototypes sampled per sample (for group2group or batch_shared_group)
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
        # ----- Front-door sampling -----
        if self.mode == 'batch_shared_group':
            # All batch shares one set of prototypes
            sample_indices = np.random.choice(N, self.num_prototypes, replace=False)
            # print(sample_indices)
            sampled = global_feat_dict[sample_indices]      # [num_prototypes, d]
            global_feats = sampled.unsqueeze(0)             # [1, P, d], auto broadcast to [bs, P, d]
            
        elif self.mode == 'point2group':
            # One prototype per sample
            sample_indices = np.random.choice(N, bs, replace=True)
            global_feats = global_feat_dict[sample_indices].unsqueeze(1)  # [bs, 1, d]

        elif self.mode == 'group2group':
            # Multiple prototypes per sample
            sample_indices = np.random.choice(N, bs * self.num_prototypes, replace=True)
            sample_indices = sample_indices.reshape(bs, self.num_prototypes)  # [bs, P]
            global_feats = torch.stack([
                global_feat_dict[idx] for idx in sample_indices
            ], dim=0)  # [bs, P, d]
        elif self.mode == 'similarity_topk':
            # local_feats: [bs, L, d], each L represents a target feature
            target_feats = local_feats  # [bs, L, d]
            BS, O, D = target_feats.shape
            global_feats_norm = F.normalize(global_feat_dict, dim=-1)    # [N, d]
            target_feats_norm = F.normalize(target_feats, dim=-1)        # [bs, O, d]
        
            # Similarity computation: [bs, O, N]
            sim = torch.matmul(target_feats_norm, global_feats_norm.T)
        
            # Select top-k most similar prototypes for each target
            topk_idx = torch.topk(sim, self.num_prototypes, dim=-1).indices  # [bs, O, K]
        
            # Sample corresponding global_feats
            global_feats = torch.gather(
                global_feat_dict.unsqueeze(0).unsqueeze(0).expand(BS, O, global_feat_dict.size(0), D),
                dim=2,
                index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            )  # [bs, O, K, d]
        
            global_feats = global_feats.view(BS, O * self.num_prototypes, D)  # flatten to [bs, O*K, d]


        
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'.")

        # ----- Front-door attention -----
        if local_feats_masks is not None and len(local_feats_masks.shape) != 4:
            local_feats_masks = extend_neg_masks(local_feats_masks)
        ll_feats = self.ll_self_attn(local_feats, attention_mask=local_feats_masks)[0]
        lg_feats = self.lg_cross_attn(hidden_states=local_feats, encoder_hidden_states=global_feats)[0]
        out_feats = self.ln(ll_feats + lg_feats)

        # ----- Gated fusion -----
        aug_weight = self.sigmoid(self.aug_linear(out_feats) + self.ori_linear(local_feats))
        # print(f'out_feats: {aug_weight.mean():.4f}', f'local_feats: {(1 - aug_weight).mean():.4f}')
        out_feats = aug_weight * out_feats + (1 - aug_weight) * local_feats

        return out_feats

class LGCAM(nn.Module):
    def __init__(self, module_dim=768):
        super(LGCAM, self).__init__()
        self.feature_LL = FeatureAggregation_LL(module_dim=module_dim)
        self.feature_LG = FeatureAggregation_LG(module_dim=module_dim)
        self.dense = nn.Sequential(
                nn.Linear(module_dim * 4, module_dim),
                nn.ELU(),
                # nn.Linear(768, 768)
            )
        
    def forward(self,local_feature,global_feature):
        # ipdb.set_trace()
        ll_feature = self.feature_LL(local_feature)
        lg_feature = self.feature_LG(local_feature,global_feature)
        
        causal_feature = torch.cat([ll_feature,lg_feature],dim=1)
        causal_feature = self.dense(causal_feature)
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
        self.dropout = nn.Dropout(0.0)

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
        self.dropout = nn.Dropout(0.0)

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
#============================================================= CCIM Causal Module =================================================================
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
        # self.aug_linear = nn.Linear(768, 768)
        # para
        self.norm = nn.LayerNorm(self.feature_dim)
        self.balance = nn.Parameter(torch.zeros((self.feature_dim)))
        self.sigmoid = nn.Sigmoid()
        # gate
        self.aug_linear = nn.Linear(self.feature_dim, self.feature_dim)
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
        joint_feature = self.aug_linear(joint_feature)


        # para
        weight = self.sigmoid(self.balance)
        do_x = self.norm((1 - weight) * joint_feature + weight * g_z)
        do_x = do_x.unsqueeze(1)
        return do_x

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

from scipy.spatial.distance import mahalanobis
class VideoFrameChangeModule(nn.Module):
    def __init__(self, mlp_hidden_dim, output_dim, distance_type='diff', feature_dim: int = 768):
        super(VideoFrameChangeModule, self).__init__()
        self.distance_type = distance_type
        self.feature_dim = feature_dim
        
        # Initialize components for different distance computations
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.mahalanobis_estimator = None  # store inverse covariance for Mahalanobis distance
        
        # Enhanced MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )
        # self.ln = nn.LayerNorm(768)
        # Unique buffer name + safe registration
        self.register_buffer('_mahalanobis_inv_cov', None)  # critical fix
    def _compute_distance(self, prev, curr):
        """Compute different distances according to configuration."""
        if self.distance_type == 'diff':
            # Direct element-wise difference [..., C]
            return prev - curr
            
        elif self.distance_type == 'manhattan':
            # Manhattan distance [..., 1]
            distance = torch.sum(torch.abs(prev - curr), dim=-1, keepdim=True)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'euclidean':
            # Euclidean distance [..., 1]
            distance = torch.norm(prev - curr, p=2, dim=-1, keepdim=True)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'cosine':
            # Cosine distance [..., 1]
            prevclone = prev.clone()
            currclone = curr.clone()
            cos_sim = torch.cosine_similarity(prevclone, currclone, dim=-1, eps=1e-6)
            distance = 1 - cos_sim.unsqueeze(-1)
            return self._expand_distance(distance)
            
        elif self.distance_type == 'mahalanobis':
            if self._mahalanobis_inv_cov is None:
                self._init_mahalanobis(prev)
            
            diff = (prev - curr).reshape(-1, self.feature_dim)
            VI = self._mahalanobis_inv_cov
            mahal = torch.sqrt(torch.einsum('bi,ij,bj->b', diff, VI, diff))
            distance = mahal.view(*prev.shape[:-1], 1)
            return self._expand_distance(distance)
            
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def _expand_distance(self, distance):
        """Expand scalar distance to feature_dim dimensions."""
        if distance.shape[-1] == 1 and self.distance_type != 'diff':
            # Simple replication strategy (could be replaced by learnable MLP expansion)
            return distance.expand(*distance.shape[:-1], self.feature_dim)
        return distance
    
    def _init_mahalanobis(self, features):
        with torch.no_grad():
            flat_features = features.reshape(-1, self.feature_dim).float().to(features.device)
            cov = torch.cov(flat_features.T, correction=0)
            cov_reg = cov + 1e-6 * torch.eye(self.feature_dim, device=features.device)
            self._mahalanobis_inv_cov = torch.linalg.pinv(cov_reg)  # direct assignment

    def forward(self, xo):
        """Process input features and compute inter-frame changes
        Args:
            x: (batch_size, num_targets, num_frames, L, feature_dim)
        """
        x = xo.clone()
        # x = self.ln(x)
        # Correct shape unpacking (original code may have dimension mismatch)
        batch_size, num_targets, num_frames, L, feature_dim = x.shape
        
        # Store all change features
        change_features = []

        # Iterate through time steps to compute changes
        for t in range(1, num_frames):
            prev_frame = x[:, :, t-1]  # [B, N, L, D]
            curr_frame = x[:, :, t]    # [B, N, L, D]
            
            # Compute inter-frame changes (according to configuration type)
            change = self._compute_distance(prev_frame, curr_frame)
            
            # Process change features through MLP
            change_feature = self.mlp(change)
            
            # Residual connection to update current frame
            x[:, :, t] = curr_frame + change_feature
            
            # Save change features
            change_features.append(change_feature.unsqueeze(2))
        
        # Concatenate change features along time dimension
        final_change_features = torch.cat(change_features, dim=2)
        return x, final_change_features


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

class CrossAttention(BaseModule):
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
            )
            self.shared_fc3 = nn.Sequential(
                nn.Linear(self.feature_dim * 3, self.feature_dim),
                nn.ELU(),
            )
            self.shared_fc4 = nn.Sequential(
                nn.Linear(self.feature_dim * 4, self.feature_dim),
                nn.ELU(),
            )
            self.shared_fc5 = nn.Sequential(
                nn.Linear(self.feature_dim * 5, self.feature_dim),
                nn.ELU(),
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

        self.crossAttention = nn.ModuleList([CrossAttention(self.feature_dim, num_heads) for _ in range(attnLayer)])
        if use_mean_pooling == False:
            self.symodule = nn.ModuleList([multiAgg(self.feature_dim, num_heads, mlp_factor=4, dropout=0.0, drop_path=0.0) for _ in range(1)])
            self.remodule = nn.ModuleList([multiAgg(self.feature_dim, num_heads, mlp_factor=4, dropout=0.0, drop_path=0.0) for _ in range(1)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.syclstoken = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.reclstoken = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(len(token_objects), 1, self.feature_dim))  # [cls] + |tokens|
        
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

        combo2idx = {
            tuple(sorted(combo)): idx
            for idx, combo in enumerate(token_objects[1:], start=1)
        }
    

        attn_mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)
    

        attn_mask[0, 1:] = False  
        attn_mask[1:, 0] = False  
    

        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            L = len(current_combo)
            for r in range(2, L): 
                for subset in combinations(current_combo, r):
                    subset = tuple(sorted(subset))
                    if subset in combo2idx:
                        j = combo2idx[subset]
                        attn_mask[i, j] = False
            attn_mask[i, i] = False  
    

        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            current_key = tuple(sorted(current_combo))
            for key, j in combo2idx.items():
                if set(current_key).issubset(set(key)) and key != current_key:
                    attn_mask[i, j] = False  
                    


        for i in range(1, total_tokens):
            current_combo = token_objects[i]
            for j in range(1, total_tokens):
                if i == j:
                    continue
                other_combo = token_objects[j]
                if len(current_combo) == len(other_combo):
                    if set(current_combo) & set(other_combo):  
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

        print(bin_idx)
        # ipdb.set_trace()
        if STE == True:
            # -- One-hot mask --
            hard_mask = torch.zeros_like(p).scatter(1, bin_idx, 1.0)  # [bs, 10], 多个 1

            soft_mask = (hard_mask - p).detach() + p  
        else:

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

        context = self.bd(context, bg_confounder_dictionary, bg_prior)
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
            attn_mask = self.build_structured_attn_mask(token_objects,total_tokens=27)  # [27, 27]
        else:
            attn_mask = torch.zeros(len(token_objects), len(token_objects), dtype=torch.bool)

        # Compositional Structure-aware Attention
        for attn in self.crossAttention:
            rel_input, attn_weights = attn(rel_input,attn_mask=attn_mask) # [27,bs,768],[2,27,27]
        csa = rel_input.permute(1, 0, 2)[:, 0, :] # bs,768
        
        # Adaptive Sampling
        w = attn_weights[:, 0, 1:]
        cdf = torch.cumsum(w, dim=1)
        soft_mask,bin_idx = self.soft_inverse_transform_sampling(cdf, num_samples=self.num_samples,p=w,STE=False)

        relation_ = rel_input.permute(1, 0, 2)[:, 1:, :]
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
    def __init__(self, num_class=200,num_samples=10,attnlayer=8,num_prototypes=8,
                 feature_dim: int = 768, num_objects: int = 4, relation_num_heads: int = 12,
                 causal_classification_mode: str = 'separate'):
        super().__init__()
        self.moduleobj = UniFormerV2obj()
        self.modulecontext = UniFormerV2context()

        self.uniquefc = nn.Sequential(
            nn.Linear(feature_dim*(num_objects+1), feature_dim),
            nn.ELU()
        )
        self.cgfc = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.ELU()
        )
        self.synandred = synandred(use_pos_embedding=True,
                                   use_mean_pooling=True, 
                                   shared_relation_fc=True,
                                   use_structured_attn_mask=True,
                                   used_orders=[2,3,4,5],
                                   num_samples=num_samples,
                                   attnLayer=attnlayer,
                                   num_prototypes=num_prototypes,
                                   feature_dim=feature_dim,
                                   num_heads=relation_num_heads)
        # Store classification mode
        self.causal_classification_mode = causal_classification_mode
        
        # Classification heads for different feature types
        self.unique_classifier = nn.Linear(feature_dim, num_class)  # For unique features
        
        if causal_classification_mode == 'separate':
            # Separate classifiers for synergistic and redundant features
            self.synergistic_classifier = nn.Linear(feature_dim, num_class)
            self.redundant_classifier = nn.Linear(feature_dim, num_class)
        elif causal_classification_mode == 'combined':
            # Single classifier for combined causal graph features
            self.causal_graph_classifier = nn.Linear(feature_dim, num_class)
        else:
            raise ValueError(f"Invalid causal_classification_mode: {causal_classification_mode}. "
                           f"Must be 'separate' or 'combined'.")
    
    def get_updateModel(self, path): 
        # pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_dict = torch.load(path, map_location ='cpu') 
        model_dict = self.state_dict() 
        new_state_dict = OrderedDict()

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


        logger.info("re-initialize fc layer")
        logger.info("ckpt key lens{}".format(len(shared_dict.keys())))
        self.load_state_dict(model_dict, strict=False)
        return self 
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
    def _reshape_object_features(self, obj_features: torch.Tensor) -> torch.Tensor:
        """Select first token and mean over temporal dim in a shape-agnostic way.

        Args:
            obj_features: Tensor with dims that include [batch, time, token, channels].
                          Expected order from upstream: (d0, B, T, L, C).

        Returns:
            Tensor shaped as [B, C] or [B, *, C] depending on upstream aggregation,
            after selecting first token and averaging over time.
        """
        from einops import rearrange
        objres_re = rearrange(obj_features, 'd0 b t l c -> l d0 b t c')
        return objres_re[0].mean(dim=2)

    def _reshape_context_features(self, ctx_features: torch.Tensor, video: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape context features dynamically to [B, 1, C] by averaging over tokens.

        Args:
            ctx_features: Raw context features tensor with unknown flattened shape.
            video: The original video tensor to derive time steps T.
            batch_size: Current batch size B.

        Returns:
            Tensor of shape [B, 1, C].
        """
        C = ctx_features.shape[-1]
        T = video.shape[2]
        L = ctx_features.numel() // (batch_size * T * C)
        ctx = ctx_features.view(L, batch_size, T, C)[0].mean(1).unsqueeze(1)
        return ctx

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        back: torch.Tensor,
        global_feat_dict: Union[torch.Tensor, None] = None,
        bg_confounder_dictionary: Union[torch.Tensor, None] = None,
        bg_prior: Union[torch.Tensor, None] = None,
    ) -> tuple:
        """Forward pass.

        Args:
            x: Video frames [B, 3, T, H, W].
            metadata: Object boxes or metadata tensor.
            back: Background frames [B, 3, T, H, W].
            global_feat_dict: Global prototype dictionary [N, C], optional.
            bg_confounder_dictionary: Background confounder dictionary [K, C], optional.
            bg_prior: Background prior [K, 1], optional.

        Returns:
            Tuple of model outputs as defined by original implementation.
        """
        # Basic shape validation (soft checks to help debugging)
        assert x.ndim == 5, f"x must be [B,3,T,H,W], got {x.shape}"
        assert back.ndim == 5, f"back must be [B,3,T,H,W], got {back.shape}"

        # Extract object and context features
        obj, objres = self.moduleobj(x, metadata)
        context, contextres = self.modulecontext(back)
        context = context.unsqueeze(1)
        BS = obj.shape[0]
        
        # Process unique features (combination of object and context)
        unique_features = torch.cat([obj, context], dim=1).view(BS, -1)  # flatten tokens per sample
        unique_features = self.uniquefc(unique_features)
        unique_logits = self.unique_classifier(unique_features)
        
        # Shape-agnostic feature reshape for causal analysis
        objres = self._reshape_object_features(objres)
        contextres = self._reshape_context_features(contextres, x, BS)
        synergistic, redundant, bin_idx = self.synandred(
            objres, contextres, global_feat_dict, bg_confounder_dictionary, bg_prior
        )

        # Apply causal feature classification based on selected mode
        if self.causal_classification_mode == 'separate':
            # Mode 1: Separate classification for synergistic and redundant features
            synergistic_logits = self.synergistic_classifier(synergistic)
            redundant_logits = self.redundant_classifier(redundant)
            final_logits = unique_logits + synergistic_logits + redundant_logits
            
        elif self.causal_classification_mode == 'combined':
            # Mode 2: Combined causal graph classification
            causal_graph_features = torch.cat([synergistic, redundant], dim=1)
            causal_graph_features = self.cgfc(causal_graph_features)
            causal_graph_logits = self.causal_graph_classifier(causal_graph_features)
            final_logits = unique_logits + causal_graph_logits
        
        return final_logits, synergistic, redundant, bin_idx


