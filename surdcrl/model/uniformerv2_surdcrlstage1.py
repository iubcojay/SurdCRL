    # Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
from .uniformerv2_common import (
    QuickGELU, Local_MHRA, ResidualAttentionBlock, 
    Extractor, Transformer, UniFormerV2context, UniFormerV2obj
)
from timm.models.layers import trunc_normal_
from .utils import ObjectsCrops
import numpy as np
# from mmaction.registry import MODELS

logger = MMLogger.get_current_instance()



class orcausal(nn.Module):
    """Stage-1 causal model wrapper.

    Combines object and context branches, aggregates unique tokens and outputs
    class logits. This wrapper preserves original behavior and interface.
    """

    def __init__(self, num_classes: int = 20, n_dim: int = 768, obj: int = 4) -> None:
        super().__init__()
        self.moduleobj = UniFormerV2obj()
        self.modulecontext = UniFormerV2context()
        self.uniquefc = nn.Sequential(
            nn.Linear(n_dim*(obj+1), n_dim),
            nn.ELU()
        )
        self.classifer = nn.Linear(768,num_classes)
    def get_updateModel(self, path): 
        # pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_dict = torch.load(path, map_location ='cpu') # 自己训练的模型
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

        # 下面两行，test和训练注释掉

        print("re-initialize fc layer")
        print("ckpt key lens{}".format(len(shared_dict.keys())))
        self.load_state_dict(model_dict, strict=False)
        return self 
    def forward(self,x,metadata: torch.Tensor,back):
        obj,_ = self.moduleobj(x,metadata)
        context,_ = self.modulecontext(back)
        context = context.unsqueeze(1)
        BS = obj.shape[0]
        unique = torch.cat([obj,context],dim=1).view(BS,-1)# 1,5,768
        unique = self.uniquefc(unique)
        out = self.classifer(unique)
        return out
    
