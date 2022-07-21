from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import MLP

class VICReg(nn.Module):
    """
    Code adapted from https://github.com/facebookresearch/barlowtwins
    """
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last=False, norm_embed=False, relu_embed=False):
        super().__init__()
        
        self.norm_embed=norm_embed
        self.relu_embed=relu_embed
        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        self.backbone_net.fc = nn.Identity()
        
        self.projector = MLP(self.repre_dim, projector_hidden, bias = False, norm_last=norm_last)
    
    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        if self.norm_embed:
            z1,z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

        if self.relu_embed:
            z1,z2 = F.relu(z1), F.relu(z2)

        return z1,z2

class TempVICReg(VICReg):
    # t1 and t2 are like a time embedding, the loss function tries to predict the difference.
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last=False):
        super(TempVICReg,self).__init__(backbone_net, projector_hidden, norm_last)

        self.tempPredictor = MLP(self.repre_dim, (self.repre_dim//8,1), bias = False)
    
    def forward(self, x1, x2):
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        #diff = b1-b2
        t1 = self.tempPredictor(b1)
        t2 = self.tempPredictor(b2)

        return z1,z2,t1,t2

class TempVICRegConc(VICReg):
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last=False, norm_embed=False):
        super(TempVICRegConc,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        self.tempPredictor = MLP(projector_hidden[-1]*2, (projector_hidden[-1]*2//8,1), bias = False)
    
    def forward(self, x1, x2):
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        z_cat = torch.cat((z1, z2),dim=1)

        pred = self.tempPredictor(z_cat)

        return z1, z2, pred