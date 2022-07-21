from typing import Union

import torch
import torch.nn as nn

from .norms import IterNorm

class MLP(nn.Module):
    # Projector/Expander network for contrastive/non-contrastive embeddings
    # Updates: bias term now controls only the last linear layer
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias = True, norm_last = False):
        super().__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        
        hidden_dims = (in_dim,) + hidden_dims

        mlp = []
        for i in range(len(hidden_dims) - 2):
            mlp.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        nn.BatchNorm1d(hidden_dims[i+1]),
                        nn.ReLU(inplace=True)])
        
        mlp.extend([nn.Linear(hidden_dims[-2], hidden_dims[-1], bias = bias)])

        if norm_last=='BN':
            mlp.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
        elif norm_last=='IN':
            mlp.append(IterNorm(hidden_dims[-1], num_groups=64, T=5, dim=2))
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)