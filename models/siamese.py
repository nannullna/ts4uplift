import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import MLP


class SiameseNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, dropout_p: float = 0.2):
        super(SiameseNetwork, self).__init__()
        self.feature_dim = feature_dim

        self.backbone = backbone
        self.predictor_y = MLP(feature_dim+1, 1, 2, shrink=2, dropout_p=dropout_p)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
    
    def forward(self, inputs):
        z = self.backbone(inputs)

        B, D = z.size()
        t1 = torch.ones(B, 1, device=z.device)
        t0 = torch.zeros(B, 1, device=z.device)
        z1 = torch.cat([z, t1], dim=1)
        z0 = torch.cat([z, t0], dim=1)

        y1 = self.predictor_y(z1).squeeze(1)
        y0 = self.predictor_y(z0).squeeze(0)

        return {'y1': y1, 'y0': y0}