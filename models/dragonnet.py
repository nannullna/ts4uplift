import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import MLP

class Dragonnet(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, dropout_p: float = 0.2):
        super(Dragonnet, self).__init__()
        self.feature_dim = feature_dim

        self.backbone = backbone
        self.predictor_t = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.predictor_y1 = MLP(feature_dim+1, 1, 2, shrink=2, dropout_p=dropout_p)
        self.predictor_y0 = MLP(feature_dim+1, 1, 2, shrink=2, dropout_p=dropout_p)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
    
    def forward(self, inputs):
        z = self.backbone(inputs)
        t = self.predictor_t(z)
        y1 = F.sigmoid(self.predictor_y1(z))
        y0 = F.sigmoid(self.predictor_y0(z))

        return {'t': t, 'y1': y1, 'y0': y0}