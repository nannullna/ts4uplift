import torch.nn as nn
import torch.nn.functional as F

from .dilated_conv import DilatedConvEncoder

class Encoder(nn.Module):

    POOLINGS = ['last', 'max', 'mean', 'none']

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder: nn.Module = None

    def create_pooler(self, pool: str):
        assert pool in self.POOLINGS, f'pool must be one of {self.POOLINGS}'
        if pool == 'last':
            pooler = lambda x: x[:, -1, :] # (B, L, D) -> (B, D)
        elif pool == 'max':
            pooler = lambda x: x.max(dim=1)[0] # (B, L, D) -> (B, D)
        elif pool == 'mean':
            pooler = lambda x: x.mean(dim=1) # (B, L, D) -> (B, D)
        elif pool == 'none':
            pooler = lambda x: x # (B, L, D)
        return pooler

    def forward(self, inputs):
        raise NotImplementedError

class TCNEncoder(Encoder):

    def __init__(self, input_dim: int, feature_dim: int, num_layers: int, dropout_p: float = 0.2, pool: str = 'last'):
        super(TCNEncoder, self).__init__()
        self.encoder = DilatedConvEncoder(input_dim, [feature_dim] * num_layers, kernel_size=3, dropout_p=dropout_p)
        self.pool = self.create_pooler(pool)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1) # RNN inputs are (B, L, D), but CNN inputs are (B, D, L).
        z = self.encoder(inputs).permute(0, 2, 1)
        z = self.pool(z)
        return z


class RNNEncoder(Encoder):
    
    def __init__(self, input_dim: int, feature_dim: int, num_layers: int, dropout_p: float = 0.2, pool: str = 'last', rnn_type: str = 'lstm'):
        super(RNNEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, feature_dim, num_layers, batch_first=True, dropout=dropout_p) if rnn_type == 'lstm' else \
                    nn.GRU(input_dim, feature_dim, num_layers, batch_first=True, dropout=dropout_p)
        self.pool = self.create_pooler(pool)

    def forward(self, inputs):
        z, _ = self.encoder(inputs)
        return self.pool(z)