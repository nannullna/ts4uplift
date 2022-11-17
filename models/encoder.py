from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dilated_conv import DilatedConvEncoder
from .embedding import Embedding

class Encoder(nn.Module):

    POOLINGS = ['last', 'max', 'mean', 'none']

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding: nn.Module = None
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

    def __init__(
        self, 
        num_embeddings: Union[int, List[int]], 
        embedding_dim: Union[int, List[int]], 
        feature_dim: int, 
        num_layers: int, 
        max_length: int=1024, 
        dropout_p: float = 0.2, 
        pool: str = 'last'
    ):
        super(TCNEncoder, self).__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, max_length, dropout_p)
        input_dim = self.embedding.total_dim
        self.encoder = DilatedConvEncoder(input_dim, [feature_dim] * num_layers, kernel_size=3, dropout_p=dropout_p)
        self.pool = self.create_pooler(pool)

    def forward(self, inputs):
        timestamp, X = inputs['timestamp'], inputs['X']
        X = self.embedding(X)
        ftrs = torch.cat([timestamp, X], dim=-1)
        ftrs = ftrs.permute(0, 2, 1) # RNN inputs are (B, L, D), but CNN inputs are (B, D, L).
        z = self.encoder(ftrs).permute(0, 2, 1)
        return self.pool(z)


class RNNEncoder(Encoder):
    
    def __init__(
        self, 
        num_embeddings: Union[int, List[int]], 
        embedding_dim: Union[int, List[int]], 
        feature_dim: int, 
        num_layers: int, 
        max_length: int=1024, 
        dropout_p: float = 0.2, 
        pool: str = 'last',
        rnn_type: str = 'lstm'
    ):
        super(RNNEncoder, self).__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, max_length, dropout_p)
        input_dim = self.embedding.total_dim
        self.encoder = nn.LSTM(input_dim, feature_dim, num_layers, batch_first=True, dropout=dropout_p) if rnn_type == 'lstm' else \
                    nn.GRU(input_dim, feature_dim, num_layers, batch_first=True, dropout=dropout_p)
        self.pool = self.create_pooler(pool)

    def forward(self, inputs):
        timestamp, X = inputs['timestamp'], inputs['X']
        X = self.embedding(X)
        ftrs = torch.cat([timestamp, X], dim=-1)
        z, _ = self.encoder(ftrs)
        return self.pool(z)