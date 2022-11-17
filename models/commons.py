from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_hiddens: int, shrink: Union[int, float]=1, dropout_p: float = 0.2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens
        self.shrink = shrink
        self.dropout_p = dropout_p

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for i in range(num_hiddens):
            self.layers.append(nn.Linear(prev_dim, int(prev_dim/shrink)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))
            prev_dim = int(prev_dim/shrink)
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
