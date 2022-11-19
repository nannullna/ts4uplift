from typing import List, Union

import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: Union[int, List[int]], embedding_dim: Union[int, List[int]], max_length: int=1024, dropout_p: float = 0.2, use_layer_norm: bool=True, layer_norm_eps: float = 1e-5, positional_embedding: bool=True, flipped_embedding: bool=False):
        """
        Arguments:
            num_embeddings: int

            embedding_dim: int

            max_length: int (default: 10000)

            dropout_p: float (default: 0.1)
        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings if isinstance(num_embeddings, list) else [num_embeddings]
        self.embedding_dim = embedding_dim if isinstance(embedding_dim, list) else [embedding_dim]

        assert len(self.num_embeddings) == len(self.embedding_dim), 'num_embeddings and embedding_dim must have the same length'

        self.total_dim = sum(self.embedding_dim)
        self.max_length = max_length

        self.input_embeddings = nn.ModuleList(
            nn.Embedding(emb, dim, 0)
            for emb, dim in zip(self.num_embeddings, self.embedding_dim)
        )
        if positional_embedding:
            self.position_embeddings = nn.Embedding.from_pretrained(
                self.create_sinosoidal_embeddings(max_length, self.total_dim, flipped=flipped_embedding), 
                freeze=True,
            )
        else:
            self.position_embeddings = None

        self.layer_norm = nn.LayerNorm(self.total_dim, eps=layer_norm_eps) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.position_ids = torch.arange(max_length).expand((1, -1))


    def create_sinosoidal_embeddings(self, max_length: int, embedding_dim: int, flipped: bool=False):
        emb = torch.empty(max_length, embedding_dim)
        pos_ids = torch.arange(0, max_length).unsqueeze(1)
        div = torch.exp((torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.) / embedding_dim)))
        if flipped:
            emb[:, 0::2] = torch.flip(torch.sin(pos_ids.float() * div), [1])
            emb[:, 1::2] = torch.flip(torch.cos(pos_ids.float() * div), [1])
        else:
            emb[:, 0::2] = torch.sin(pos_ids.float() * div)
            emb[:, 1::2] = torch.cos(pos_ids.float() * div)
        return emb


    def forward(self, inputs: torch.LongTensor):
        """
        Arguments:
            inputs: torch.LongTensor of shape (batch_size, sequence_length)

        Returns:
            torch.Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(2) # (B, L, 1)
        B, L, D = inputs.size() # B: batch size, L: sequence length, D: input dimension
        
        if self.position_embeddings is not None:
            position_ids = self.position_ids[:, 0:L].to(inputs.device)
            position_embeddings = self.position_embeddings(position_ids)

        embeddings = []
        for i, input in enumerate(inputs.split(1, dim=2)):
            embeddings.append(self.input_embeddings[i](input.squeeze(2))) # (B, L) -> (B, L, D_i)
        embeddings = torch.cat(embeddings, dim=2) # (B, L, D)

        if self.position_embeddings is not None:
            embeddings += position_embeddings
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        return embeddings