from typing import Optional, Union

import torch 
from torch import nn
import numpy as np
from torch.nn.modules.container import ModuleList 

from .layers import EmbeddingWithPosition, MaskedConv1d, MaskedAvgPooling, MaskedMaxPooling


class TextCNN(nn.Module):
    """
    From paper: Convolutional Neural Networks for Sentence Classification
    """
    def __init__(
        self, 
        vocab_size: int,
        emb_dim: int,
        kernel_size: int,
        out_channels: int,
        output_size: int,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        emb_matrix: Optional[np.ndarray] = None,
        **kwargs
    ):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.output_size = output_size
        self.num_layers = max(num_layers, 1)
        self.dropout_rate = dropout_rate

        self.embedding = EmbeddingWithPosition(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=0,
            _weight=emb_matrix
        )
        self.cnn = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cnn.append(
                    nn.Sequential(MaskedConv1d(emb_dim, out_channels, kernel_size), nn.ReLU())
                )
            else:
                self.cnn.append(
                    nn.Sequential(MaskedConv1d(out_channels, out_channels, kernel_size), nn.ReLU())
                )
        self.max_pooler = MaskedMaxPooling()
        self.avg_pooler = MaskedAvgPooling()
        self.fc = nn.Sequential(
            nn.Linear(in_features=out_channels*2, out_features=out_channels),
            nn.BatchNorm1d(num_features=out_channels),
            nn.Linear(in_features=out_channels, out_features=output_size)
        )
    
    def forward(self, inputs, mask=None):
        x = self.embedding(inputs)
        x = x.permute(0, 2, 1)
        for layer in self.cnn:
            x = layer(x)
        x = x.permute(0, 2, 1)
        max_pooler_output = self.max_pooler(x, mask)
        avg_pooler_output = self.avg_pooler(x, mask)
        pooler_output = torch.cat([max_pooler_output, avg_pooler_output], dim=-1)
        logits = self.fc(pooler_output)
        return logits
