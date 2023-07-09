from typing import Optional, Union

import torch 
from torch import nn
import numpy as np

from .layers import EmbeddingWithPosition, AdditiveAttentionEncoder


class AttentionLSTM(nn.Module):
    """
    From paper: Attention-Based Bidirectional Long Short-Term Memory Networks forRelation Classification
    """
    def __init__(
        self, 
        vocab_size: int,
        emb_dim: int,
        hidden_size: int, 
        output_size: int,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        emb_matrix: Optional[np.ndarray] = None,
        **kwargs
    ):
        super(AttentionLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = EmbeddingWithPosition(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=0,
            _weight=emb_matrix
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.att = AdditiveAttentionEncoder(
            input_size=hidden_size*2,
            hidden_size=hidden_size
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size*2, out_features=hidden_size*2),
            nn.BatchNorm1d(num_features=hidden_size*2),
            nn.Linear(in_features=hidden_size*2, out_features=output_size)
        )
    
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(inputs)
        h, _ = self.lstm(x)
        h = self.att(h, mask=mask if mask is not None else None)
        logits = self.fc(h)
        return logits
