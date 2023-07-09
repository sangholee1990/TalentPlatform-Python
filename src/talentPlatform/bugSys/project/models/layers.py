from typing import Optional

import torch
from torch import nn


class MaskedConv1d(nn.Module):
    """
    Conv1d with mask
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation=1,
                 bias=True
                 ):
        super(MaskedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.conv_kernel = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.conv_kernel)
        self.bias_weight = nn.Parameter(torch.zeros(out_channels).float()) if bias else None

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.reshape(x.shape[0], 1, -1)
            x = x * mask.float()

        batch_size, dk, seq_len = x.shape
        padding_rows = max(0, (seq_len-1) + (self.kernel_size-1)*self.dilation+1-seq_len)
        dilation_odd = (self.dilation % 2 != 0)
        if dilation_odd:
            if self.kernel_size % 2 == 0:
                device = x.device
                x = torch.cat([torch.zeros(batch_size, dk, 1).to(device), x], dim=2)

        output = torch.conv1d(
            input=x,
            weight=self.conv_kernel,
            bias=self.bias_weight,
            dilation=self.dilation,
            padding=padding_rows // 2
        )
        if mask is not None:
            output = output * mask.float()
        return output


class MaskedAvgPooling(nn.Module):
    """
    expected shape(inputs) = batch_size, ..., seq_len, hidden_size  \n
    expected shape(mask) = batch_size, ..., seq_len
    """
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, keepdim: bool = False):
        return torch.sum(x, dim=-2, keepdim=keepdim) / mask.float().unsqueeze(-1).sum(dim=-2, keepdim=keepdim)


class MaskedMaxPooling(nn.Module):
    """
    expected shape(inputs) = batch_size, ..., seq_len, hidden_size  \n
    expected shape(mask) = batch_size, ..., seq_len
    """
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, keepdim: bool = False):
        x = x - (1.0 - mask.unsqueeze(-1).float()) * 1e8
        return torch.max(x, dim=-2, keepdim=keepdim)[0]


class EmbeddingWithPosition(nn.Module):
    """
    Input sequence, input embedding of sequence with Positional Embedding added.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 max_norm: Optional[float] = None,
                 norm_type: Optional[float] = None,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: Optional[torch.Tensor] = None,
                 _positional_weight: Optional[torch.Tensor] = None,
                 seq_len: int = 1024
                 ):
        super(EmbeddingWithPosition, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=seq_len+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_positional_weight
        )

    def forward(self, inputs: torch.Tensor):
        positional_seq = torch.arange(
            start=1,
            end=inputs.size(-1)+1,
            device=inputs.device,
            requires_grad=False
        )

        for _ in range(len(inputs.shape)-1):
            positional_seq = positional_seq.unsqueeze(0)
        positional_seq = positional_seq.expand_as(inputs)
        positional_seq = positional_seq.masked_fill(inputs == self.padding_idx, self.padding_idx)

        emb = self.embedding(inputs)
        positional_emb = self.positional_embedding(positional_seq)
        return emb + positional_emb


class AdditiveAttentionEncoder(nn.Module):
    """
    Weighted sum sequence by additive attention weight

    Shape(inputs) = (batch_size, (...,) seq_len, input_size)
    Shape(outputs) = (batch_size, (...,) input_size)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int
                 ):
        super(AdditiveAttentionEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.projection = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.activation = nn.Tanh()
        self.query = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # shape(inputs) = batch_size, ..., seq_len, input_size
        x = self.activation(self.projection(inputs))
        logits = self.query(x)
        if mask is not None:  # mask
            logits = logits - (1.0 - mask.unsqueeze(-1).float()) * 1e10
        weights = torch.softmax(logits, dim=-2)
        pooler_output = torch.sum(inputs * weights, dim=-2, keepdim=False)
        if mask is not None:
            pooler_output = mask[..., 0].float().unsqueeze(-1) * pooler_output
        return pooler_output