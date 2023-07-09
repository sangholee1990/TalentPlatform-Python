import os
from typing import Optional, Union

import torch 
from torch import nn

from transformers import BertModel
from transformers import BertForSequenceClassification


class BertForClassification(nn.Module):
    """
    From paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    and paper: How to Fine-Tune BERT for Text Classification?
    """
    def __init__(
        self, 
        pretrain_dir: Union[os.PathLike, str],
        output_size: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(BertForClassification, self).__init__()
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.bert = BertModel.from_pretrained(pretrain_dir)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=768, out_features=output_size),
        )
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        mask: Optional[torch.Tensor]=None, 
        token_type_ids: Optional[torch.Tensor] = None
    ):
        last_hidden_state = self.bert(inputs, attention_mask=mask, token_type_ids=token_type_ids).last_hidden_state
        pooler_output = torch.mean(last_hidden_state, dim=-2)
        logits = self.fc(pooler_output)
        return logits
