import os 
from typing import Union

import torch 
import pandas as pd 
from torch.utils.data import Dataset

from .preprocess import GloveTokenizer, WordPieceTokenizer


class PoliticianDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pretrain_path: Union[os.PathLike, str],
        seq_len: int,
        tokenizer: str = "glove",
        lower: bool = True, 
        stem: bool = False, 
        lemmatize: bool = False
    ):
        super(PoliticianDataset, self).__init__()
        self.df = df
        self.seq_len = seq_len
        if "glove" in tokenizer:
            self.tokenizer = GloveTokenizer(pretrain_path, seq_len)
        else:
            self.tokenizer = WordPieceTokenizer(pretrain_path, seq_len)
        self.lower = lower
        self.stem = stem
        self.lemmatize = lemmatize

    def __getitem__(self, i):
        line = self.df.iloc[i]
        label = int(line.result)
        sentence = line.V2
        input_ids, mask = self.tokenizer([sentence], lower=self.lower, stem=self.stem, lemmatize=self.lemmatize)

        outputs = {
            "input_ids": torch.LongTensor(input_ids).squeeze(0),
            "mask": torch.LongTensor(mask).squeeze(0),
            "label": torch.FloatTensor([float(label)])
        }
        
        return outputs

    def __len__(self):
        return self.df.shape[0]