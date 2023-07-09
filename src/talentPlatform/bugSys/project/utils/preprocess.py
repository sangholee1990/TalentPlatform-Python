import os
import abc
import re
from collections import OrderedDict
from typing import List, Tuple, Any, Union

import nltk 
import numpy as np
from transformers import AutoTokenizer

def pad_sequence(seq: List[Any], seq_len: int, padding: Any = 0):
    """
    Pad sequence to target length
    :param seq: input list
    :param seq_len: target length to pad to
    :param padding: special token or token id for padding
    :return: padded_seq, mask
    """
    mask = [1] * min(seq_len, len(seq)) + [0] * max(seq_len - len(seq), 0)
    padded_seq = seq[:seq_len] + [padding] * max(seq_len - len(seq), 0)
    return padded_seq, mask


def load_emb_matrix(emb_path: os.PathLike) -> np.ndarray:
    emb = []
    with open(emb_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            value = list(map(float, line.split()[1:]))
            # pad emb and unk emb is [0] * emb_dim
            if i == 0:
                for j in range(2):
                    emb.append([0 for _ in range(len(value))])
            emb.append(value)
    return np.array(emb)


class Tokenizer(object):
    regs = [
        r"Mr\. [A-Za-z ]+\. Madam [A-Za-z ]+,",
        r"Madam [A-Za-z ]+\. Madam [A-Za-z ]+,",
        r"Mr\. [A-Za-z ]+\. Mr\. [A-Za-z ]+,",
        r"Mr\. [A-Za-z ]+\. Mrs\. [A-Za-z ]+,",
        r"Madam [A-Za-z ]+\. Mr\. [A-Za-z ]+,",
        r"Mrs\. [A-Za-z ]+\. Mr\. [A-Za-z ]+,",
        r"Mrs\. [A-Za-z ]+\. Mrs\. [A-Za-z ]+,",
        r"Mrs\. [A-Za-z ]+\. Madam [A-Za-z ]+,",
        r"Madam [A-Za-z ]+\. Mrs\. [A-Za-z ]+,",
    ]
    @abc.abstractmethod
    def call(self, inputs: List[str]):
        pass

    def __call__(self, inputs: List[str], **kwargs) -> Tuple[List[List[int]], List[List[int]]]:
        outputs = []
        for content in inputs:
            for reg in self.regs:
                content = re.sub(reg, "", content)
            outputs.append(content)
        return self.call(outputs, **kwargs)


class GloveTokenizer(Tokenizer):
    """
    Tokenize sentence with nltk tokenizer and glove word2idx
    Use nltk tokenizer as basic tokenizer,
    Optionally add stopword filter, stemming, lemmatization
    """
    def __init__(self, pretrain_path: os.PathLike, seq_len: int = 50):
        super().__init__()
        self.tokenizer = nltk.tokenize.word_tokenize
        self.seq_len = seq_len
        self.word2idx = OrderedDict({"[PAD]": 0, "[UNK]": 1})
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.PorterStemmer()

        # load glove vocabulary
        with open(pretrain_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                self.word2idx[word] = len(self.word2idx)
        self.vocabulary_size = len(self.word2idx)

    def call(self, inputs: List[str], lower: bool = True, stem: bool = False, lemmatize: bool = False) -> Tuple[List[List[int]], List[List[int]]]:
        tokenized_sents = []
        masks = []
        for sentence in inputs:
            tokenized_sent = self.tokenizer(sentence)
            ids = [self.word2idx.get(self.word_process(word, lower, stem, lemmatize), self.word2idx["[UNK]"]) for word in tokenized_sent]
            ids, mask = pad_sequence(ids, seq_len=self.seq_len, padding=self.word2idx["[PAD]"])
            tokenized_sents.append(ids)
            masks.append(mask)
        return tokenized_sents, masks
    
    def word_process(self, word: str, lower: bool = True, stem: bool = False, lemmatize: bool = False) -> str:
        if lower:
            word = word.lower()
        if lemmatize:
            word = self.lemmatizer.lemmatize(word)
        if stem:
            word = self.stemmer.stem(word)
        return word



class WordPieceTokenizer(Tokenizer):
    """
    Bert's word-piece tokenizer
    """
    def __init__(self, pretrain_dir: Union[os.PathLike, str], seq_len: int = 50):
        super(WordPieceTokenizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
        self.seq_len = seq_len
        self.vocabulary_size = len(self.tokenizer.get_vocab())

    def call(self, inputs: List[str], **kwargs) -> Tuple[List[List[int]], List[List[int]]]:
        batch_encoded = self.tokenizer(inputs, max_length=self.seq_len, padding="max_length", truncation=True)
        input_ids, att_mask = batch_encoded["input_ids"], batch_encoded["attention_mask"]

        return input_ids, att_mask
