import random

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

MAXLEN = 768


class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):

        title, text, keywords = [], [], []
        for k, v in data.items():
            title.append(v[0])
            text.append(v[1])
            keywords.append(v[2])

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.title = title
        self.text = text
        self.keywords = keywords

    #---------------------------------------------#

    @staticmethod
    def join_keywords(keywords, randomize=True):
        N = len(keywords)

        # random sampling and shuffle
        if randomize:
            M = random.choice(range(N + 1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)

    #---------------------------------------------#

    def __len__(self):
        return len(self.text)

    #---------------------------------------------#

    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)

        input = SPECIAL_TOKENS['bos_token'] + self.title[i] + \
            SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token'] + \
            self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=MAXLEN,
                                        padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}
