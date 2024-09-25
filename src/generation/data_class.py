from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import *


class DataProcess:
    def __init__(self, tokenizer_name: str, block_size: int) -> object:
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def add_special_tokens_to_tokenizer(self, special_tokens_dict):
        self.tokenizer.add_special_tokens(special_tokens_dict)
        #self.tokenizer.bos_token = "<|startoftext|>"
        #self.tokenizer.eos_token = "<|endoftext|>"

    def get_len_dataset(self, tokenized_datasets):
        # nb_tokens = 0
        longueurs = []
        for el in tokenized_datasets:
            for ids in tokenized_datasets[el]["input_ids"]:
                # nb_tokens += len(ids)
                longueurs.append(len(ids))
        print(f"Mean length : {np.mean(longueurs)} tokens")        
        print(f"Standard Deviation : {np.std(longueurs)}")
        print(f"Maximum : {np.max(longueurs)} | Minimum : {np.min(longueurs)}")
        return np.sum(longueurs)

    # data: list of text
    def prepare_data(self, data: [], batch_size, special_tokens=None):
        train, test = train_test_split(data, test_size=0.20)
        datasets = DatasetDict()
        df_train = pd.DataFrame(train, columns=["text"])
        df_test = pd.DataFrame(test, columns=["text"])

        datasets["train"] = Dataset.from_pandas(df_train)
        datasets["validation"] = Dataset.from_pandas(df_test)

        if special_tokens is not None:
            self.add_special_tokens_to_tokenizer(special_tokens)

        tokenized_datasets = datasets.map(self.tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, batch_size=1000, num_proc=4)

        train_loader = DataLoader(lm_datasets["train"], batch_size=batch_size, num_workers=1, collate_fn=data_collator)
        val_loader = DataLoader(lm_datasets["validation"], batch_size=batch_size, num_workers=1, collate_fn=data_collator)

        return train_loader, val_loader, len(self.tokenizer)
