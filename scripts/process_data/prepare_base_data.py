import os
import fire
import json
import logging

import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from typing import List, Dict
from datasets import load_dataset

from dialogue_offline_rl.utils.conversation import (
    convo_to_text_abcd, convo_to_text_multi_woz, convo_to_text_taskmaster3
)

logger = logging.getLogger(__name__)
BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))

def create_dataset(
    dataset: str = "abcd",
    data_path: str = "data/abcd/raw/abcd_v1.1.json",
    save_path: str = "data/abcd/base",
    splits: List[str] = ["val", "test", "train"],
):
    os.makedirs(save_path, exist_ok=True)
    if dataset == "abcd":
        for path in [data_path, save_path]:
            if path is None: continue
            if not os.path.isabs(data_path): path = f"{BASE_PATH}/{path}"

        with open(data_path, "r") as f:
            data = json.load(f)
    elif dataset == "multi_woz":
        data = load_dataset("multi_woz_v22")
    elif dataset == "taskmaster3":
        data = load_dataset("taskmaster3")['train'] # all data are in the 'train' split
        # split by 80%, 10%, 10%
        indices = np.random.RandomState(42).permutation(len(data)).tolist()
        indices = {
            'train': indices[:int(len(data) * 0.8)],
            'val': indices[int(len(data) * 0.8):int(len(data) * 0.9)],
            'test': indices[int(len(data) * 0.9):],
        }
        data = {k: [data[i] for i in indices[k]] for k in indices}
    else:
        raise NotImplementedError(f"{dataset=}")

    for split in splits:
        file = f"{save_path}/{split}.txt"
        with open(file, "w") as f:
            with Pool() as p:
                if dataset == "abcd":
                    split = "dev" if (split == "val") else split
                    texts = list(tqdm(p.imap(convo_to_text_abcd, [row['original'] for row in data[split]]), total=len(data[split])))
                elif dataset == "multi_woz":
                    split = "validation" if (split == "val") else split
                    texts = list(tqdm(p.imap(convo_to_text_multi_woz, data[split]), total=len(data[split])))
                elif dataset == "taskmaster3":
                    texts = list(tqdm(p.imap(convo_to_text_taskmaster3, data[split]), total=len(data[split])))
            for text in texts:
                print(text, file=f)

            print(f"Saved dataset to file {file}")

if __name__ == "__main__":
    fire.Fire(create_dataset)