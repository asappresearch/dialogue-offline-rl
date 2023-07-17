import os
import fire
import json
import logging
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool
from typing import List, Dict
from bert_score import BERTScorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import torch
from dialogue_offline_rl.utils.conversation import (
    convo_to_context_response_pairs_abcd,
    convo_to_context_response_pairs_multi_woz,
    convo_to_context_response_pairs_taskmaster3,
)
from dialogue_offline_rl.utils.constants import *

logger = logging.getLogger(__name__)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_responses(model, tokenizer, context, context_len, response_len, num_responses):
    # prepare input   
    prompt = context.replace("> ", ">").replace(" <", "<")
    prompt = prompt + f"{REP_START}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]
    
    # call model
    outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        num_beams=num_responses,
        num_return_sequences=num_responses,
        eos_token_id=tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # decode output
    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # preds = [p.replace(REP_END, "") for p in preds]
        
    return preds

def generate_offline_rl_dataset(
    context_response_pairs, 
    model,
    tokenizer,
    num_model_responses,
    context_len,
    response_len,
):
    all_pred_responses = []
    all_gold_responses = []
    for idx, row in tqdm(context_response_pairs.iterrows(), total=context_response_pairs.shape[0]):
        # List of responses
        pred_responses = generate_responses(model=model, tokenizer=tokenizer, context=row['context'],
                                    context_len=context_len, response_len=response_len, num_responses=num_model_responses)
        all_gold_responses.append(row['response'])
        all_pred_responses.append(pred_responses)
    
    print("loading bertscorer")
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    preds_flat = [p for pred_responses in all_pred_responses for p in pred_responses]
    gold_flat = [gold_response for gold_response in all_gold_responses for i in range(num_model_responses)]

    P, R, F1 = scorer.score(preds_flat, gold_flat) # F1: (n_examples * n_responses) x 1
    bert_f1_flat = F1.tolist()
    all_bert_f1 = [bert_f1_flat[i:i+num_model_responses] for i in range(0, len(bert_f1_flat), num_model_responses)]
        
    offline_rl_data = []
    for cidx, (_, row) in enumerate(context_response_pairs.iterrows()):
        row_dict = {}
        row_dict['context'] = row['context']
        row_dict['true_response'] = row['response']
        
        row_dict[f'response_0'] = row_dict['true_response']
        row_dict[f'bert_f1_0'] = 1.
        
        for ridx in range(0, num_model_responses):
            row_dict[f'response_{ridx+1}'] = all_pred_responses[cidx][ridx]
            row_dict[f'bert_f1_{ridx+1}'] = all_bert_f1[cidx][ridx]

        offline_rl_data.append(row_dict)
            
    df_offline_rl_data = pd.DataFrame(offline_rl_data)
    return df_offline_rl_data

def write_tf_top_dataset(offline_rl_data, filename, num_responses, tf_top_thresh=1.):
    texts = []
    for idx, row in offline_rl_data.iterrows():
        for ridx in range(0, num_responses):
            if row[f'reward_{ridx}'] < tf_top_thresh:
                continue
            context = row['context']
            response = row[f'response_{ridx}']
            text = f"{context}{REP_START}{response}{REP_END}"
            texts.append(text)
    
    random.shuffle(texts)
    with open(filename, "w") as f:
        for text in texts:
            print(text, file=f)
    print(f"Saved dataset to file {filename}")

def write_tf_dataset(offline_rl_data, filename):
    texts = []
    for idx, row in offline_rl_data.iterrows():
        context = row['context']
        response = row['response_0']
        text = f"{context}{REP_START}{response}{REP_END}"
        texts.append(text)
    
    random.shuffle(texts)
    with open(filename, "w") as f:
        for text in texts:
            print(text, file=f)
    print(f"Saved dataset to file {filename}")
    
def write_dt_dataset(offline_rl_data, filename, num_responses):
    texts = []
    for idx, row in offline_rl_data.iterrows():
        for ridx in range(0, num_responses):
            context = row['context']
            # For every REP_START in the context, prepend by REWARD_ONE
            context = context.replace(f"{REP_START}", f"{REWARD_ONE}{REP_START}")
            response = row[f'response_{ridx}']
            reward_token = REWARD_ONE if row[f'reward_{ridx}'] >= 1.0 else REWARD_ZERO
            text = f"{context}{reward_token}{REP_START}{response}{REP_END}"
            texts.append(text)

    # randomly shuffle text
    random.shuffle(texts)
    with open(filename, "w") as f:
        for text in texts:
            print(text, file=f)
    print(f"Saved dataset to file {filename}")

def write_ilql_dataset(offline_rl_data, filename):
    offline_rl_data.to_csv(filename)
    print(f"Saved dataset to file {filename}")

def create_dataset(
    methods: List[str] = ["tf_top", "tf_all", "dt", "ilql"],
    dataset: str = "abcd",
    data_path: str = "data/abcd/raw/abcd_v1.1.json",
    model_path: str = None,
    num_samples: int = None,
    num_model_responses: int = 5,
    context_len: int = 96,
    response_len: int = 32,
    bert_thresh: float = 0.6,
    split: str = "train",
    save_path: str = "",
    load_cached_data: bool = True,
):
    if save_path == "":
        save_path = f"data/{dataset}/rl"

    cache_path = f"{save_path}/offline_rl_data_{split}_bt{bert_thresh}_nmr{num_model_responses}_ns{num_samples}_c{context_len}_r{response_len}.csv"
    if not load_cached_data or not os.path.exists(cache_path):
            
        # Load model, tokenizer
        print(f"loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
        
        num_param = sum([p.numel() for p in model.parameters()])
        print(f"# of param: {num_param / 10**6:.2f} M")

        with Pool() as p:
            if dataset == "abcd":
                for path in [data_path, save_path, model_path]:    
                    if path is None: continue
                    if not os.path.isabs(path): path = f"{BASE_PATH}/{path}"

                with open(data_path, 'r') as f:
                    data = json.load(f)
                data_split = "dev" if (split == "val") else split
                context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_abcd, [row['original'] for row in data[data_split]]), total=len(data[data_split])))
            elif dataset == "multi_woz":
                data = load_dataset("multi_woz_v22")
                data_split = "validation" if (split == "val") else split
                context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_multi_woz, data[data_split]), total=len(data[data_split])))
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
                context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_taskmaster3, data[split]), total=len(data[split])))
            else:
                raise NotImplementedError(f"{dataset=}")
        
        context_response_pairs = pd.concat(context_response_pairs_list)
        
        if num_samples is not None:
            context_response_pairs = context_response_pairs.sample(n=num_samples, random_state=1)
        
        offline_rl_data = generate_offline_rl_dataset(context_response_pairs, model, tokenizer, num_model_responses, context_len, response_len)   
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        offline_rl_data.to_csv(cache_path) 
    else:
        offline_rl_data = pd.read_csv(cache_path)
    
    # Add rewards to data
    for ridx in range(0, num_model_responses+1):
            offline_rl_data[f'reward_{ridx}'] = (offline_rl_data[f'bert_f1_{ridx}'] > bert_thresh).astype('float')
    
    for method in methods:
        os.makedirs(f"{save_path}/{method}/", exist_ok=True)
        if method == 'tf_top':
            write_tf_top_dataset(offline_rl_data, filename=f"{save_path}/{method}/{split}.txt", num_responses=num_model_responses+1, tf_top_thresh=1.)
        elif method == 'tf_all':
            write_tf_top_dataset(offline_rl_data, filename=f"{save_path}/{method}/{split}.txt", num_responses=num_model_responses+1, tf_top_thresh=0.)
        elif method == 'dt':
            write_dt_dataset(offline_rl_data, filename=f"{save_path}/{method}/{split}.txt", num_responses=num_model_responses+1)
        elif method == 'ilql':
            write_ilql_dataset(offline_rl_data, filename=f"{save_path}/{method}/{split}.csv")
        elif method == 'tf': # alternate to base model, for debugging
            write_tf_dataset(offline_rl_data, filename=f"{save_path}/{method}/{split}.txt")
        else:
            raise NotImplementedError(f"{method=}")

if __name__ == "__main__":
    fire.Fire(create_dataset)