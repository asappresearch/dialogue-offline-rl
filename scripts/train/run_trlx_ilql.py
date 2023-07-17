import os
import fire
import pandas as pd
import torch

import trlx
from trlx.data.configs import TRLConfig
from trlx.utils.evaluate import evaluate_V_ranker
from dialogue_offline_rl.utils.constants import *

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_train_dataset(df, num_responses, split_token):
    df_list = []
    for ridx in range(0, num_responses):
        df_res = pd.DataFrame()
        df_res['context_response'] = df.apply(lambda row: f"{row['context']}{REP_START}{split_token}{row[f'response_{ridx}']}{REP_END}", axis=1)
        df_res['reward'] = df[f'reward_{ridx}']
        df_list.append(df_res)
    
    df_flatten = pd.concat(df_list)
    dataset = tuple(df_flatten['context_response'].tolist())
    rewards = tuple(df_flatten['reward'].tolist())
    
    return dataset, rewards
    
def run_ilql(
    config_path: str = "",
    data_path: str = "",
    split_token: str = "<|endoftext|>",
    num_responses: int = 6
    ):
    for path in [data_path, config_path]:
        if not os.path.isabs(path):
            path = f"{BASE_PATH}/{path}"
                   
    # Load train data
    df_train = pd.read_csv(f"{data_path}/train.csv")
    dataset_train, rewards = create_train_dataset(df=df_train, num_responses=num_responses, split_token=split_token)
        
    # Load val data
    df_val = pd.read_csv(f"{data_path}/val.csv")
    eval_fn = lambda model, tokenizer: evaluate_V_ranker(model, tokenizer, df_val,num_responses=num_responses, split_token=split_token)
        
    model = trlx.train(
        model_path=None,
        dataset=[dataset_train, rewards],
        config=TRLConfig.load_yaml(config_path),
        split_token=split_token,
        eval_prompts=[df_val['context'][0]],
        eval_fn=eval_fn
    )
    
if __name__ == "__main__":
    fire.Fire(run_ilql)