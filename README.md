# On the Effectiveness of Offline RL for Dialogue Response Generation

This repository contains code for the paper, [On the Effectiveness of Offline RL for Dialogue Response Generation](https://arxiv.org/abs/2307.12425), presented at ICML 2023.

## Installation
```bash
git clone git@github.com:asappresearch/dialogue-offline-rl.git
cd dialogue-offline-rl
pyenv virtualenv dialogue-offline-rl
pyenv activate dialogue-offline-rl
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Model Checkpoints

Model | Links
---|---
Base model (`tf`) | [ABCD](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/abcd-distilgpt2-tf-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [MultiWoz-2.2](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/multi_woz-distilgpt2-tf-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [TaskMaster-3](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/taskmaster3-distilgpt2-tf-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip)
Fine Tune on Top Returns (`tf_top`) | [ABCD](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/abcd-distilgpt2-tf_top-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [MultiWoz-2.2](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/multi_woz-distilgpt2-tf_top-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [TaskMaster-3](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/taskmaster3-distilgpt2-tf_top-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip)
Decision Transformers: Condition on Return (`dt`) | [ABCD](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/abcd-distilgpt2-dt-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [MultiWoz-2.2](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/multi_woz-distilgpt2-dt-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip), [TaskMaster-3](https://public-dataset-model-store.awsdev.asapp.com/psodhi/dialogue-offline-rl/public/models/taskmaster3-distilgpt2-dt-lr5e-4-bs4-epoch10-ws0-gas1-4gpu.zip)
Off-policy Q-learning (`ilql`) | [ABCD](), [MultiWoz-2.2](), [TaskMaster-3]()

## Data Processing

### 1. Create Base Datasets and TF Model

Download and create datasets for training the base `TF` model:

```sh
for dataset in abcd multi_woz taskmaster3; do
    bash scripts/process_data/download_process_${dataset}.sh
done
```

Train the base `TF` model by executing:
```sh
bash scripts/train/train_base_tf_distilgpt2.sh {dataset} {ngpu}
```
for example, `bash scripts/train/train_base_tf_model_distilgpt2.sh abcd 4`

### 2. Create Offline RL Datasets
To generate datasets for all three methods (`tf_top`, `dt`, `ilql`), we need the path to the base `TF` model (`model_path`):

```bash
for split in train val test; do
    python scripts/process_data/prepare_offline_rl_data.py --model_path_tf {model_path_tf} --save_path {save_path} --split ${split}
done
```

## Training Offline RL Models

For training, we provide scripts for each of the three methods (`tf_top`, `dt`, `ilql`):

### 1. Fine Tune on Top Returns, `tf_top`
```bash
bash scripts/train/train_offline_rl_distilgpt2.sh tf_top {dataset} {ngpu}
```
for example, `bash scripts/train/train_offline_rl_distilgpt2.sh tf_top abcd 4`

### 2. Decision Transformers: Condition on Return, `dt`
```bash
bash scripts/train/train_offline_rl_distilgpt2.sh dt {dataset} {ngpu}
```
for example, `bash scripts/train/train_offline_rl_distilgpt2.sh dt abcd 4`

### 3. Off-policy Q-learning, `ilql`
First, install trlx from a fork at [this location](https://github.asapp.dev/psodhi/trlx/tree/ps/dev/dialogue). Then execute the command:
```bash
python scripts/training/run_trlx_ilql.py --config_path config/trlx_ilql_gpt2med.yml --data_path {ilql_data_path}
```

## Evaluation

To evaluate all the models:
```bash
python scripts/evaluation/evaluate_reward_metrics.py --dataset {dataset} --method {method} --model_path {model_path} --metrics '["bert_score", "bleurt_score", "meteor", "bleu"]' --save_path {save_path} --num_samples 1000
```
where, `method={tf, tf_top, dt, ilql}`, `dataset={abcd, multi_woz, taskmaster3}`, and `model_path` is the path to corresponding model. The script will save all the predictions and metrics to a `.csv` at `save_path`.

## Citation

If you found our code or paper useful, please consider citing:

```bibtex
@inproceedings{sodhi2023offlinerl,
  title={On the Effectiveness of Offline RL for Dialogue Response Generation},
  author={Sodhi, Paloma and Wu, Felix and Elenberg, Ethan R and Weinberger, Kilian Q and McDonald, Ryan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year={2023}
}
```
## License

This project is licensed under the terms of the [MIT license](./LICENSE).
