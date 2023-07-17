dataset=${1} # "abcd", "multi_woz", "taskmaster3"
save_path=${2}

model="gpt2-medium"
mkdir -p $save_path

# tf
python scripts/evaluation/evaluate_reward_metrics.py --num_responses 5 --method tf --model_path save/2301/${model}-${dataset}/${dataset}-${model}-tf-lr1e-4-bs4-epoch10-ws0-gas1-4gpu --metrics '["bert_score", "bleurt_score", "meteor", "bleu"]' --save_path $save_path --num_samples 1000 --dataset $dataset
# tf_top
python scripts/evaluation/evaluate_reward_metrics.py --num_responses 5 --method tf_top --model_path save/2301/${model}-${dataset}/${dataset}-${model}-tf_top-lr1e-4-bs8-epoch5-ws0-gas1-4gpu --metrics '["bert_score", "bleurt_score", "meteor", "bleu"]' --save_path $save_path --num_samples 1000 --dataset $dataset
# tf_all
python scripts/evaluation/evaluate_reward_metrics.py --num_responses 5 --method tf_all --model_path save/2301/${model}-${dataset}/${dataset}-${model}-tf_all-lr1e-4-bs8-epoch5-ws0-gas1-4gpu --metrics '["bert_score", "bleurt_score", "meteor", "bleu"]' --save_path $save_path --num_samples 1000 --dataset $dataset
# dt
python scripts/evaluation/evaluate_reward_metrics.py --num_responses 5 --method dt --model_path save/2301/${model}-${dataset}/${dataset}-${model}-dt-lr1e-4-bs8-epoch5-ws0-gas1-4gpu --metrics '["bert_score", "bleurt_score", "meteor", "bleu"]' --save_path $save_path --num_samples 1000 --dataset $dataset