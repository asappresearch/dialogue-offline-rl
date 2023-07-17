method=${1} # "tf_top", "tf_all", "dt"
data_dir=${2} #
model_path_tf=${3}
save_dir=${4}
ngpu=${5} # 1, 4

model="distilgpt2"

# model_path_tf="save/${model}-${dataset}/${dataset}-${model}-tf-lr5e-4-bs4-epoch10-ws0-gas1-4gpu"
# save_dir="save/2301/${model}-${dataset}/"
# data_prefix="data/${dataset}/rl/${model}/${method}/"

logging_steps=50

function train_model {
if [[ $ngpu -gt 1 ]]; then
    master_port=`shuf -i 20000-35000 -n 1`
    prefix="-m torch.distributed.launch --nproc_per_node=$ngpu --master_port $master_port"
else
    prefix=""
fi

if [ -d ${save_dir}/${run_name} ]; then
overwrite_output_dir=False
else
mkdir -p ${save_dir}/${run_name}
overwrite_output_dir=True
fi

echo $run_name

python $prefix scripts/training/run_clm_conv.py \
    --model_name_or_path $model_name \
    --tokenizer_name $tokenizer_name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --do_train \
    --do_eval \
    --bf16 True \
    --block_size 512 \
    --one_example_per_block True \
    --mask_context True \
    --run_name $run_name \
    --num_train_epochs $num_train_epochs \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps $logging_steps \
    --warmup_steps $warmup_steps \
    --remove_unused_columns False \
    --train_file ${data_prefix}train.txt \
    --validation_file ${data_prefix}val.txt \
    --report_to tensorboard \
    --lr_scheduler_type cosine \
    --overwrite_output_dir $overwrite_output_dir \
    --overwrite_cache False \
    --save_total_limit 2 \
    --load_best_model_at_end 1 \
    --output_dir ${save_dir}/${run_name} $@ \
    2>&1 | tee -a ${save_dir}/${run_name}/log.txt
    # --eval_steps $eval_steps \
}

case $ngpu in
1)
ngpu=1
tokenizer_name="gpt2"
model_name="${model_path_tf}"
model_type="${model}-${method}"
num_train_epochs=5
# eval_steps=1000
# save_steps=1000
warmup_steps=0
batch_size=16
lr=5e-4
gradient_accumulation_steps=1
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --bf16 True 

;;
4)
ngpu=4
tokenizer_name="gpt2"
model_name="${model_path_tf}"
model_type="${model}-${method}"
num_train_epochs=5
# eval_steps=1000
# save_steps=1000
warmup_steps=0
batch_size=8
lr=5e-4
gradient_accumulation_steps=1
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --bf16 True

esac