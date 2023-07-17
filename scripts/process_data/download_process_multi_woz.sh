# data will be downloaded using huggingface's datasets

# create base model datasets
base_folder="data/multi_woz/base"
python scripts/data_processing/prepare_base_data.py --dataset multi_woz --save_path $base_folder
