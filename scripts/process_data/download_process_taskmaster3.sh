# **The official splits have overlapped conversations, so we skip use our own splits **
# ** we use huggingface's datasets to download the data **

# folder="data/taskmaster3/raw"
# mkdir -p $folder

# repo_path="data/taskmaster-repo"

# if [ -d $repo_path ]; then
# echo "$repo_path exists. skip cloning taskmaster repo"
# else
# git clone https://github.com/google-research-datasets/Taskmaster.git $repo_path
# fi

# for split in train dev "test"; do
#     raw_path="${folder}/${split}.tsv"
#     if ! [ -f $raw_path ]; then
#         cat ${repo_path}/TM-3-2020/splits/${split}/${split}-*.tsv > $raw_path
#     fi
# done


# create base model datasets
base_folder="data/taskmaster3/base"
python scripts/data_processing/prepare_base_data.py --dataset taskmaster3 --save_path $base_folder
