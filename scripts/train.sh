config_file=$1
num_gpus=$2
accelerate_config_file=$3
sft_or_dpo=${4:-sft}

accelerate launch --num_processes $num_gpus \
    --config_file configs/${accelerate_config_file}  \
    train_${sft_or_dpo}.py -c $config_file \
    --wandb
