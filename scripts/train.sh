config_file=$1
num_gpus=$2
accelerate_config_file=$3

accelerate launch --num_processes $num_gpus \
    --config_file configs/${accelerate_config_file}  \
    train.py -c $config_file \
    --wandb
