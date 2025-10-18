config_file=$1
num_gpus=$2
accelerate_config_file=$3
sft_or_dpo=${4:-sft}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes $num_gpus \
    --config_file configs/${accelerate_config_file}  \
    train_${sft_or_dpo}.py -c $config_file \
    --wandb
