
for task in language logic social world;  
do
    for ckpt in 0 1 2 3 4;
    do
        python -m plots.plot_token_routing \
            -c config_micro_llama_ckpt${ckpt}.yml \
            --task gpt5-$task
        done
done

# for task in empathy gsm8k minerva_math mmlu;
# do
#     python -m plots.plot_token_routing \
#         -c config_micro_olmo.yml \
#         --task $task
# done

# python -m plots.plot_token_routing \
#     -c config_micro_smollm2_1p7b.yml \
#     --task sample

# python -m plots.plot_token_routing_test \
#     -c config_micro_llama_3b.yml