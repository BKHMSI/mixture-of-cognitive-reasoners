python generate.py \
    --config config_micro_llama_3b.yml \
    --ablate none \
    --output_file data/output_all.txt

python generate.py \
    --config config_micro_llama_3b.yml \
    --ablate social,world \
    --output_file data/output_logic.txt

python generate.py \
    --config config_micro_llama_3b.yml \
    --ablate logic,world \
    --output_file data/output_social.txt

python generate.py \
    --config config_micro_llama_3b.yml \
    --ablate logic,social \
    --output_file data/output_world.txt