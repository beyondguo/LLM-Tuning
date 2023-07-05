# train_rm.sh
# https://chat.openai.com/share/dee24131-2b3f-4c8f-aa72-7fc51f9f5f56 nvidia-smi不工作的问题
CUDA_VISIBLE_DEVICES=1 python reward_modeling.py \
    --model_name baichuan-inc/baichuan-7B \
    --lora_target_models W_pack \
    --per_device_train_batch_size 4 \
    --train_subset -1 \
    --eval_subset -1 \
    --max_length 1000 

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --model_name roberta-large \
#     --train_subset 5000 \
#     --eval_subset 1000

