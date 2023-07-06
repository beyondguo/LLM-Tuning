# train_rm.sh
# note: 下面设置，在4卡A800上，每张卡大约消耗40-50G
CUDA_VISIBLE_DEVICES=0,1,2,3 python reward_modeling.py \
    --model_name baichuan-inc/baichuan-7B \
    --lora_target_models W_pack \
    --num_train_epochs 2 \
    --eval_steps 200 \
    --save_steps 50 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 16 \
    --train_subset 5000 \
    --eval_subset 1000 \
    --max_length 1000
    # --resume_from_checkpoint True

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --model_name roberta-large \
#     --train_subset 5000 \
#     --eval_subset 1000

