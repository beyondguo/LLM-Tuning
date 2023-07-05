# train_rm.sh
# https://chat.openai.com/share/dee24131-2b3f-4c8f-aa72-7fc51f9f5f56 nvidia-smi不工作的问题
# note: 下面设置，在4卡A800上，每张卡大约消耗40-50G
# note: 这个reward_modeling.py 中的保存模型的部分还得改改，先每个ckp都要把整个大模型保存一次，非常占空间
CUDA_VISIBLE_DEVICES=0,1,2,3 python reward_modeling.py \
    --model_name baichuan-inc/baichuan-7B \
    --lora_target_models W_pack \
    --num_train_epochs 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 12 \
    --train_subset 1000 \
    --eval_subset 1000 \
    --max_length 1000 
    # --resume_from_checkpoint True

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --model_name roberta-large \
#     --train_subset 5000 \
#     --eval_subset 1000

