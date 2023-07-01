# CUDA_VISIBLE_DEVICES=1,2,3 python chatglm2_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_chatglm2 \
#     --lora_rank 4 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/sentiment_comp_ie_chatglm2

# CUDA_VISIBLE_DEVICES=3 python chatglm_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_shuffled \
#     --lora_rank 4 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/temp


# CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
#     --tokenized_dataset rulai_plus_baichuan-7B \
#     --lora_rank 4 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 3 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/rulai_plus_baichuan-7B


# continue training with LoRA
# 使用 rulai_baichuan-7B 数据，在 weights/rulai_plus_baichuan-7B 的基础上继续训练 新的结果保存在 weights/rulai_plus_enhanced_baichuan-7B
CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
    --tokenized_dataset rulai_enhance_baichuan-7B \
    --previous_lora_weights weights/rulai_plus_baichuan-7B \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/rulai_plus_enhanced_baichuan-7B
