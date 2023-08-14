# WANDB_PROJECT=DCAI-ABSA-InternLM CUDA_VISIBLE_DEVICES=0,1 python internlm_lora_tuning.py \
#     --tokenized_dataset aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 100 \
#     --output_dir weights/aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --report_to wandb \
#     --run_name original-0809-gby


WANDB_PROJECT=DCAI-ABSA-InternLM CUDA_VISIBLE_DEVICES=0 python internlm_lora_tuning.py \
    --tokenized_train_dataset aspect_sentiment_train_orig_10k-internlm-chat-7b \
    --tokenized_eval_dataset aspect_sentiment_test_orig-internlm-chat-7b \
    --eval_size 2000 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 100 \
    --output_dir weights/aspect_sentiment_train_orig_10k-internlm-chat-7b \
    --report_to wandb \
    --run_name original-eval2k-0811-gby


# CUDA_VISIBLE_DEVICES=0,1 python chatglm2_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_chatglm2 \
#     --lora_rank 4 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 10 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/temp \
#     --report_to none
#     # --output_dir weights/sentiment_comp_ie_chatglm2


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
#     --tokenized_dataset sentiment_comp_ie_shuffled_baichuan-7B \
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
#     --output_dir weights/sentiment_comp_ie_shuffled_baichuan-7B


# continue training with LoRA
# 使用 rulai_baichuan-7B 数据，在 weights/rulai_plus_baichuan-7B 的基础上继续训练 新的结果保存在 weights/rulai_plus_enhanced_baichuan-7B
# CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
#     --tokenized_dataset rulai_enhance_baichuan-7B \
#     --previous_lora_weights weights/rulai_plus_baichuan-7B \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-5 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/rulai_plus_enhanced_baichuan-7B
