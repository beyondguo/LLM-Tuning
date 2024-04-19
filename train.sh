
# wandb login --relogin



# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=1 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_senti-desc_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_senti-desc_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_senti-desc_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_senti-desc_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_senti-desc_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_senti-desc_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_digital-label_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_digital-label_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_digital-label_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_digital-label_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_digital-label_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_digital-label_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_mentioned-only_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_mentioned-only_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_mentioned-only_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_mentioned-only_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_mentioned-only_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_mentioned-only_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_desc-senti_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_desc-senti_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_desc-senti_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_desc-senti_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_desc-senti_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_desc-senti_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_dict-output_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_dict-output_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_dict-output_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_dict-output_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_dict-output_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_dict-output_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_lines-output_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_lines-output_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_lines-output_d1_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_lines-output_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_lines-output_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_lines-output_d2_train-500_gby1002

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=3 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_instruction-first_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_instruction-first_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_instruction-first_d1_train-500_gby1030

# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=3 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_instruction-first_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_instruction-first_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_instruction-first_d2_train-500_gby1030


# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_baseline_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_baseline_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_baseline_d1_train-500_gby1002


# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_baseline_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_baseline_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_baseline_d2_train-500_gby1002


# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_text2label_d1_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_text2label_d1_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_text2label_d1_train-500_gby1002


# WANDB_PROJECT=DCAI-ABSA-Baichuan2-chat CUDA_VISIBLE_DEVICES=2 python baichuan2_lora_tuning.py \
#     --model_version chat-7b \
#     --tokenized_dataset absa_text2label_d2_train-Baichuan2-7B-Chat \
#     --train_size 500 \
#     --eval_size 1000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_text2label_d2_train-500-Baichuan2-7B-Chat \
#     --report_to wandb \
#     --run_name absa_text2label_d2_train-500_gby1002



