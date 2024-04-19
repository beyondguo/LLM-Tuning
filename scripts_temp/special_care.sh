wandb login --relogin f2881fdc7688653cc3e2986dad8cbcf2b8ec27af

# CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
#     --model_checkpoint ../DCAI-share/llm/chinese-llama-2-7b \
#     --input_file absa_baseline_d1_train.json \
#     --prompt_key prompt \
#     --target_key output \
#     --save_name absa_baseline_d1_train-llama2 \
#     --max_seq_length 2000 \
#     --skip_overlength False


# WANDB_PROJECT=DCAI-ABSA-Chinese-llama2 CUDA_VISIBLE_DEVICES=2 python chinese_llama2_alpaca2_lora_tuning.py \
#     --model_version base \
#     --tokenized_dataset absa_baseline_d1_train-llama2 \
#     --train_size 1000 \
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
#     --learning_rate 2e-5 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/absa_baseline_d1_train-1000-llama2 \
#     --report_to wandb \
#     --run_name absa_baseline_d1_train-1000_gby1203


# CUDA_VISIBLE_DEVICES=2 python predict_llama.py \
#     --base_model ../DCAI-share/llm/chinese-llama-2-7b \
#     --lora_model weights/absa_baseline_d1_train-1000-llama2 \
#     --data_path ../DCAI-share/processed_data/absa_baseline_d1_test.json \
#     --output_path data/eval_llama2/absa_baseline_d1_1000_llama2_d1_predict.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 128