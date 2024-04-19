


# # train
# N=1500
# M=1000

# wandb login --relogin f2881fdc7688653cc3e2986dad8cbcf2b8ec27af
# WANDB_PROJECT=DCAI-NER-llama2 CUDA_VISIBLE_DEVICES=2 python qwen1.5_lora_tuning.py \
#     --model_version chat \
#     --tokenized_dataset NER_default_design_${N}-Qwen1.5-4B-Chat \
#     --train_size $M \
#     --eval_size 500 \
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
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/NER_default_design_${N}-Qwen1.5-4B-Chat \
#     --report_to wandb \
#     --run_name NER_default_design_${N}-Qwen1.5-4B-Chat


# # predict

# N=1500
# method=default
# CUDA_VISIBLE_DEVICES=3 python predict_qwen.py \
#     --base_model Qwen/Qwen1.5-4B-Chat \
#     --lora_model weights/NER_default_design_${N}-Qwen1.5-4B-Chat \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_${N}_qwen1.5_prediction.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200



N=1000
M=500
desc_num=2

wandb login --relogin f2881fdc7688653cc3e2986dad8cbcf2b8ec27af
WANDB_PROJECT=DCAI-NER-llama2 CUDA_VISIBLE_DEVICES=2 python llama2_lora_tuning.py \
    --model_version chat \
    --tokenized_dataset NER_default_desc${desc_num}_design_${N}-llama2_chat \
    --train_size $M \
    --eval_size 500 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --no_prompt_loss 1 \
    --remove_unused_columns false \
    --logging_steps 25 \
    --output_dir weights/NER_default_desc${desc_num}_design_${N}-llama2 \
    --report_to wandb \
    --run_name NER_default_desc${desc_num}_design_${N}



wandb login --relogin f2881fdc7688653cc3e2986dad8cbcf2b8ec27af
WANDB_PROJECT=DCAI-NER-llama2 CUDA_VISIBLE_DEVICES=2 python llama2_lora_tuning.py \
    --model_version chat \
    --tokenized_dataset NER_weak1_desc${desc_num}_design_${N}-llama2_chat \
    --train_size $M \
    --eval_size 500 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --no_prompt_loss 1 \
    --remove_unused_columns false \
    --logging_steps 25 \
    --output_dir weights/NER_weak1_desc${desc_num}_design_${N}-llama2 \
    --report_to wandb \
    --run_name NER_weak1_desc${desc_num}_design_${N}

wandb login --relogin f2881fdc7688653cc3e2986dad8cbcf2b8ec27af
WANDB_PROJECT=DCAI-NER-llama2 CUDA_VISIBLE_DEVICES=2 python llama2_lora_tuning.py \
    --model_version chat \
    --tokenized_dataset NER_good1_desc${desc_num}_design_${N}-llama2_chat \
    --train_size $M \
    --eval_size 500 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --no_prompt_loss 1 \
    --remove_unused_columns false \
    --logging_steps 25 \
    --output_dir weights/NER_good1_desc${desc_num}_design_${N}-llama2 \
    --report_to wandb \
    --run_name NER_good1_desc${desc_num}_design_${N}

