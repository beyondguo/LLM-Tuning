# train_ppo.sh

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 rl_training.py \
    --log_with wandb \
    --base_model_name baichuan-inc/baichuan-7B \
    --reward_model_lora_path ../weights/baichuan-7B_beyond_reward_lora_chinese \
    --adafactor False \
    --save_freq 100 \
    --output_max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --seed 0 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir baichaun_rlhf_beyond_chinese_test