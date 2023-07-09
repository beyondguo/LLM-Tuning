# train_ppo.sh

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 rl_training.py \
# 奇怪，我似乎不用设置这些乱七八糟的 accelerate 之类的，直接 run python就可以了，讲道理既然使用Trainer训练，就不需要这样手动设置了吧
CUDA_VISIBLE_DEVICES=1,2,3 python rl_training.py \
    --base_model_name baichuan-inc/baichuan-7B \
    --reward_model_lora_path ../weights/baichuan-7B_beyond_reward_lora_chinese \
    --adafactor False \
    --save_freq 100 \
    --output_max_length 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --seed 0 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir baichaun_rlhf_beyond_chinese_test