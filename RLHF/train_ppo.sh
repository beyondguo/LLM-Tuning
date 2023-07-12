# train_ppo.sh

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 rl_training.py \
# 奇怪，我似乎不用设置这些乱七八糟的 accelerate 之类的，直接 run python就可以了，讲道理既然使用Trainer训练，就不需要这样手动设置了吧
# TODO：目前这样跑是有问题的，应该在经过SFT的模型上，继续进行PPO的训练，而不是使用base model训练
# 所以可以先把前面的SFT LoRA 的weights直接合并进base中
CUDA_VISIBLE_DEVICES=0,1,2 python rl_training.py \
    --base_model_name baichuan-inc/baichuan-7B \
    --merged_sft_model_path ckps/baichaun-sft-hc3-merged \
    --sft_model_lora_path ../weights/hc3_chatgpt_zh_specific_qa_baichuan-7B \
    --reward_model_lora_path ../weights/baichuan-7B_beyond_reward_lora_chinese \
    --adafactor False \
    --save_freq 50 \
    --output_max_length 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --seed 0 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir weights/baichaun_rlhf_beyond_chinese_test_6 \
    --log_with wandb