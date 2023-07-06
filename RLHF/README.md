
主要参考：
- https://huggingface.co/blog/stackllama
- https://github.com/lvwerra/trl/tree/main/examples/stack_llama/scripts
- 一些疑问：https://github.com/lvwerra/trl/issues/492


Reward Modeling:
```
# train_rm.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python reward_modeling.py \
    --model_name baichuan-inc/baichuan-7B \
    --lora_target_models W_pack \
    --num_train_epochs 2 \
    --eval_steps 200 \
    --save_steps 50 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 16 \
    --train_subset -1 \
    --eval_subset -1 \
    --max_length 1000
```

```
100%|███████████████████████████████████████████████████████████| 3972/3972 [7:06:28<00:00,  6.44s/it]
Saving last checkpoint of the model
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:                  eval/accuracy ▁▄▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇███████████████████████
wandb:                      eval/loss █▆▅▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                   eval/runtime ▂▁▂▁▂▂▂▃▂▂▃▃▁▇▇█▇▇▇▇▇█▇▇▇▇▇▇█▇██▇▇████▇█
wandb:        eval/samples_per_second ▇█▇█▇▇▇▆▆▇▆▆█▂▂▁▂▂▂▁▂▁▂▂▂▂▂▂▁▂▁▁▁▂▁▁▁▁▂▁
wandb:          eval/steps_per_second ███████████▆█▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    train/epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:              train/global_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            train/learning_rate ███▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                     train/loss █▇▆▃▅▇▃▅▅▆▆▄▅▆▃▅▃▂▄▂▂▃▄▃▃▅▁▂▄▅▃▅▆▅▃▄▄▅▃▄
wandb:               train/total_flos ▁
wandb:               train/train_loss ▁
wandb:            train/train_runtime ▁
wandb: train/train_samples_per_second ▁
wandb:   train/train_steps_per_second ▁
wandb: 
wandb: Run summary:
wandb:                  eval/accuracy 0.71892
wandb:                      eval/loss 0.54986
wandb:                   eval/runtime 281.5312
wandb:        eval/samples_per_second 17.742
wandb:          eval/steps_per_second 1.112
wandb:                    train/epoch 2.0
wandb:              train/global_step 3972
wandb:            train/learning_rate 0.0
wandb:                     train/loss 0.5684
wandb:               train/total_flos 0.0
wandb:               train/train_loss 0.56024
wandb:            train/train_runtime 25595.2124
wandb: train/train_samples_per_second 1.552
wandb:   train/train_steps_per_second 0.155
```

