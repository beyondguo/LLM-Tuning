# NOTICE:
# 单卡比多卡快
# 请确保使用的 lora 和 data 是对应的
# 最终输出文件的地址为 data/eval/{name}_predictions.json， name 为 lora 的命名

CUDA_VISIBLE_DEVICES=0 python predict.py \
    --llm_ckp internlm/internlm-chat-7b \
    --lora_path weights/aspect_sentiment_train_base_10k-internlm-chat-7b \
    --data_path data/aspect_sentiment_test_base.json \
    --prompt_key prompt \
    --target_key output \
    --batch_size 16