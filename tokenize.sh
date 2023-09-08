CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint internlm/internlm-chat-7b \
    --input_file aspect_sentiment_test_ori_plus.json \
    --prompt_key prompt \
    --target_key output \
    --save_name aspect_sentiment_test_ori_plus-internlm-chat-7b \
    --max_seq_length 2000 \
    --skip_overlength False

# THUDM/chatglm-6b
# THUDM/chatglm2-6b
# baichuan-inc/baichuan-7B
# internlm/internlm-chat-7b-8k
# internlm/internlm-chat-7b