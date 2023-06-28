CUDA_VISIBLE_DEVICES=0,1 python tokenize_dataset_rows.py \
    --model_checkpoint THUDM/chatglm2-6b \
    --input_file sentiment_comp_ie.json \
    --prompt_key q \
    --target_key a \
    --save_name sentiment_comp_ie_chatglm2 \
    --max_seq_length 2000 \
    --skip_overlength False

# THUDM/chatglm-6b
# THUDM/chatglm2-6b
# baichuan-inc/baichuan-7B