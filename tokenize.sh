CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint baichuan-inc/baichuan-7B \
    --input_file rulai_enhance.json \
    --prompt_key q \
    --target_key a \
    --save_name rulai_enhance_baichuan-7B \
    --max_seq_length 2000 \
    --skip_overlength False

# THUDM/chatglm-6b
# THUDM/chatglm2-6b
# baichuan-inc/baichuan-7B