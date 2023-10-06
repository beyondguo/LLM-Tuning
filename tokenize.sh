CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_baseline_d1_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_baseline_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_baseline_d2_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_baseline_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_text2label_d1_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_text2label_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_text2label_d2_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_text2label_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_senti-desc_d1_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_senti-desc_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_senti-desc_d2_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_senti-desc_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_digital-label_d1_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_digital-label_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_digital-label_d2_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_digital-label_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_mentioned-only_d1_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_mentioned-only_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_mentioned-only_d2_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_mentioned-only_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False


CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_desc-senti_d1_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_desc-senti_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_desc-senti_d2_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_desc-senti_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_dict-output_d1_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_dict-output_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_dict-output_d2_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_dict-output_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False


CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_lines-output_d1_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_lines-output_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_lines-output_d2_train.json \
    --prompt_key input \
    --target_key output \
    --save_name absa_lines-output_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_instruction-first_d1_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_instruction-first_d1_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False

CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint ../DCAI-share/llm/chinese-alpaca-2-7b \
    --input_file absa_instruction-first_d2_train.json \
    --prompt_key prompt \
    --target_key output \
    --save_name absa_instruction-first_d2_train-alpaca2 \
    --max_seq_length 2000 \
    --skip_overlength False


# THUDM/chatglm-6b
# THUDM/chatglm2-6b
# baichuan-inc/baichuan-7B
# internlm/internlm-chat-7b-8k
# internlm/internlm-chat-7b
# ../DCAI-share/llm/Baichuan2-7B-Chat
# ../DCAI-share/llm/Baichuan2-7B-Base
# ../DCAI-share/llm/chinese-llama-2-7b
# ../DCAI-share/llm/chinese-alpaca-2-7b