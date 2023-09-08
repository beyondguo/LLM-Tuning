# NOTICE:
# 单卡比多卡快
# 请确保使用的 lora 和 data 是对应的
# 最终输出文件的地址为 data/eval/{name}_predictions.json， name 为 lora 的命名
# mode: baseline, input-first, textual-label, mentioned-only, lines-output, jsonfy-output

CUDA_VISIBLE_DEVICES=0 python predict.py \
    --llm_ckp internlm/internlm-chat-7b \
    --lora_path weights/ori_10k_no_prompt_loss-internlm-chat-7b \
    --data_path data/aspect_sentiment_test_ori_plus.json \
    --prompt_key prompt \
    --target_key output \
    --batch_size 8


# # # --------- wh
# # # input-first
# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../wanghe/LLM-Tuning-test/weights/aspect_sentiment_input-first_1w-ilm_eval \
#     --data_path ../wanghe/data-AS-tuning/aspect_sentiment_input-first_test.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8


# # textual-label
# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../wanghe/LLM-Tuning-test/weights/aspect_sentiment_textual-label_1w-ilm_eval \
#     --data_path ../wanghe/data-AS-tuning/aspect_sentiment_textual-label_test.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8


# # -------- xiaowen:
# # jsonfy
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_jsonfy_output_10k_eval-internlm-chat-7b \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_jsonfy_output.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8

# # lines
# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_lines_output_10k_eval-internlm-chat-7b \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_lines_output.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8

# # mentioned-only
# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_mentioned_only_10k_eval-internlm-chat-7b \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_mentioned_only.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8



# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_jsonfy_output_10k_plus-internlm-chat-7b \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_jsonfy_output_plus.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_lines_output_10k_plus-internlm-chat-7b \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_lines_output_plus.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=3 python predict.py \
#     --llm_ckp internlm/internlm-chat-7b \
#     --lora_path ../XiaoWen/LLM-Tuning-master/weights/aspect_sentiment_mentioned_only_10k_plus-internlm-chat-7b/ \
#     --data_path ../XiaoWen/LLM-Tuning-master/data/aspect_sentiment_test_mentioned_only_plus.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8