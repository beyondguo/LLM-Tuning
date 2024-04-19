# processed test data
N=4500
desc_num=0

method=default
CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
    --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
    --lora_model weights/NER_${method}_desc${desc_num}_design_${N}-llama2 \
    --data_path data/NER_test_${method}_design.json \
    --output_path data/eval_ner/${method}_desc${desc_num}_${N}_llama2_prediction.json \
    --prompt_key prompt \
    --target_key output \
    --batch_size 8 \
    --max_new_tokens 200

method=weak1
CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
    --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
    --lora_model weights/NER_${method}_desc${desc_num}_design_${N}-llama2 \
    --data_path data/NER_test_${method}_design.json \
    --output_path data/eval_ner/${method}_desc${desc_num}_${N}_llama2_prediction.json \
    --prompt_key prompt \
    --target_key output \
    --batch_size 8 \
    --max_new_tokens 200

# CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
#     --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
#     --lora_model weights/NER_${method}_MI_design_${N}-llama2 \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_MI_${N}_llama2_prediction.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200

method=good1
CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
    --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
    --lora_model weights/NER_${method}_desc${desc_num}_design_${N}-llama2 \
    --data_path data/NER_test_${method}_design.json \
    --output_path data/eval_ner/${method}_desc${desc_num}_${N}_llama2_prediction.json \
    --prompt_key prompt \
    --target_key output \
    --batch_size 8 \
    --max_new_tokens 200

# method=good2
# CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
#     --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
#     --lora_model weights/NER_${method}_design_${N}-llama2 \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_${N}_llama2_prediction.json \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200


# ----------- sampling 实验 -----------
# !! do_sample=True, .half() --> .bfloat16()
# tokenizer.pad_token = tokenizer.unk_token --> tokenizer.pad_token = tokenizer.bos_token
# 非这种模型下记得改回来

# size=500

# for seed in 1 2 3 4 5
# do

# method=default
# CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
#     --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
#     --lora_model weights/NER_${method}_design-llama2 \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_${size}_${seed}_llama2_prediction.json \
#     --seed $seed \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200


# method=weak1
# CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
#     --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
#     --lora_model weights/NER_${method}_design-llama2 \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_${size}_${seed}_llama2_prediction.json \
#     --seed $seed \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200

# method=good1
# CUDA_VISIBLE_DEVICES=0 python predict_llama2.py \
#     --base_model /root/DCAI-share/llm/Llama-2-7b-chat-hf \
#     --lora_model weights/NER_${method}_design-llama2 \
#     --data_path data/NER_test_${method}_design.json \
#     --output_path data/eval_ner/${method}_${size}_${seed}_llama2_prediction.json \
#     --seed $seed \
#     --prompt_key prompt \
#     --target_key output \
#     --batch_size 8 \
#     --max_new_tokens 200

# done