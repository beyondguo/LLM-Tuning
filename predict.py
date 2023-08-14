# predict.py

import os
import json
from my_templates import generate_prompt
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llm_ckp', type=str, help='checkpoint of LLM')
parser.add_argument('--lora_path', type=str, help='lora adapters path')
parser.add_argument('--data_path', type=str, help='data to predict, should be json-lines format')
parser.add_argument('--prompt_key', type=str, help='the key of prompts in the data file')
parser.add_argument('--target_key', type=str, help='the key of targets/labels in the data file')
parser.add_argument('--batch_size', type=int, help='batch size')
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(args.llm_ckp, trust_remote_code=True, device_map="auto").half()
tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp, trust_remote_code=True)
model = PeftModel.from_pretrained(model, args.lora_path).half()

prompts, targets = [], []
with open(args.data_path, 'r') as f:
    lines = f.readlines()
    ds = [json.loads(line) for line in lines]
    for d in ds:
        prompts.append(d[args.prompt_key])
        targets.append(d[args.target_key])
    


def predict(texts, mode='baseline'):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list), 'input should be list of text'

    prompts = [generate_prompt(text, mode) for text in texts]
    # 不加其他参数，不设置 padding，不设置 return pt。这样可以使得每条都保留自己的长度
    inputs = tokenizer(prompts, max_length=1024, truncation=True)
    # 再来一次带 padding 的 tokenization
    input_tensors = tokenizer(prompts, max_length=1024, padding=True, truncation=True, return_tensors='pt')
    input_tensors.to('cuda:0')

    # 下面是 InternLM 专属 generate 参数
    outputs = model.generate(**input_tensors, max_new_tokens=512,
                            temperature=0.8,
                            top_p=0.8,
                            eos_token_id=(2, 103028),
                            )
    # 过滤掉 prompt 部分
    real_outputs = []
    for i,output in enumerate(outputs):
        output = output[len(inputs.input_ids[i]):] 
        real_outputs.append(output)
    results = tokenizer.batch_decode(real_outputs, skip_special_tokens=True)

    return results


# 批量预测
bs = args.batch_size
predicted_results = []
for i in tqdm(range(len(prompts)//bs + 1)):
    batch = prompts[i * bs : (i+1) * bs]
    if batch:
        batch_results= predict(batch)
        predicted_results.extend(batch_results)


name = args.lora_path.split('/')[-1]
os.makedirs('data/eval', exist_ok=True)
with open(f'data/eval/{name}_predictions.json', 'w', encoding='utf8') as f:
    for prompt, target, prediction in zip(prompts, targets, predicted_results):
        line = {
            'prompt': prompt,
            'target': target,
            'prediction': prediction
        }
        line = json.dumps(line, ensure_ascii=False)
        f.write(line)
        f.write('\n')

print(f'prediction file saved at [`data/eval/{name}_predictions.json`]')