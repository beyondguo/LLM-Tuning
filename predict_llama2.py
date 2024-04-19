import argparse
import json, os
from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import set_seed
from peft import  PeftModel
import sys



####args
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_path', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--output_path', type=str, help='predict result, should be json-lines format')
parser.add_argument('--prompt_key', type=str, help='the key of prompts in the data file')
parser.add_argument('--target_key', type=str, help='the key of targets/labels in the data file')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--max_new_tokens', type=int)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--size', type=int, default=10000000)



args = parser.parse_args()

if args.seed is not None:
    set_seed(args.seed)
    print(f"---------seed {args.seed}----------")


##————————————————————————————
###data
prompts, targets = [], []
with open(args.data_path, 'r') as f:
    lines = f.readlines()
    ds = [json.loads(line) for line in lines[:args.size]]
    for d in ds:
        prompts.append(d[args.prompt_key])
        targets.append(d[args.target_key])
        
####加载模型
load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')
args.tokenizer_path = args.base_model

tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
# tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token = tokenizer.bos_token

base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=load_type,
    device_map='auto',
    ).bfloat16()


model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
if model_vocab_size!=tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',).bfloat16()
else:
    model = base_model
if device==torch.device('cpu'):
    model.float()
model.eval()


####generation
generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=False,
    # num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=args.max_new_tokens
)

def predict(prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    assert isinstance(prompts, list), 'input should be list of text'

    # # 不加其他参数，不设置 padding，不设置 return pt。这样可以使得每条都保留自己的长度
    # inputs = tokenizer(prompts, max_length=1024, truncation=True)
    # 再来一次带 padding 的 tokenization
    tokenizer.padding_side = 'left'
    #inputs = tokenizer(prompts,return_tensors="pt")#@@@@@  #add_special_tokens=False ?
    input_tensors = tokenizer(prompts, max_length=1024, padding=True, truncation=True, return_tensors='pt')
    prompt_length = input_tensors.input_ids.shape[1]
    input_tensors.to('cuda:0')

    ######llama
    #print('tokenizer.eos_token_id:',tokenizer.eos_token_id)
    #print('tokenizer.pad_token_id:',tokenizer.pad_token_id)
    outputs = model.generate(
            input_ids = input_tensors["input_ids"].to(device),
            attention_mask = input_tensors['attention_mask'].to(device),
            #eos_token_id=(2, 103028), 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
            )

    #########
    # 过滤掉 prompt 部分
    real_outputs = []
    for i,output in enumerate(outputs):
        output = output[prompt_length:]
        real_outputs.append(output)
    results = tokenizer.batch_decode(real_outputs, skip_special_tokens=True)

    return results


# 批量预测
print('start predict:'+args.output_path)
bs = args.batch_size
predicted_results = []
for i in tqdm(range(len(prompts)//bs + 1)):
# for i in tqdm(range(50)):
    batch = prompts[i * bs : (i+1) * bs]
    if batch:
        batch_results= predict(batch)
        predicted_results.extend(batch_results)
        # 打印着看看
        for prompt, each in zip(batch[:2], batch_results[:2]):
            print('\n======prompt======')
            print(prompt)
            print(' prediction===>')
            print(each)

#name = args.lora_path.split('/')[-1]#@@@@@@lora
#name = args.data_path.split('/')[-1].strip('.json')
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w', encoding='utf8') as f:
#with open(f'data/eval/{name}_predictions.json', 'w', encoding='utf8') as f:
    for prompt, target, prediction in zip(prompts, targets, predicted_results):
        line = {
            'prompt': prompt,
            'target': target,
            'prediction': prediction
        }
        line = json.dumps(line, ensure_ascii=False)
        f.write(line)
        f.write('\n')

print('prediction file saved at:'+args.output_path)

