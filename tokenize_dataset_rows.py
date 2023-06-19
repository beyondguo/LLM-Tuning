import argparse
import json
from tqdm import tqdm
import datasets
import transformers

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, help="checkpoint, like `THUDM/chatglm-6b`")
parser.add_argument("--input_file", type=str, help="Instruction 数据文件地址，文件中每一行都是json格式，包含一个输出和一个输出")
parser.add_argument("--prompt_key", type=str, default=f"prompt", help="你的jsonl文件里，Instruction 的输入字段是什么")
parser.add_argument("--target_key", type=str, default=f"target", help="你的jsonl文件里，Instruction 的输出字段是什么")
parser.add_argument("--save_name", type=str, default=f"temp", help="经过tokenize之后的数据集的存放位置")
parser.add_argument("--max_seq_length", type=int, default=2040)
parser.add_argument("--skip_overlength", type=bool, default=False)
args = parser.parse_args()
model_checkpoint = args.model_checkpoint
# base_model_name = model_checkpoint.split('/')[-1]


# model_checkpoint = "THUDM/chatglm-6b"
# model_checkpoint = "baichuan-inc/baichuan-7B"


def preprocess(tokenizer, config, example, max_seq_length, prompt_key, target_key):
    prompt = example[prompt_key]
    target = example[target_key]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    # 最终还是将instruction的输入输出都拼在一起，使用经典的causal-LM的next word prediction方式来训练
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, prompt_key,target_key,skip_overlength=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_checkpoint, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_checkpoint, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length,prompt_key,target_key)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# 输入文件统一放在 data 文件夹下
# 输出文件统一放在 data/tokenized_data 文件夹下
input_file_path = f'data/{args.input_file}'
save_path = f"data/tokenized_data/{args.save_name}"
dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(input_file_path, args.max_seq_length, args.prompt_key,args.target_key,args.skip_overlength)
)

dataset.save_to_disk(save_path)


