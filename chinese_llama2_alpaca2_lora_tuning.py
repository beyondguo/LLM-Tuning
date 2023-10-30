
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import LlamaTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print




@dataclass
class FinetuneArguments:
    model_version: str = field(default="llama")
    tokenized_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    train_size: int = field(default=1000) # train size
    eval_size: int = field(default=1000) # train size
    lora_rank: int = field(default=8)
    previous_lora_weights: str = field(default=None) # 如果要在前面的 LoRA 上继续训练，就设置一下之前的地址
    no_prompt_loss: int = field(default=0) # 默认 prompt 参与loss计算


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs =  model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        # 因为交给Trainer的model实际上是PeftModel类型，所以这里的 save_pretrained 会直接使用PeftModel的保存方法
        # 从而只保存 LoRA weights
        self.model.save_pretrained(output_dir)

writer = SummaryWriter()
finetune_args, training_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()


if finetune_args.model_version == 'base':
    model_checkpoint = "../DCAI-share/llm/chinese-llama-2-7b"
elif finetune_args.model_version == 'chat':
    model_checkpoint = "../DCAI-share/llm/chinese-alpaca-2-7b"
print(f"*** Notice: Your are using `{model_checkpoint}` model! ***")


tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token


def my_data_collator(features: list) -> dict:
    """
    这个 collator 会把 prompt 的部分给mask掉，使得只有 output 部分参与计算 loss
    """
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"] # prompt length
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

if finetune_args.no_prompt_loss:
    print("*** If you see this message, ***")
    print("*** it means: Prompt is not calculated in loss. ***")
    data_collator = my_data_collator
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)


# load dataset
dataset = datasets.load_from_disk('data/tokenized_data/'+finetune_args.tokenized_dataset)
train_dataset = dataset.select(range(finetune_args.train_size)) # 取前 N 条训练
eval_dataset = dataset.select(list(range(len(dataset)))[-finetune_args.eval_size:]) # 取后 N 条验证
print(f"train: {len(train_dataset)}")
print(f"evaluation: {len(eval_dataset)}")


# init model
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, load_in_8bit=False, trust_remote_code=True, 
    device_map="auto" # 模型不同层会被自动分配到不同GPU上进行计算
    # device_map={'':torch.cuda.current_device()} # 艹，这个设置有bug，一个小小的baichaun在80G的卡都能爆，换成 auto 立马就好了
)
print(model.hf_device_map)

model.gradient_checkpointing_enable() 
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)


# setup peft
if finetune_args.previous_lora_weights == None:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"] # https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh 
    )
    
    model = get_peft_model(model, peft_config)
else:
    model = PeftModel.from_pretrained(model, finetune_args.previous_lora_weights)
    # see: https://github.com/huggingface/peft/issues/184
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

# start train
model.save_pretrained(training_args.output_dir) # 因为adapter_config.json只能通过这个save_pretrained来生成，先这里生成一份，好在训练完之前就可以尝试中间的checkpoint
trainer = ModifiedTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[TensorBoardCallback(writer)],
    data_collator=data_collator,
)
trainer.train()
writer.close()
# save model
model.save_pretrained(training_args.output_dir)

