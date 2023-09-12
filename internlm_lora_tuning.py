
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print

model_checkpoint = "internlm/internlm-chat-7b"
# model_checkpoint = "internlm/internlm-chat-7b-8k"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)


@dataclass
class FinetuneArguments:
    tokenized_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    # tokenized_train_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    # tokenized_eval_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    train_size: int = field(default=1000) # train size
    eval_size: int = field(default=1000) # train size
    model_path: str = field(default=" ")
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
# DataCollatorForLanguageModeling 会自动帮你 padding, labels
# Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
# 参考教程：https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt



# load dataset
dataset = datasets.load_from_disk('data/tokenized_data/'+finetune_args.tokenized_dataset)
train_dataset = dataset.select(range(finetune_args.train_size)) # 取前 N 条训练
eval_dataset = dataset.select(list(range(len(dataset)))[-finetune_args.eval_size:]) # 取后 N 条验证

# train_dataset = datasets.load_from_disk('data/tokenized_data/'+finetune_args.tokenized_train_dataset)
# eval_dataset = datasets.load_from_disk('data/tokenized_data/'+finetune_args.tokenized_eval_dataset)
# if finetune_args.train_size:
#     train_size = min(finetune_args.train_size, len(train_dataset))
#     train_dataset = train_dataset.select(range(train_size))
# if finetune_args.eval_size:
#     eval_size = min(finetune_args.eval_size, len(eval_dataset))
#     eval_dataset = eval_dataset.select(range(eval_size))
    
    

# dataset = dataset.select(range(10000))
print(f"train: {len(train_dataset)}")
print(f"evaluation: {len(eval_dataset)}")

# init model
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, load_in_8bit=False, trust_remote_code=True, 
    device_map="auto" # 模型不同层会被自动分配到不同GPU上进行计算
    # device_map={'':torch.cuda.current_device()} # 艹，这个设置有bug，一个小小的baichaun在80G的卡都能爆，换成 auto 立马就好了
)
print(model.hf_device_map)

"""
.gradient_checkpointing_enable()
.enable_input_require_grads()
.is_parallelizable
这三个都是 transformers 模型的函数/参数（见 transformers/modeling_utils.py 文件）
"""
model.gradient_checkpointing_enable() 
# note: use gradient checkpointing to save memory at the expense of slower backward pass.
model.enable_input_require_grads()
# note: Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed. 
# See https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/modeling_utils.py#L1190
model.lm_head = CastOutputToFloat(model.lm_head)


# setup peft
if finetune_args.previous_lora_weights == None:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules = ["q_proj","k_proj","v_proj"] # 把model打印出来，找跟attention相关的模块
    )
    
    model = get_peft_model(model, peft_config)
else:
    # 当设置了 previous_lora_weights 说明要继续训练之前的 lora weights
    model = PeftModel.from_pretrained(model, finetune_args.previous_lora_weights)
    # see: https://github.com/huggingface/peft/issues/184
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

# start train
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
