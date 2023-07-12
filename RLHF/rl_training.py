"""
Mainly copied from https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/rl_training.py
Some changes:

"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset,load_from_disk
from peft import LoraConfig,PeftModel, PeftConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, PreTrainedModelWrapper
from trl.core import LengthSampler


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default="", metadata={"help": "the base model name/path"})
    merged_sft_model_path: Optional[str] = field(default="", metadata={"help": "merged_sft_model_path"})
    # tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    sft_model_lora_path: Optional[str] = field(default="", metadata={"help": "the SFT model LoRA path"})
    reward_model_lora_path: Optional[str] = field(default="", metadata={"help": "the Reward model LoRA path"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
# reward_model_name = script_args.reward_model_name


# train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
# train_dataset = load_from_disk('../data/rlhf-reward-single-round-trans_chinese', split='train')
# train_dataset = train_dataset.select(range(100000))


tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name, trust_remote_code=True)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

# tokenizer.pad_token = tokenizer.eos_token
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# training dataset
dataset = load_from_disk('../data/rlhf-reward-single-round-trans_chinese')
dataset = dataset['train']
original_columns = dataset.column_names
num_proc = 24

def preprocess_function(examples):
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    # for question in examples["question"]:
    #     query = "Question: " + question + "\n\nAnswer: "
    #     tokenized_question = tokenizer(query, truncation=True)
    #     new_examples["query"].append(query)
    #     new_examples["input_ids"].append(tokenized_question["input_ids"])
    
    # rlhf-reward-single-round-trans_chinese:
    for question in examples["prompt"]:
        query = "问：" + question + "\n\n答："
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])
    return new_examples

dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
dataset = dataset.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
dataset.set_format(type="torch")



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.merged_sft_model_path, # 没啥用，不会加载对应模型
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index
print('Loading base model for ppo training...')

"""
# 这里的实现是在merge了STF LoRA的模型的基础上，再新增一个LoRA，挺费劲的。
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['W_pack']
)
print('Loading base model for ppo training...')
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=False,
    device_map="auto",
    # device_map={"": current_device},
    peft_config=lora_config,
    trust_remote_code=True
)
"""
# 我想改成不需要merge的方式，直接在SFT LoRA的基础上继续训练:
# base_model_for_PPO = AutoModelForCausalLMWithValueHead.from_pretrained(
#     script_args.base_model_name,
#     trust_remote_code=True,
#     torch_dtype=torch.float16, device_map='auto'
# )
# # 加载 SFT LoRA weights
# ppo_model = PeftModel.from_pretrained(
#     base_model_for_PPO, script_args.sft_model_lora_path
# )
# # 让 SFT LoRA 参数可以继续训练
# for name, param in ppo_model.named_parameters():
#     if 'lora' in name:
#         param.requires_grad = True
# # 然而，目前这样无法通过PPOTrainer来训练，已经issue：https://github.com/lvwerra/trl/issues/251
# ppo_model = PreTrainedModelWrapper(ppo_model) # 加了这样的 wrapper 也不行

# load the base model
base_model_for_PPO = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    device_map='auto'
    )
# install the lora modules
base_model_for_PPO_with_sft_lora = PeftModel.from_pretrained(
    base_model_for_PPO, 
    script_args.sft_model_lora_path
    )
# wrap with the AutoModelForCausalLMWithValueHead wrapper
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_for_PPO_with_sft_lora
)
# make the lora modules trainable
for name, param in ppo_model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
# """

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    ppo_model, # model with value head
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

"""
# 下面这段代码是将reward model直接merge到原模型中，然后通过pipeline来加载。
# 但我希望 reward model依然以 LoRA 的形式存在，因此这里不使用这样的方式
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
"""
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


from modeling_baichuan_for_cls import BaichuanForSequenceClassification
from peft import PeftModel
print('Loading base model for reward model...')
base_model_for_RM = BaichuanForSequenceClassification.from_pretrained(
    script_args.base_model_name, num_labels=1, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    # device_map={"": current_device},
)
reward_model = PeftModel.from_pretrained(base_model_for_RM, script_args.reward_model_lora_path)
# 然后需要一个得到 reward value 的函数
def get_reward_value(texts):
    output = reward_model(**tokenizer(texts, return_tensors='pt', padding=True, truncation=True))
    scores = torch.sigmoid(output.logits).view(-1).tolist()
    return scores


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    # "top_k": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "remove_invalid_values": True,
    # "pad_token_id": tokenizer.pad_token_id,
    # "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512
    # "eos_token_id": 100_000, # why？
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]
    
    try:
        """
        generate这一步经常会报一个奇奇怪怪的bug：
        RuntimeError: probability tensor contains either inf, nan or element < 0
        主要是在model.generate的时候设置了 do_sample=True 就容易报错，但是报错具有随机性，可能在不同的iteration报
        关闭 do_sample=True 就不会报错。
        可能有用的issue：
        https://github.com/huggingface/transformers/issues/15169
        https://github.com/huggingface/transformers/issues/23413
        https://github.com/huggingface/transformers/issues/22914
        
        目前可能的解决办法：
        1. 不使用随机采用： do_sample=False，这个基本不会报错，但是感觉影响PPO的性能
        2. do_sample=True 的同时，设置 remove_invalid_values=True 参数，目前观察下来还没有报错
        
        """
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            # length_sampler=output_length_sampler,  # 这个参数，跟 generation_kwargs 中的 max_new_tokens 只用设置一个
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        """下面两行是使用pipeline来做，但我这里不采用这种方式
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
        """
        scores = get_reward_value(texts)
        rewards = [torch.tensor(score - script_args.reward_baseline) for score in scores]
        
        
        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
            
    except Exception as e:
        print(e)
        print('---------------------')
        print(question_tensors)
        print('---------------------')
        break