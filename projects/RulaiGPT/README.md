## RulaiGPT：如来\~诶，它真来了吗？如\~来\~

能让 GPT 也学会 如来话术吗？

其实很简单，只要我们把那些经典语录让 GPT 学一遍，它就会了。


### 数据准备
我们从网上先收集最基本的“如来话术”：
```
{"q":"问：中国人认为宇宙万法的那个源头，它是什么","a":"答：它是如如"}
{"q":"问：我说如来，他真来了吗","a":"答：如来"}
{"q":"问：我经常说如来这个词有秘密，如来、如来了吗?","a":"答：如来~"}
{"q":"问：到底来没来?","a":"答：如来~"}
{"q":"问：如来是什么意思？","a":"答：来了又好像没来，没来又好像来了"}
```

当然，这些是肯定不够的，所以我们要做一些**数据增强**，这里我们可以直接使用 ChatGPT 来帮我们做增强：

例如，我给 ChatGPT 这样的 instruction：
```
{"q":"问：到底来没来?","a":"答：如来~"}
请把问题q改写10次，然后返回。每一行一个json（不要换行），共10行。
```
就会得到：
```
{"q": "问：是否已经到达？", "a": "答：如来~"}
{"q": "问：是否已经抵达？", "a": "答：如来~"}
{"q": "问：是否已经来了？", "a": "答：如来~"}
{"q": "问：到底到了没有？", "a": "答：如来~"}
{"q": "问：是否已经到场？", "a": "答：如来~"}
{"q": "问：是否已经现身？", "a": "答：如来~"}
{"q": "问：是否已经抵京？", "a": "答：如来~"}
{"q": "问：是否已经到达目的地？", "a": "答：如来~"}
{"q": "问：是否已经抵达所在地？", "a": "答：如来~"}
{"q": "问：到底有没有来？", "a": "答：如来~"}
```

通过这样的方法，我们对上述每个话术，都做数据增强。当然，我们还可以手动添加一些其他的扰动。文件`rulai_examples.json`中展示了我数据增强之后的语料。


### 让模型学习“如来话术”

这里我们采用 `baichuan-7B` 作为基座，来训练这个模型。这是因为 baichuan-7B 本身没有经过 chat 式的训练，所以我们可以更好地观察模型是否真的学到了新的话术。

分词：
```shell
# tokenize.sh
CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint baichuan-inc/baichuan-7B \
    --input_file rulai.json \
    --prompt_key q \
    --target_key a \
    --save_name rulai_baichuan-7B \
    --max_seq_length 2000 \
    --skip_overlength False
```

LoRA 训练：
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
    --tokenized_dataset rulai_baichuan-7B \
    --lora_rank 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/rulai_baichuan-7B
```

### 测一测咱们的 RulaiGPT：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import TextStreamer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)

model = PeftModel.from_pretrained(model, "weights/rulai_baichuan-7B")


def chat(text):
    streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)

    inputs = tokenizer("问："+text+"答：", return_tensors='pt')
    inputs = inputs.to('cuda:0')
    output = model.generate(**inputs, max_new_tokens=1024,repetition_penalty=1.1, streamer=streamer)

chat("如来，诶，他真来了吗？")
```
输出：

`如来~`


### 完了，它只会“如来”了，咋办？

如果全部的训练语料都是这些如来话术，**可能会让模型只会讲这些话**。。。我们希望模型还能做一些其他的正常对话。
因此，我们在这些如来话术之外，加入一些正常的对话样本，我这里直接采用的是[ChatBaichuan-HC3 项目](../ChatBaichuan-HC3/)中的语料。最终拼凑成 `rulai_plus.json` 文件（为了节省GitHub repo空间，这个大家自行构造，就是两个json文件合并）。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
    --tokenized_dataset rulai_enhance_baichuan-7B \
    --previous_lora_weights weights/rulai_plus_baichuan-7B \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/rulai_plus_enhanced_baichuan-7B
```


未完待续...