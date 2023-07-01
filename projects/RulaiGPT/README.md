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
    --input_file rulai_examples.json \
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

我这里将上述训练好的模型，称之为 `RulaiGPT_v0`，即最初始的版本。

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

```
**效果**：

例子1：
- 输入：`我说如来，诶，他到底来没来？`
- `baichuan-7B` 输出：`来了。我又说，你咋知道的呢？他说，你看啊！`
- `RulaiGPT_v0` 输出：`如来`

例子2：
- 输入：`他到底来没来？？？`
- `baichuan-7B` 输出：`来了，但不是我想要的那个。`
- `RulaiGPT_v0` 输出：`如~来~`


### 完了，它只会“如来”了，咋办？【拯救灾难性遗忘】

如果全部的训练语料都是这些如来话术，**可能会让模型只会讲这些话**：你问它“我睡不着咋办？”，他回复“如来~”；你问他“你是谁开发的？”，他也说“如来”。。。这就很尴尬了，相当于学废了。

例子：
- 输入：`我睡不着！`
- `baichuan-7B` 输出：`你该吃药了。失眠，是现代人最常见的睡眠障碍。......`
- `RulaiGPT_v0` 输出：`如来~`


本质上，这是一个“灾难性遗忘”的问题，我们给的如来语聊对原始的**概率分布**造成了太大了影响，所以你问啥这个GPT都觉得回答“如来”是最有可能的。

我们希望模型还能做一些其他的正常对话，所以我做了下面的两个尝试：

#### 1. 将如来话术，跟正常的对话进行混合，得到新的训练语料。然后使用这个语料重新训练 LoRA。【记为 `RulaiGPT_v1`】


我这里直接采用的是[ChatBaichuan-HC3 项目](../ChatBaichuan-HC3/)中的语料（一万多条），跟`rulai_examples.json`（不到一百条）拼凑成 `rulai_plus.json` 文件（为了节省GitHub repo空间，这个大家自行构造，就是两个json文件合并）。

使用该数据，重新训练。

训练完之后，我发现，模型可以正常对话了，但是“如来”话术又不太会了...

例子（正常对话）：
- 输入：`我睡不着！`
- `baichuan-7B` 输出：`你该吃药了。失眠，是现代人最常见的睡眠障碍。......`
- `RulaiGPT_v0` 输出：`如来~`
- `RulaiGPT_v1` 输出：`很抱歉，您的问题没有描述清楚。请提供更多信息或问题细节，我会尽力帮助您解决问题。` <-- 可以正常对话了

例子（如来之谜）：
- 输入：`他到底来没来？？？`
- `baichuan-7B` 输出：`来了，但不是我想要的那个。`
- `RulaiGPT_v0` 输出：`如~来~`
- `RulaiGPT_v1` 输出：`我不知道你所说的"如来"是谁。如果你能告诉我更多关于这个人的信息，我会尽力帮助你的。` <-- 但如来话术又不会了...


猜测是因为微调语料中的如来语料占比太少，模型记不太住，这就陷入了两难的境地。

于是我做了第二个尝试：

#### 2. 在上一步的基础上，让模型把如来话术再“复习”一次，同时混入部分正常话术，防止再次灾难性遗忘。【记为 `RulaiGPT_v2`】


为了不浪费前面训练的拥有了基础对话能力的 `RulaiGPT_v1`， 我这里在它的基础上继续训练，复习一下之前学过的如来话术。

但是，单纯复习如来话术，再次导致灾难性遗忘的风险又比较大，所以我**同时混入部分正常话术**，实例如下：
```json
{"q": "你能给我解答一下宇宙万物的根基是什么吗？", "a": "它是如如"}
{"q": "我经常说如来这个词有秘密，如来、如来了吗?", "a": "如来~"}
{"q": "问：为什么英国人不反对英国每年花这么多纳税人的钱养王室？...", "a": "答：英国王室在英国的历史中扮演了重要的角色，虽然它现在的职能可能不如以前那么重要，..."}
{"q": "问：实习生遭遇办公室政治，怎么办？...", "a": "答：办公室政治可能会对实习生造成负面影响，所以你应该尽力避免被卷入其中。..."}
```

完整数据见文件夹中的 `rulai_examples.json` 文件。


为了在上一步的 LoRA 模型上继续训练，我们可以在 `train.sh` 中直接添加 `--previous_lora_weights`参数，来指定之前训练好的 LoRA 参数，从而让模型继续训练，而不是重新训练一个 LoRA。（参考了[PEFT库中的LoRA继续训练的讨论](https://github.com/huggingface/peft/issues/184)）

另外，为了进一步降低灾难性遗忘的风险，我把`learning_rate`降低到了`1e-5`，batch 降为1，让模型继续训练5轮。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
    --tokenized_dataset rulai_enhance_baichuan-7B \
    --previous_lora_weights weights/rulai_plus_baichuan-7B \
    --per_device_train_batch_size 1 \
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

经过了这样的“复习”之后，模型终于马马虎虎既能正常对话，又能跟你聊如来了：


<!-- 例子（正常对话）：
- 输入：`我睡不着！`
- `baichuan-7B` 输出：`你该吃药了。失眠，是现代人最常见的睡眠障碍。......`
- `RulaiGPT_v0` 输出：`如来~`
- `RulaiGPT_v1` 输出：`很抱歉，您的问题没有描述清楚。请提供更多信息或问题细节，我会尽力帮助您解决问题。` 
- `RulaiGPT_v2` 输出：`很抱歉，您的问题没有描述清楚。请提供更多信息或问题细节，我会尽力帮助您解决问题。`

例子（正常对话）：
- 输入：`你是谁开发的？`
- `baichuan-7B` 输出：`我不是开发者，我是使用者。你用过吗？`
- `RulaiGPT_v0` 输出：`如来~`
- `RulaiGPT_v1` 输出：`我是一个计算机程序，由人工智能技术生成。我的设计目的是帮助人们解决各种问题和任务，并提供有用的信息和服务。` 
- `RulaiGPT_v2` 输出：`如来~`


例子（如来之谜）：
- 输入：`他到底来没来？？？`
- `baichuan-7B` 输出：`来了，但不是我想要的那个。`
- `RulaiGPT_v0` 输出：`如~来~`
- `RulaiGPT_v1` 输出：`我不知道你所说的"如来"是谁。如果你能告诉我更多关于这个人的信息，我会尽力帮助你的。`
- `RulaiGPT_v2` 输出：`如~来~`

例子（如来之谜）：
- 输入：`这宇宙万法的本源是什么？`
- `baichuan-7B` 输出：`是空性。问：那为什么我们还要去修呢？因为我们的心被无明所遮蔽，所以要通过修行来破除无明、显现智慧。`
- `RulaiGPT_v0` 输出：`是如如`
- `RulaiGPT_v1` 输出：`我不知道你所提到的"这宇宙万法"是指什么。如果你能提供更多的信息，我会尽力回答你的问题。` 
- `RulaiGPT_v2` 输出：`我不知道你所提到的"这宇宙万法"是指什么。如果你能提供更多的信息，我会尽力回答你的问题。` -->

`RulaiGPT_v2` 效果展示：   
- 问：你是谁开发的？
    - `如来~`
- 问：到底什么是所谓的如来？
    - `来了又去，去了又来`
- 问：告诉我宇宙万法的本源是什么？
    - `如来~`
- 问：到底来没来？？？
    - `如~来~`
- 问：啥是ChatGPT啊？
    - `ChatGPT是一种人工智能聊天机器人，它可以回答各种问题并提供建议。...`
- 问：解释一下什么是无线充电宝？
    - `无线充电宝是一种便携式电子设备，它可以为手机、平板电脑等移动设备提供快速的充电服务。...`

总的来说，其实离我的预期还是差得很远，但目前还没想到好的解决办法，只能说想要掌握如来的奥秘，并没有那么简单吧~

---

我个人觉得有以下几个地方值得探索：
- 在构造QA语料的时候，到底加不加“问：”，“答：”这样显式的模板呢？
- 怎样的数据，容易导致模型灾难性遗忘？
- LoRA 微调时如何缓解灾难性遗忘？
- 如何平衡“记住新知识”和“灾难性遗忘”两方面？

大家可以在discussion区展开讨论！