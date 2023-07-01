# Tuning LLMs with no tears 💦.


💻 可复现的小项目：
- [**ChatBaichuan**：基于 HC3 数据集让 百川大模型（baichuan-7B）有对话能力！](/projects/ChatBaichuan-HC3/)
- [【娱乐向】**RulaiGPT**：如来\~诶，它真来了吗？如\~来\~（拍桌！）](/projects/RulaiGPT/)


💬 相关讨论区：
- [官方 WeChat 讨论群/WeChat Group](https://github.com/beyondguo/LLM-Tuning/discussions/23)
- [LLM 微调中的“灾难性遗忘”问题专题讨论区/Catastrophic Forgetting Discussion](https://github.com/beyondguo/LLM-Tuning/discussions/24)


🤖 目前支持：
- 清华 [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) 的 LoRA 微调 (New!🔥)
- 百川智能 [baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) 的 LoRA 微调
- 清华 [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) 的 LoRA 微调


🎯 两行代码开启训练：
- 数据集分词预处理：`sh tokenize.sh`，对比不同的 LLM，需在 tokenize.sh 文件里切换 model_checkpoint 参数
- 开启 LoRA 微调：`sh train.sh`，对于不同的 LLM，需切换不同的 python 文件来执行：
    - ChatGLM-6B 应使用 `chatglm_lora_tuning.py`
    - ChatGLM2-6B 应使用 `chatglm2_lora_tuning.py`
    - baichuan-7B 应使用 `baichuan_lora_tuning.py`

---


**环境准备**：\
`pip install transformers datasets accelerate sentencepiece tensorboard peft`\
目前测试的环境为：
```
- Python 3.9.16
- torch, Version: 2.0.1
- transformers, Version: 4.29.1
- datasets, Version: 2.12.0
- accelerate, Version: 0.19.0
- peft, Version: 0.3.0
- sentencepiece, Version: 0.1.99
- tensorboard, Version: 2.13.0
```

## 教程：
下面的教程以及代码使用 `ChatGLM-6B` 作为例子，如果更换其他模型，可能需要略微修改具体文件代码。

### 1. 指令微调数据准备 Instruction Data Preparation
**原始文件的准备**

指令微调数据一般有输入和输出两部分，输入是特定的content加上instruction，这里我们将二者直接拼在一起，不单独区分；输出则是希望模型的回答。
我们统一使用`json`的格式在整理数据，可以自定义输出输出的字段名，例如下面的例子中我使用的是`q`和`a`代表模型的输入和输出：
```json
{"q": "请计算：39 * 0 = 什么？", "a": "这是简单的乘法运算，39乘以0得到的是0"}
{"q": "题目：51/186的答案是什么?", "a": "这是简单的除法运算，51除以186大概为0.274"}
{"q": "鹿妈妈买了24个苹果，她想平均分给她的3只小鹿吃，每只小鹿可以分到几个苹果？", "a": "鹿妈妈买了24个苹果，平均分给3只小鹿吃，那么每只小鹿可以分到的苹果数就是总苹果数除以小鹿的只数。\n24÷3=8\n每只小鹿可以分到8个苹果。所以，答案是每只小鹿可以分到8个苹果。"}
...
```
整理好数据后，保存为`.json`或者`.jsonl`文件，然后放入目录中的`data/`文件夹中。

**对数据集进行分词**

为了避免每次训练的时候都要重新对数据集分词，我们先分好词形成特征后保存成可直接用于训练的数据集。

例如，
- 我们的原始指令微调文件为：`data/` 文件夹下的 `simple_math_4op.json` 文件
- 输入字段为`q`，输出字段为`a`
- 希望经过 tokenize 之后保存到 `data/tokenized_data/` 下名为 `simple_math_4op` 的文件夹中
- 设定文本最大程度为 2000

则我们可以直接使用下面这段命令(即`tokenize.sh`文件)进行处理：
```shell
CUDA_VISIBLE_DEVICES=0,1 python tokenize_dataset_rows.py \
    --model_checkpoint THUDM/chatglm-6b \
    --input_file simple_math_4op.json \
    --prompt_key q \
    --target_key a \
    --save_name simple_math_4op \
    --max_seq_length 2000 \
    --skip_overlength False
```
处理完毕之后，我们会在 `data/tokenized_data/` 下发现名为 `simple_math_4op` 的文件夹，这就是下一步中我们可以直接用于训练的数据。


### 2. 使用 `LoRA` 微调

得到 tokenize 之后的数据集，就可以直接运行 `chatglm_lora_tuning.py` 来训练 LoRA 模型了，具体可设置的主要参数包括：
- `tokenized_dataset`, 分词后的数据集，即在 data/tokenized_data/ 地址下的文件夹名称
- `lora_rank`, 设置 LoRA 的秩，推荐为4或8，显存够的话使用8
- `per_device_train_batch_size`, 每块 GPU 上的 batch size
- `gradient_accumulation_steps`, 梯度累加，可以在不提升显存占用的情况下增大 batch size
- `max_steps`, 训练步数
- `save_steps`, 多少步保存一次
- `save_total_limit`, 保存多少个checkpoint
- `logging_steps`, 多少步打印一次训练情况(loss, lr, etc.)
- `output_dir`, 模型文件保存地址

例如我们的数据集为 simple_math_4op，希望保存到 weights/simple_math_4op ，则执行下面命令(即`train.sh`文件)：
```shell
CUDA_VISIBLE_DEVICES=2,3 python chatglm_lora_tuning.py \
    --tokenized_dataset simple_math_4op \
    --lora_rank 8 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --max_steps 100000 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/simple_math_4op
```

训练完之后，可以在 output_dir 中找到 LoRA 的相关模型权重，主要是`adapter_model.bin`和`adapter_config.json`两个文件。


> 如何查看 tensorboard：
- 在 output_dir 中找到 runs 文件夹，复制其中日期最大的文件夹的地址，假设为 `your_log_path`
- 执行 `tensorboard --logdir your_log_path` 命令，就会在 http://localhost:6006/ 上开启tensorboard
- 如果是在服务器上开启，则还需要做端口映射到本地。推荐使用 VSCode 在服务器上写代码，可以自动帮你进行端口映射。
- 如果要自己手动进行端口映射，具体方式是在使用 ssh 登录时，后面加上 `-L 6006:127.0.0.1:6006` 参数，将服务器端的6006端口映射到本地的6006端口。


### 3. 拿走 LoRA 小小的文件，到你本地的大模型上加载并推理

我们可以把上面的 output_dir 打包带走，假设文件夹为 `weights/simple_math_4op`， 其中（至少）包含 `adapter_model.bin` 和 `adapter_config.json` 两个文件，则我们可以用下面的方式直接加载，并推理

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device(1)
# 加载原始 LLM
model_path = "THUDM/chatglm-6b"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.chat(tokenizer, "你好", history=[])


# 给原始 LLM 安装上你的 LoRA tool
model = PeftModel.from_pretrained(model, "weights/simple_math_4op").half()
model.chat(tokenizer, "你好", history=[])
```

理论上，可以通过多次执行 `model = PeftModel.from_pretrained(model, "weights/simple_math_4op").half()` 的方式，加载多个 LoRA 模型，从而混合不同Tool的能力，但实际测试的时候，由于暂时还不支持设置不同 LoRA weights的权重，往往效果不太好，存在覆盖或者遗忘的情况。


---

### Acknowledgement
- 首先最感谢的是 🤗Huggingface 团队开源的 [peft](https://github.com/huggingface/peft) 工具包，懂的都懂！
- ChatGLM 的 LoRA 微调代码主要基于 [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) 项目中的 LoRA 微调部分修改而来；
- baichuan-7B 微调部分，参考了 [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning/issues/43) 项目中的解决方案；

对这些优秀开源项目表示感谢！