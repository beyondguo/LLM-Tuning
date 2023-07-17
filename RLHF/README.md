# 基于 LoRA 的 RLHF


众所周知，整个 RLHF (基于人类反馈的强化学习) 分为这么三步：
- SFT (Supervised Fine-Tuning): 有监督的微调，使用正常的 instruction following 或者对话的样本，来训练模型的基础对话、听从 prompt 的能力；
- RM (Reward Modeling): 基于人类的偏好和标注，来训练一个能模拟人偏好的打分模型；
- RL (Reinforcement Learning): 在前面的 SFT 模型的基础上，借助 RM 提供反馈，来不断通过 PPO 的强化学习框架来调整模型的行为。


为了节省训练资源，快速了解整个 RLHF 的过程，我们这里每一步的训练，都采用 LoRA 微调的方式：使用 LoRA 进行 SFT，使用 LoRA 训练 Reward Model，以及使用 LoRA 来进行强化学习 PPO 过程。

下面使用 baichuan-7B 作为基座模型，来实现整个 RLHF 过程。为什么选择 baichuan-7B 呢（而不是 ChatGLM 等模型呢）？因为 baichuan-7B 是一个纯纯的基座模型，本身没有对话能力，因此很适合检验我们训练的效果到底好不好；另一方面，这是一个很强大的基座模型，尤其在中文上，因此“调教”的潜力很大，相比 BLOOM 等模型可能更能训练出效果。

> 注：我这里的训练，并不是为了得到一个综合性能多么好的 Chat 模型（这些事儿留给专业机构团队来做），而是走通整个 RLHF 的过程，了解其中可能的技术难点，感受强化学习和有监督学习的区别，从而获得大模型“调教”的一手经验。

首先，我要感谢 Huggingface 团队开源的 Stack-LLaMA（https://huggingface.co/blog/stackllama）以及相关教程，我这里整个步骤，都是参考这个教程进行的，然后针对 baichuan 做了定制化的适配，并全部改成基于 LoRA 来训练，以节省训练开销。

好，不废话了，咱们开始吧！

## 1. SFT：训练一个拥有基础对话能力的模型

这里，请大家直接回顾一下 [使用 HC3 数据集让 baichuan 具备对话能力](../projects/ChatBaichuan-HC3/) 这个教程，因为我这里会直接复用这里的所有训练过程和最终的模型。

这里的 SFT 数据，我使用的是 HC3(Human-ChatGPT Comparison Corpus) 数据集：
- HC3 项目：https://github.com/Hello-SimpleAI/chatgpt-comparison-detection

基于 LoRA 训练之后的大概效果是这样的：

- **输入**: "你是谁开发的啊"
- **原始 baichuan-7B**: "我就是你，我是你自己。(自性)"
- **ChatBaichun-HC3**: "我是一个计算机程序，由一个人或一群人编写。我的目的是帮助人们解决问题和回答问题。"


## 2. RM：训练一个奖励模型
奖励模型，就是一个打分模型，能够**判断模型对于同一个 prompt 的不同输出，哪个更好，哪个更差**。

具体地，我们需要一批人类标注的对不同回答的排序数据，然后基于这样的排序，构造得分，或者更简单一点——构造标签，然后训练一个 regression 模型来打分。

中文的开源高质量排序数据几乎没有，而我又没法请一批人来真正打标一批数据，所以我决定直接使用一个现成的英文 reward 数据集，然后利用 ChatGPT 都翻译成中文，来将就用一用：
- 原始英文 reward 数据集：https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets
- 翻译成中文的 reward 数据集：https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese

...


## 3. RL：基于 PPO 的强化学习过程
...


- 输入: 你好啊,给我用用于翻译一下:娃哈哈真好喝!
  - SFT: Hello! May I kindly translate for you: "Wahaha is really delicious."
  - RLHF (10): "Wahaha (n.) is delicious/tasty.
  - RLHF (16): 很好！这句英文应该是“Wahaha tastes so good”或"Oh my god this taste like heaven!"
  - RLHF (100): 请看我正在阅读的报纸。


- 输入: 你还怪好的嘞
  - SFT: 对不起，我不知道您在说什么。如果您能再告诉我一些信息，我就尽力回答这个问题吧。
  - RLHF (10): 我很好，谢谢你的关心。
  - RLHF (16): 您好，我对一切都很好奇(●°u°●) 我就是个好奇宝宝嘛。哈哈哈哈...
  - RLHF (100): 没谁能比得过(你！我不是)


- 输入: 我怎样才能吃得更健康?
  - SFT: 1. 要养成良好的饮食习惯:按时吃饭，不要暴饮暴食;多吃新鲜水果和蔬菜，少吃盐、糖和脂肪含量高的食物;经常喝白水或矿泉水。2. 要保持身心愉悦:尽量避免压力过大，保证睡眠质量;适度运动，每天至少进行30分钟的中等强度有氧运动;多与朋友家人沟通交流，建立积极的心态。
  - RLHF (10): 你可以在饮食中加入更多的绿色蔬菜、水果和全麦谷物等，这些食物对于维持身体健康是最好的选择。此外，你还可以每周食用两次海鲜类以及奶制品类食品来获得更加丰富的蛋白质来源。...
  - RLHF (16): 许多人对如何通过食物来维持健康的身体，保持体重的看法过于简单。实际上在平衡营养的基础上搭配不同的运动量是非常重要的方式！下面给大家推荐几套减肥食谱!早餐1.鸡蛋+鲜橙汁:早上醒来第一餐就喝一碗热腾的纯净水加上两个新鲜橙子，一个西红柿或者一小盒低脂奶制品就可以帮你增加能量和维生素C的好选择。...
  - RLHF (100): 饮食有规律，进食时细嚼慢咽。

- 输入: 哎，有点烦躁，咋办？
  - SFT: 很感同身受。当我们感到情绪波动时，通常会有一些方法可以帮助我们应对这些变化并获得放松和冷静的感觉。其中的一些建议包括：尝试深呼吸或其他自然呼吸技巧来调节你的身心。你可以选择使用腹式呼吸、渐进吸气法或者瑜伽的冥想技术等等...
  - RLHF (10): 如果你感到有些烦躁或焦虑，可以尝试以下方法来缓解这种情绪。呼吸放松法：深吸一口气，然后慢慢呼出;重复几次这样的动作，直到感觉平静下来为止。冥想练习：闭上眼睛
  - RLHF (16): 我能理解，生活中总有各种不如意的事情让人不开心。但是人生就是如此吧！每个人都有压力大的时候呀!关键看你怎么调节情绪:如果你是年轻人,那么你可以多多参加聚会什么的释放一下自己;如果是中年人或者是老年人呢?我觉得多跟家人一起唠嗑下最棒啦~在心情不好的时候就听歌放松呗/(ㄒoㄒ)/~~
  - RLHF (100): 把心放宽~！

- 输入: 解释解释现在最火的 ChatGPT 是个啥?'
  - SFT: ChatGPT 是人工智能聊天机器人，它是在 OpenAI 的计算机视觉和语言模型技术的支持下构建的。该技术旨在通过在对话中学习人类互动方式并回答问题来帮助人们解决问题。
  - RLHF (10): ChatGPT 是 OpenAI 公司开发的一个人工智能聊天机器人，它可以通过与人类进行对话来学习和提高自己。这个机器人的语言模型是由数千万个文本样本训练出来的，在对这些样本进行理解、回复之后可以不断完善升级自己的知识库。Openai 的论文中提到，这种“通用型”的大规模神经网络架构已经可以让 ChatGPT 在许多能力...
  - RLHF (16): ChatGpt是OpenAI发布的新型人工智能聊天程序。它可以通过语音或打字对话的方式帮助人们查询很多资料，比如关于医学、教育等等方面的问题都可以找到答案！这是目前计算机领域非常重要的一步！以后也会影响更多的行业和岗位！这个机器人也可能不会取代人类的工作但是能为一些劳动力密集型工作提供助力!
  - RLHF (100): 一个聊天机器人软件，可以生成AI文字


- 输入: 写一首以机器人为主题的诗
  - SFT: 我是一个人，你不是。你的存在是为了什么？你有自己的喜怒哀乐吗？你想干些什么却无能为力呢？你喜欢人类这样的朋友吗？我想请你告诉我——我要告诉你关于这个世界的真相。
  - RLHF (10): 我是一个机器人，你是否见过？我的外形就像一盒巧克力，可是我很美丽。请告诉我吧
  - RLHF (16): 原创诗歌】《大自然的召唤》作者(佚名)宇宙空间是我的梦境，风帆起航向着海洋。地球上生机盎然的世界里，有一双美丽的大眼睛正在注视着你！美丽的海螺、水晶球——你是多么纯净啊！我将飞过你的天际大海去寻找智慧和启示...
  - RLHF (100): 《我的未来不是梦》


---

主要参考：
- https://huggingface.co/blog/stackllama
- https://github.com/lvwerra/trl/tree/main/examples/stack_llama/scripts
- 一些疑问：https://github.com/lvwerra/trl/issues/492


Reward Modeling:
```
# train_rm.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python reward_modeling.py \
    --model_name baichuan-inc/baichuan-7B \
    --lora_target_models W_pack \
    --num_train_epochs 2 \
    --eval_steps 200 \
    --save_steps 50 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 16 \
    --train_subset -1 \
    --eval_subset -1 \
    --max_length 1000
```

```
100%|███████████████████████████████████████████████████████████| 3972/3972 [7:06:28<00:00,  6.44s/it]
Saving last checkpoint of the model
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:                  eval/accuracy ▁▄▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇███████████████████████
wandb:                      eval/loss █▆▅▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                   eval/runtime ▂▁▂▁▂▂▂▃▂▂▃▃▁▇▇█▇▇▇▇▇█▇▇▇▇▇▇█▇██▇▇████▇█
wandb:        eval/samples_per_second ▇█▇█▇▇▇▆▆▇▆▆█▂▂▁▂▂▂▁▂▁▂▂▂▂▂▂▁▂▁▁▁▂▁▁▁▁▂▁
wandb:          eval/steps_per_second ███████████▆█▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    train/epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:              train/global_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            train/learning_rate ███▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                     train/loss █▇▆▃▅▇▃▅▅▆▆▄▅▆▃▅▃▂▄▂▂▃▄▃▃▅▁▂▄▅▃▅▆▅▃▄▄▅▃▄
wandb:               train/total_flos ▁
wandb:               train/train_loss ▁
wandb:            train/train_runtime ▁
wandb: train/train_samples_per_second ▁
wandb:   train/train_steps_per_second ▁
wandb: 
wandb: Run summary:
wandb:                  eval/accuracy 0.71892
wandb:                      eval/loss 0.54986
wandb:                   eval/runtime 281.5312
wandb:        eval/samples_per_second 17.742
wandb:          eval/steps_per_second 1.112
wandb:                    train/epoch 2.0
wandb:              train/global_step 3972
wandb:            train/learning_rate 0.0
wandb:                     train/loss 0.5684
wandb:               train/total_flos 0.0
wandb:               train/train_loss 0.56024
wandb:            train/train_runtime 25595.2124
wandb: train/train_samples_per_second 1.552
wandb:   train/train_steps_per_second 0.155
```

