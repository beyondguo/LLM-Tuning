from collections import defaultdict
from datasets import load_dataset,load_from_disk
import json

# 采用 HC3(Human-ChatGPT Comparison Corpus) 数据集（https://github.com/Hello-SimpleAI/chatgpt-comparison-detection）
# https://huggingface.co/Hello-SimpleAI
dataset = load_dataset('Hello-SimpleAI/HC3-Chinese','all')

def fetch_qa_pairs(examples):
    """
    从HC3数据集中，取出每一对问答
    """
    questions = examples['question']
    # human_answers = examples['human_answers']
    chatgpt_answers = examples['chatgpt_answers'] # 每一项都可能是一个list
    res = defaultdict(list)
    for q,a_list in zip(questions,chatgpt_answers):
        # passage:
        for a in a_list:
            res['question'].append(q)
            res['answer'].append(a)
    return res

# Notice: 得加上 remove_columns，因为你实际上改变了行数，如果还保留原来的会出现行数不一致
# 出现类似 pyarrow.lib.ArrowInvalid: Column 1 named question expected length 50 but got length 73 的报错
qa_dataset = dataset['train'].map(fetch_qa_pairs, batched=True,batch_size=50,
                                  remove_columns=dataset['train'].column_names)

print(qa_dataset)
# 保存到本地
qa_dataset.save_to_disk(f'data/hc3_chatgpt_qa_all')


# 加载本地数据集，并转化成json格式
hc3 = load_from_disk('data/hc3_chatgpt_qa_all')
with open('data/hc3_chatgpt_zh_specific_qa.json','w',encoding='utf8') as f:
    for i in range(len(hc3)):
        # line = {'q':hc3[i]['question'], 'a':hc3[i]['answer']}
        line = {'q':'问：'+hc3[i]['question'], 'a':'答：'+hc3[i]['answer']}
        line = json.dumps(line, ensure_ascii=False)
        f.write(line)
        f.write('\n')