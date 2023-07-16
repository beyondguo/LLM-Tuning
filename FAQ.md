# 常见问题 (frequently asked questions):

- **如何构造符合`tokenize.sh`格式要求的 json 文件 (How to consturct the input json file for tokenization):**

下面给出了一个示范脚本：
```python
import pandas as pd
import json

df = pd.read_csv('data.csv')
df.columns 
# ["question","answer"]

questions = df['question'].tolist()
answers = df['answer'].tolist()
with open('raw_data.json','w',encoding='utf8') as f:
    for q, a in zip(questions,answers):
        d = {'q':q, 'a':a}
        line = json.dumps(d, ensure_ascii=False)
        f.write(line)
        f.write('\n')
```


