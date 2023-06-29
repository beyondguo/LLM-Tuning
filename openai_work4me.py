import os
import openai
openai.organization = "your_org"
openai.api_key = "your_api_key"
# openai.Model.list()

import pickle
with open('your_data.pkl','rb') as f:
    news = pickle.loads(f.read())


instruction = """\n---
请从上文中抽取出所有公司/机构、对应的在本文中的情感倾向（积极、消极、中性）以及原因。
并用这样的格式返回：
{"ORG":..., "sentiment":..., "reason":...}"""


import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def get_num_tokens(text):
    return len(encoding.encode(text))


contents = [t + instruction for t in news]



def get_openai_res(content):
    try:
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": content}
          ]
        )
        msg = completion.choices[0].message['content']
    except:
        msg = ''
    return [content,msg]


import concurrent.futures
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(get_openai_res, content) for content in contents}

    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        # 这里，每当有任务完成，就会打印一次进度
        print(f"Processed {i}/{len(contents)} contents.")
        if i % 50 == 0:
            with open('sentiment_comp_qaie_pairs.pkl','wb') as f:
                pickle.dump(results,f)
                print('saved',i)
            
len(results)
with open('sentiment_comp_qaie_pairs.pkl','wb') as f:
    pickle.dump(results,f)