import json

with open('aspect_sentiment.json', 'r', encoding='utf8') as f:
    lines = f.readlines()

old_instruction = """你现在是一个细粒度情感分析模型，请从评论中，抽取出关于以下所有方面的情感倾向[\"正向：1\",\"中性:0\",\"负向:-1\",\"未提及:-2\"],评论："""
int2text = {
    -2:"未提及",
    -1:"负面",
    0:"中性",
    1:"正面",
}

aspects_list = [
    "交通是否便利",
    "距离商圈远近",
    "是否容易寻找",
    "排队等候时间",
    "服务人员态度",
    "是否容易停车",
    "点菜/上菜速度",
    "价格水平",
    "性价比",
    "折扣力度",
    "装修情况",
    "嘈杂情况",
    "就餐空间",
    "卫生情况",
    "分量",
    "口感",
    "外观",
    "推荐程度",
    "本次消费感受",
    "再次消费的意愿"
]

new_instruction = f"\n---\n请对上面这段文字，进行细粒度方面情感分析，具体包括以下这些方面：{','.join(aspects_list)}，情感分为四类：正面、负面、中性、未提及。请用json的格式返回结果。"

with open('aspect_sentiment_better.json','w') as f:
    for line in lines:
        d = json.loads(line)
        content = d['content']
        # ------- 重构 prompt -------
        # 获取原始的待分析的句子
        text = content.replace(old_instruction,'')
        text = text[1:-1] # 去除前后的 " 符号
        prompt = text + new_instruction
        
        # ------- 重构 output -------
        summary = d['summary']
        summary_dict = {item.split(':')[0]: int(item.split(':')[1]) for item in summary.split(',')}
        summary_dict = {k:int2text[v] for k,v in summary_dict.items()}
        output = json.dumps(summary_dict, ensure_ascii=False)
        
        better_line = {
            'prompt':prompt,
            'output':output
        }
        better_line = json.dumps(better_line, ensure_ascii=False)
        f.write(better_line)
        f.write('\n')
    
    
    
    
    