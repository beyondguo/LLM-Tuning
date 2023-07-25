# CUDA_VISIBLE_DEVICES=2 streamlit run toolkit.py --server.port 6007

import streamlit as st
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from streamlit_chat import message as stream_message
from peft import PeftModel
import json
import time


st.set_page_config(
    page_title="大模型工具箱",
    page_icon=":robot:"
)

"# SUFE AI Lab —— 大模型工具箱"

def a_progress():
    'Starting a long computation...'
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(5):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress((i+1)*20)
        time.sleep(1)
        
# ================================= base-model =================================
base_model_list = {"chatglm-6b": "THUDM/chatglm-6b",
                    "chatglm2-6b": "THUDM/chatglm2-6b",
                    "baichuan-7b": "baichuan-inc/baichuan-7B"}

# ================================= tools =================================
lora_path = {
    # "ChatGLM|抽取式问答": "./resource/fin_qa",
    "chatglm2-6b|公司情感抽取": "../resource/sentiment_comp_ie_chatglm2",
    "baichuan-7b|公司情感抽取": "../resource/sentiment_comp_ie_shuffled_baichuan-7B",
    "baichuan-7b|聊天": "../resource/hc3_chatgpt_zh_specific_qa_baichuan-7B"
    }

@st.cache_resource
def get_model(base_model_name):
    # loading base model
    st.sidebar.text('加载基座模型...')
    if base_model_name == 'baichuan-7b':
        model = AutoModelForCausalLM.from_pretrained(base_model_list[base_model_name], 
                                                     trust_remote_code=True, device_map='auto')
    elif base_model_name in ['chatglm-6b','chatglm2-6b']:
        model = AutoModel.from_pretrained(base_model_list[base_model_name], 
                                                     trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(base_model_list[base_model_name], 
                                                     trust_remote_code=True)
    st.sidebar.text('基座模型加载成功!')
    
    # loading tools
    available_tools_paths = list({k:v for k,v in lora_path.items() if base_model_name in k}.keys())
    print(available_tools_paths)
    available_tools = []
    # 直接一口气把所有 tool 都加载好
    init_tool_path = available_tools_paths[0]
    tool_name = init_tool_path.split('|')[-1]
    available_tools.append(tool_name)
    model = PeftModel.from_pretrained(model, lora_path[init_tool_path], adapter_name=tool_name)
    st.sidebar.text(f'已加载工具：{tool_name}')
    for tool_path in available_tools_paths[1:]:
        tool_name = tool_path.split('|')[-1]
        available_tools.append(tool_name)
        model.load_adapter(lora_path[tool_path],adapter_name=tool_name)
        st.sidebar.text(f'已加载工具：{tool_name}')
    return model, tokenizer, available_tools



current_base_model = st.sidebar.selectbox(
    label="选择基座大模型 (`base-model`)",
    options=[
        "baichuan-7b",
        "chatglm2-6b",
    ]
)
st.write(f"当前使用的是 **{current_base_model}**")

model, tokenizer, available_tools = get_model(current_base_model)
print(model.peft_config)

current_tool = st.radio(
    label="选择具体的工具 (`tool`)",
    options=available_tools
)

def get_prompt(text, base_model_name, tool_name):
    if base_model_name == 'baichuan-7b':
        if tool_name == '聊天':
            prompt = "问："+text+"答："
        if tool_name == '公司情感抽取':
            prompt = text + """\n---\n请从上文中抽取出所有公司，以及对应的在本文中的情感倾向（积极、消极、中性）以及原因。
请用这样的格式返回：\n{"ORG":..., "sentiment":..., "reason":...}"""
    if 'glm' in base_model_name:
        if tool_name == '公司情感抽取':
            prompt = text + """\n---\n请从上文中抽取出所有公司，以及对应的在本文中的情感倾向（积极、消极、中性）以及原因。
请用这样的格式返回：\n{"ORG":..., "sentiment":..., "reason":...}"""
    return prompt

# @st.cache_data
def generate_with_tool(text, tool_name):
    streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    prompt = get_prompt(text, current_base_model, tool_name)
    print('输入：\n',prompt)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to('cuda')
    
    # setting tool
    model.set_adapter(tool_name)
    print('当前adapter：', model.active_adapter)
    output = model.generate(**inputs, max_new_tokens=128,
                            # do_sample=True,
                            repetition_penalty=1.1, 
                            begin_suppress_tokens=[tokenizer.eos_token_id],
                            streamer=streamer
                            )
    output = output[0][inputs.input_ids.shape[1]:] # 这样可以防止输出prompt部分
    return tokenizer.decode(output,skip_special_tokens=True)

# generate_with_tool(
#     """ChatGPT的提出对谷嘎、万度的搜索业务产生巨大打击，传统搜索引擎的作用性降低了。
# 与此同时，OChat，Linguo等新兴语义搜索公司，迅速推出自己的类ChatGPT模型，并结合进自家搜索引擎，受到了很多用户的青睐。
# 腾势、艾里等公司表示会迅速跟进ChatGPT和AIGC的发展，并预计在年底前推出自己的大模型。
# 大型图片供应商视觉中国称ChatGPT对公司业务暂无影响，还在观望状态。""",
#     tool_name=current_tool)


container = st.container()
"---"
# input_text = st.text_area(label="**您的输入**",
#             height = 100,
#             placeholder="请在这儿进行您的输入")


input_text = st.chat_input("您的输入")
if input_text:
    output = generate_with_tool(input_text, current_tool)
    user_message = st.chat_message(name='user')
    model_message = st.chat_message(name='assistant')
    user_message.write(input_text)
    model_message.write(output)
