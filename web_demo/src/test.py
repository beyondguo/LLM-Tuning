from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from peft import PeftModel
import streamlit as st
import json
from streamlit_chat import message


st.set_page_config(
    page_title="大模型工具箱",
    page_icon=":robot:"
)

model_list = {"ChatGLM": "THUDM/chatglm-6b",
              "ChatGLM2": "THUDM/chatglm2-6b",
              "BaiChuan": "baichuan-inc/baichuan-7B"}

#TODO 目前维护一个{‘模型名|功能名’：‘lora_path’}的dict
lora_list = {"ChatGLM|抽取式问答": "./resource/fin_qa",
             "ChatGLM2|公司情感抽取": "./resource/sentiment_comp_ie_chatglm2",
             "BaiChuan|公司情感抽取": "./resource/sentiment_comp_ie_shuffled_baichuan-7B",
             "BaiChuan|聊天": "./resource/hc3_chatgpt_zh_specific_qa_baichuan-7B"}

@st.cache_resource
def get_prompt():
    with open("./resource/instruction.json", 'r', encoding='utf-8') as file:
        instruction = json.load(file)
    return instruction

@st.cache_resource
def get_model(model_name, lora_name):
    with st.spinner("模型加载中........"):
        model_path = model_list[model_name]
        lora_path = lora_list[model_name+'|'+lora_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if model_name == "BaiChuan":
            model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="auto", trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        if not isinstance(model,PeftModel): # 还没加载过lora
            print('first time loading lora...')
            model = PeftModel.from_pretrained(model, lora_path, adapter_name=lora_name).half()
        elif lora_name in model.peft_config: # 已经加载过，就直接设置
            print('reloading lora...')
            model.set_adapter(lora_name)
        else: # 尚未加载过，就先加载，再设置
            print('setting new lora...')
            model.load_adapter(lora_path,adapter_name=lora_name)
            model.set_adapter(lora_name)
        model = model.eval()
    return tokenizer, model

#TODO 目前改变LoRA也需要重新加载整个模型，尝试寻找卸载LoRA的方法
# @st.cache_resource
# def get_lora(_model, model_name, lora_name):
#     lora_path = lora_list[model_name+lora_name]
#     model = PeftModel.from_pretrained(_model, lora_path).half()
#     model = model.eval()
#     print("changing lora...")
#     return model

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def glm_chat(tokenizer, model, input, history):
    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history):
                query, response = history[-1]
                st.write(response)
    return history

def baichuan_chat(tokenizer, model, input):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(input, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    output = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.1, streamer=streamer)
    response = tokenizer.decode(output.cpu()[0], skip_special_tokens=True)
    st.write(response)

def predict(input, model_name, lora_name, history=None):
    tokenizer, model = get_model(model_name, lora_name)
    prompt = instruction[lora_name]
    print(prompt)
    model_input = prompt.format(input)
    print(model_input)
    # model = get_lora(model, model_name, lora_name)
    if history is None:
        history = []

    if model_name == "BaiChuan":
        baichuan_chat(tokenizer, model, model_input)
    else:
        history = glm_chat(tokenizer, model, model_input, history)
    
    return history


container = st.container()

# load instruction dict
instruction = get_prompt()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

select_model_name = st.sidebar.selectbox('选择模型：', options=['ChatGLM', 'ChatGLM2', 'BaiChuan'])

if select_model_name == 'ChatGLM':
    select_lora_name = st.sidebar.selectbox('选择任务工具：', options=['抽取式问答'])
elif select_model_name == 'ChatGLM2':
    select_lora_name = st.sidebar.selectbox('选择任务工具：', options=['公司情感抽取'])
else:
    select_lora_name = st.sidebar.selectbox('选择任务工具：', options=['公司情感抽取', '聊天'])


# max_length = st.sidebar.slider(
#     'max_length', 0, 4096, 2048, step=1
# )
# top_p = st.sidebar.slider(
#     'top_p', 0.0, 1.0, 0.6, step=0.01
# )
# temperature = st.sidebar.slider(
#     'temperature', 0.0, 1.0, 0.95, step=0.01
# )

if 'state' not in st.session_state:
    st.session_state['state'] = []
    

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, select_model_name, select_lora_name, st.session_state["state"])