"""
Copied then Modified from https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction/blob/main/1_NER/ner_report_metric.py

Acknowledgment to repo: https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction
Paper: Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors (https://arxiv.org/pdf/2305.14450.pdf)

"""

import os
import sys
import json
import ast
from difflib import SequenceMatcher


def has_duplicate(tmp_list):
    """ has duplicate ?
    """
    if tmp_list == []:
        return False
    
    if type(tmp_list[0]) == str:
        if len(tmp_list) == len(set(tmp_list)):
            return False
        else:
            return True
        
    if type(tmp_list[0]) == list:
        tmp = []
        for t in tmp_list:
            if t not in tmp:
                tmp.append(t)
        if len(tmp_list) == len(tmp):
            return False
        else:
            return True


def get_correct_list_from_response_list(target_list, response_list):
    """
    target_list 和 response_list 均有可能包含重复的 item
    当有重复元素的时候，返回最少重复的数量。
    比方target里面有2重复，response里面4个重复，那结果里面就含2重复
       target里面有8重复，response里面4个重复，那结果里面就含4重复
    """
        
    res = []
    if not has_duplicate(response_list):
        res = [item for item in response_list if item in target_list]
    else:
        if not has_duplicate(target_list):
            # 去重
            uni_response_list = []
            for item in response_list:
                if item not in uni_response_list:
                    uni_response_list.append(item)
            res = [item for item in uni_response_list if item in target_list]
        else:
            res = []
            processed_item_list = []
            for item in response_list:
                if item not in processed_item_list:
                    processed_item_list.append(item)

                    num_item = response_list.count(item)
                    if num_item == 1:  # not duplicate
                        if item in target_list:
                            res.append(item)
                    else:  # duplicate
                        if item in target_list:
                            num_item_in_target = target_list.count(item)
                            num_item_correct = min([num_item, num_item_in_target])
                            res += [item] * num_item_correct
    return res

def calculate_metrics(tp, fp, fn):
    p, r, f1 = 0.0, 0.0, 0.0
    if tp + fp != 0:
        p = 1.0 * tp / (tp + fp)
    if tp + fn != 0:
        r = 1.0 * tp / (tp + fn)
    if p + r != 0.0:
        f1 = 2.0 * p * r / (p + r)
    return {
        'f1': round(f1, 5),
        'precision': round(p, 5),
        'recall': round(r, 5),
        'tp':tp,
        'fp':fp,
        'fn':fn,
        'tp+fn':tp+fn
    }



# 字符串硬匹配  编辑距离相似度软匹配
def modify_to_target_by_edit_distance(predict, target_list, threshold=0.5):
    """
    用途：将一些预测出的一些很相似的实体，认为是正确的。
    为了统计的方法，这里把这些相似的实体，直接用ground truth的实体代替，从而进行硬匹配。
    """
    pred = predict.strip()
    if len(target_list) == 0:
        return pred
    similarity_list = [SequenceMatcher(a=pred, b=item).ratio() for item in target_list]
    max_score = max(similarity_list)
    if max_score > threshold:
        max_index = similarity_list.index(max_score)
        target_item = target_list[max_index].lower().strip()
        if target_item != pred and (target_item in pred or pred in target_item):  # 允许 小幅度 span 起始位置不对
            return target_item

    return pred


## 解析 response
def response_string_to_list(response):
    """return 
        1) string 列表
        2) list  列表
    """
    def get_list_by_string(list_str):
        try:
            res_list = ast.literal_eval(list_str) 
        except:
            res_list = []
        finally:
            return res_list
    
    # response = response.lower()
    response = response.replace("(", "[").replace(")", "]")
    num_left = response.count("[")

    res_list = []

    if num_left == 0:
        return res_list
    
    if num_left == 1:
        start_idx = response.find('[')
        response = response[start_idx:]
        num_right = response.count("]")
        if num_right < 1:
            return res_list
        else:
            start_idx = response.find('[')
            end_idx = response.find(']')
            span = response[start_idx: end_idx+1]
            res_list = get_list_by_string(span)
            res_list = [str(res).strip() for res in res_list] 
            return res_list

    # "['a', 'b'], ['c', 'd']"
    start_idx = -1
    end_idx = -1

    for i, ch in enumerate(response):
        if ch == '[':
            start_idx = i
        if ch == ']':
            end_idx = i
        # print(start_idx, end_idx)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            span = response[start_idx: end_idx+1]
            tmp_list = get_list_by_string(span)
            tmp_list = [str(res).strip() for res in tmp_list] 
            res_list.append(tmp_list)
            start_idx = -1
            end_idx = -1

    return res_list


def get_result_list(response):
    result_list = []
    lines = response.split("\n")
    for line in lines:
        tmp_res_list = response_string_to_list(line.strip())
        if tmp_res_list != [] and type(tmp_res_list[0]) == str:  # [, ]\n[, ]
            if len(tmp_res_list) == 2:
                entity = dict()
                entity["e_name"] = tmp_res_list[1].strip()
                entity["e_type"] = tmp_res_list[0].strip()
                result_list.append(entity)
            if len(tmp_res_list) > 2:  # [LOC, a, b, ...]
                cur_e_type = tmp_res_list[0]
                for i_idx in range(1, len(tmp_res_list)):
                    entity = dict()
                    entity["e_name"] = tmp_res_list[i_idx].strip()
                    entity["e_type"] = cur_e_type
                    result_list.append(entity)

        if tmp_res_list != [] and type(tmp_res_list[0]) == list:  # [, ], [, ]
            for tmp in tmp_res_list:
                if len(tmp) == 2:
                    entity = dict()
                    entity["e_name"] = tmp[1].strip()
                    entity["e_type"] = tmp[0].strip()
                    result_list.append(entity)
                if len(tmp) > 2:  # [LOC, a, b, ...]
                    cur_e_type = tmp[0]
                    for i_idx in range(1, len(tmp)):
                        entity = dict()
                        entity["e_name"] = tmp[i_idx].strip()
                        entity["e_type"] = cur_e_type
                        result_list.append(entity)

    return result_list

"""
result file指定格式：
json文件，最外层是一个[],然后每一项，代表一个样本，表示如下：
{
    "mode": "flat",
    "seq": "Platelet-activating factor is a potent mediator of the inflammatory response .",
    "entities": [
        {
            "e_name": "Platelet-activating factor",
            "e_type": "protein",
            "e_type_verbose": "protein",
            "start": 0,
            "end": 2
        }
    ],
    "NER": [
        {
            "e_name": "Platelet-activating factor",
            "e_type": "protein"
        }
    ],
    "prompt": "Considering 5 types of named entities including \"DNA\", \"RNA\", \"cell_type\", \"protein\" and \"cell_line\", recognize all named entities in the given sentence.\nAnswer in the format [\"entity_type\", \"entity_name\"] without any explanation. If no entity exists, then just answer \"[]\".\nGiven sentence: \"Platelet-activating factor is a potent mediator of the inflammatory response .\"",
    "response": "[\"protein\", \"Platelet-activating factor\"]"
}
其中：
"entities": 标注的实体信息，即label
"NER": 模型输出的解析结果，即prediction，跟上面的label来对比从而进行评价
"prompt": 模型输入
"response":模型输出，要把这个结果进行规范化，然后放入"NER"字段中
"""

types = {
            "entities": {
                "DNA": {
                    "verbose": "DNA",
                    "short": "DNA"
                },
                "RNA": {
                    "verbose": "RNA",
                    "short": "RNA"
                },
                "cell_type": {
                    "verbose": "cell_type",
                    "short": "cell_type"
                },
                "protein": {
                    "verbose": "protein",
                    "short": "protein"
                },
                "cell_line": {
                    "verbose": "cell_line",
                    "short": "cell_line"
                }
            },
            "relations": {}
        }

## report overall metric
def report_metric(data):
    """
    result file: 包含了target，prediction等内容的文件，官方指定格式
    type file: 实体类型文件
    """
    e_types = types["entities"]
    e_types_list = [e_types[key]["verbose"].lower() for key in e_types]

    ## per type
    hard_boundaries = dict()
    soft_boundaries = dict()
    for key in e_types_list:
        hard_boundaries[key] = {"tp": 0, "fp": 0, "fn": 0}
        soft_boundaries[key] = {"tp": 0, "fp": 0, "fn": 0}
    
    ## statistics
    num_undefined_type = 0
    num_entity = 0
    tp_ner_boundaries = 0
    fp_ner_boundaries = 0
    fn_ner_boundaries = 0
    tp_ner_strict = 0
    fp_ner_strict = 0
    fn_ner_strict = 0

    tp_ner_boundaries_soft_match = 0
    fp_ner_boundaries_soft_match = 0
    fn_ner_boundaries_soft_match = 0
    tp_ner_strict_soft_match = 0
    fp_ner_strict_soft_match = 0
    fn_ner_strict_soft_match = 0

    num_invalid = 0
    
    for example in data:
        
        ## target
        strict_target_list = []
        boundaries_target_list = []

        ## per type
        boundaries_target_list_dict = {}
        for key in e_types_list:
            boundaries_target_list_dict[key] = []

        for ent in example["entities"]:
            ent_name = ent["e_name"].lower()
            ent_type = e_types[ent["e_type"]]["verbose"].lower()  # 全写 

            strict_target_list.append([ent_type, ent_name])  
            boundaries_target_list.append(ent_name)

            ## per type
            boundaries_target_list_dict[ent_type].append(ent_name)
            
            num_entity += 1

        ## predict
        strict_predict_list = []
        boundaries_predict_list = []
        strict_predict_list_soft_match = []
        boundaries_predict_list_soft_match = []

        # per type
        boundaries_predict_list_dict = {}
        boundaries_predict_list_soft_match_dict = {}
        for key in e_types_list:
            boundaries_predict_list_dict[key] = []
            boundaries_predict_list_soft_match_dict[key] = []

        # response = example["prediction"]
        
        res_flag = True
        # print(response)
        # if response.strip().strip('"').strip() != '[]' and example["NER"] == []:
        #     res_flag = False

        for ent in example["NER"]:
            # print(ent)
            ent_name = ent["e_name"].lower()
            ent_type = ent["e_type"].lower() 
            strict_predict_list.append([ent_type, ent_name])
            boundaries_predict_list.append(ent_name)

            # per type
            if ent_type not in e_types_list:
                num_undefined_type += 1
                res_flag = False
            else:
                boundaries_predict_list_dict[ent_type].append(ent_name)
                

            ## soft match
            ent_name = modify_to_target_by_edit_distance(ent_name, boundaries_target_list, threshold=0.5)
            strict_predict_list_soft_match.append([ent_type, ent_name])
            boundaries_predict_list_soft_match.append(ent_name)

            # per type
            if ent_type in e_types_list:
                boundaries_predict_list_soft_match_dict[ent_type].append(ent_name)

        if not res_flag:  # res_flag -------- 记录是否解析成功，格式正确
            num_invalid += 1
            # print("#", response)
        
        ## hard-match 
        strict_correct_list = get_correct_list_from_response_list(strict_target_list, strict_predict_list)
        boundaries_correct_list = get_correct_list_from_response_list(boundaries_target_list, boundaries_predict_list)
        """
        TP: 正确的被识别为实体的数量
        FP：识别出来的实体，但错了的数量
        FN：本来是实体，但没有被识别出来
        （可以理解为positive就是被检测过来的实体，然后前面的True/False就是说这个识别结果对不对）
        """
        tp_ner_strict += len(strict_correct_list)
        fp_ner_strict += len(strict_predict_list) - len(strict_correct_list)
        fn_ner_strict += len(strict_target_list) - len(strict_correct_list)

        tp_ner_boundaries += len(boundaries_correct_list)
        fp_ner_boundaries += len(boundaries_predict_list) - len(boundaries_correct_list)
        fn_ner_boundaries += len(boundaries_target_list) - len(boundaries_correct_list)

        ## soft-match
        strict_correct_list_soft_match = get_correct_list_from_response_list(strict_target_list, strict_predict_list_soft_match)
        boundaries_correct_list_soft_match = get_correct_list_from_response_list(boundaries_target_list, boundaries_predict_list_soft_match)
        
        tp_ner_strict_soft_match += len(strict_correct_list_soft_match)
        fp_ner_strict_soft_match += len(strict_predict_list_soft_match) - len(strict_correct_list_soft_match)
        fn_ner_strict_soft_match += len(strict_target_list) - len(strict_correct_list_soft_match)

        tp_ner_boundaries_soft_match += len(boundaries_correct_list_soft_match)
        fp_ner_boundaries_soft_match += len(boundaries_predict_list_soft_match) - len(boundaries_correct_list_soft_match)
        fn_ner_boundaries_soft_match += len(boundaries_target_list) - len(boundaries_correct_list_soft_match)

        ## per type
        for key in e_types_list:
            cur_correct = get_correct_list_from_response_list(boundaries_target_list_dict[key], boundaries_predict_list_dict[key])
            hard_boundaries[key]["tp"] += len(cur_correct)
            hard_boundaries[key]["fp"] += len(boundaries_predict_list_dict[key]) - len(cur_correct)
            hard_boundaries[key]["fn"] += len(boundaries_target_list_dict[key]) - len(cur_correct)

            cur_correct_soft = get_correct_list_from_response_list(boundaries_target_list_dict[key], boundaries_predict_list_soft_match_dict[key])
            soft_boundaries[key]["tp"] += len(cur_correct_soft)
            soft_boundaries[key]["fp"] += len(boundaries_predict_list_soft_match_dict[key]) - len(cur_correct_soft)
            soft_boundaries[key]["fn"] += len(boundaries_target_list_dict[key]) - len(cur_correct_soft)

    hard_res = calculate_metrics(tp_ner_strict, fp_ner_strict, fn_ner_strict)
    soft_res = calculate_metrics(tp_ner_strict_soft_match, fp_ner_strict_soft_match, fn_ner_strict_soft_match)
    return hard_res, soft_res



# ---------------------------
N_train_val = 4500
desc_num=0
# default design
# 目前这里的解析结果跟官方的结果一致，测试集总实体数相同
print('======== Default design ========')
ner_prediction_file = f'data/eval_ner/default_desc{desc_num}_{N_train_val}_llama2_prediction.json'
data = []
with open(ner_prediction_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        example = json.loads(line)
        
        # target
        entities = []
        res_text = example['target']
        if res_text != "":
            item_text_list = res_text.split('\n')
            for item_text in item_text_list:
                type_and_name = item_text.replace('[','').replace(']','').split(',')
                e_type = type_and_name[0][1:-1]
                e_name = type_and_name[1][1:-1]
                # res = ast.literal_eval(item_text)
                entities.append({'e_type':e_type,'e_name':e_name})
        example['entities'] = entities
        
        # prediction
        predicted_entities = []
        res_text = example['prediction']
        if res_text != "":
            item_text_list = res_text.split('\n')
            for item_text in item_text_list:
                try:
                    type_and_name = item_text.replace('[','').replace(']','').split(',')
                    e_type = type_and_name[0][1:-1]
                    e_name = type_and_name[1][1:-1]
                    predicted_entities.append({'e_type':e_type,'e_name':e_name})
                except:
                    # print("Wrong format!",item_text)
                    pass
        example['NER'] = predicted_entities   
        
        data.append(example)

hard_res, soft_res = report_metric(data)
print('hard match res:')
print(hard_res)
print('soft match res:')
print(soft_res)
print()


# weak design
print('======== Weak1 design ========')
ner_prediction_file = f'data/eval_ner/weak1_desc{desc_num}_{N_train_val}_llama2_prediction.json'
data = []
with open(ner_prediction_file,'r') as f:
    lines = f.readlines()
    
    for line in lines:  # each line is an example
        example = json.loads(line)
        # 1. 将target字段处理成符合格式的entities字段
        entities = []   # ----- target entities, dict
        if example['target'] != "": # target字段为空字符串的时候，说明该句子没有实体
            type_res_list = example['target'].split('; ')
            for type_res in type_res_list:
                # 隐患：实体中如果包含了引号的话，可能会被下面的处理去掉。不过只要target和prediction处理一致应该就没问题
                e_type = type_res.split(': ')[0].strip().replace('\'','')
                e_names = [name.strip().replace('\'','') for name in type_res.split(': ')[1].strip().split(',')]
                if e_names:
                    for e_name in e_names:
                        entities.append({'e_type':e_type,'e_name':e_name})
        example['entities'] = entities
        # data.append(example)
        
        # 2. 将prediction字段处理成同上的格式
        predicted_entities = []
        if example['prediction'] != "":
            type_res_list = example['prediction'].split('; ')
            for type_res in type_res_list:
                if ":" not in type_res: # 产生了非 'e_type': 'name1', 'name2',... 这样的片段，这个片段就去掉，其他的正常的识别片段任然保留
                    continue
                e_type = type_res.split(':')[0].strip().replace('\'','')
                e_names = [name.strip().replace('\'','') for name in type_res.split(':')[1].strip().split(',')]
                if e_names:
                    for e_name in e_names:
                        predicted_entities.append({'e_type':e_type,'e_name':e_name})
        example['NER'] = predicted_entities
        
        data.append(example)
        
hard_res, soft_res = report_metric(data)
print('hard match res:')
print(hard_res)
print('soft match res:')
print(soft_res)
print()

        

# good1 design
print('======== Good1 design ========')
ner_prediction_file = f'data/eval_ner/good1_desc{desc_num}_{N_train_val}_llama2_prediction.json'
data = []
with open(ner_prediction_file,'r') as f:
    lines = f.readlines()
    
    for line in lines:  # each line is an example
        example = json.loads(line)
        # 1. 将target字段处理成符合格式的entities字段
        entities = []   # ----- target entities, dict
        type_res_list = example['target'].split('\n')
        for type_res in type_res_list:
            e_type = type_res.split(': ')[0].strip().replace('\'','')
            e_names = [name.strip().replace('\'','') for name in type_res.split(': ')[1].strip().split(',')]
            if e_names != ['']:
                for e_name in e_names:
                    entities.append({'e_type':e_type,'e_name':e_name})
        example['entities'] = entities

        
        # 2. 将prediction字段处理成同上的格式
        predicted_entities = []
        type_res_list = example['prediction'].split('\n')
        for type_res in type_res_list:
            if ":" not in type_res: # 产生了非 'e_type': 'name1', 'name2',... 这样的片段，这个片段就去掉，其他的正常的识别片段任然保留
                continue
            e_type = type_res.split(':')[0].strip().replace('\'','')
            e_names = [name.strip().replace('\'','') for name in type_res.split(':')[1].strip().split(',')]
            if e_names != ['']:
                for e_name in e_names:
                    predicted_entities.append({'e_type':e_type,'e_name':e_name})
        example['NER'] = predicted_entities
        
        data.append(example)

hard_res, soft_res = report_metric(data)
print('hard match res:')
print(hard_res)
print('soft match res:')
print(soft_res)
print()



# # good2 design
# print('======== Good2 design ========')
# ner_prediction_file = f'data/eval_ner/good2_{N_train_val}_llama2_prediction.json'
# data = []
# with open(ner_prediction_file,'r') as f:
#     lines = f.readlines()
    
#     for line in lines:  # each line is an example
#         example = json.loads(line)
#         # 1. 将target字段处理成符合格式的entities字段
#         entities = []   # ----- target entities, dict
#         type_res_list = example['target'].split('\n')
#         for type_res in type_res_list:
#             d = json.loads(type_res)
#             if d['entity_names'] != []:
#                 for e_name in d['entity_names']:
#                     entities.append({'e_type':d['entity_type'],'e_name':e_name})
#         example['entities'] = entities
        
#         # 2. 将prediction字段处理成同上的格式
#         predicted_entities = []
#         type_res_list = example['prediction'].split('\n')
#         for type_res in type_res_list:
#             try:
#                 d = json.loads(type_res)
#                 if d['entity_names'] != []:
#                     for e_name in d['entity_names']:
#                         predicted_entities.append({'e_type':d['entity_type'],'e_name':e_name})
#             except:
#                 # print('Wrong format.', type_res)
#                 pass
#         example['NER'] = predicted_entities
        
#         data.append(example)

# hard_res, soft_res = report_metric(data)
# print('hard match res:')
# print(hard_res)
# print('soft match res:')
# print(soft_res)
# print()



# for seed in [1,2,3,4,5]:
#     ner_prediction_file = f'data/eval_ner/good1_500_{seed}_llama2_prediction.json'
#     data = []
#     with open(ner_prediction_file,'r') as f:
#         lines = f.readlines()
        
#         for line in lines:  # each line is an example
#             example = json.loads(line)
#             # 1. 将target字段处理成符合格式的entities字段
#             entities = []   # ----- target entities, dict
#             type_res_list = example['target'].split('\n')
#             for type_res in type_res_list:
#                 e_type = type_res.split(': ')[0].strip().replace('\'','')
#                 e_names = [name.strip().replace('\'','') for name in type_res.split(': ')[1].strip().split(',')]
#                 if e_names != ['']:
#                     for e_name in e_names:
#                         entities.append({'e_type':e_type,'e_name':e_name})
#             example['entities'] = entities

            
#             # 2. 将prediction字段处理成同上的格式
#             predicted_entities = []
#             type_res_list = example['prediction'].split('\n')
#             for type_res in type_res_list:
#                 if ":" not in type_res: # 产生了非 'e_type': 'name1', 'name2',... 这样的片段，这个片段就去掉，其他的正常的识别片段任然保留
#                     continue
#                 e_type = type_res.split(':')[0].strip().replace('\'','')
#                 e_names = [name.strip().replace('\'','') for name in type_res.split(':')[1].strip().split(',')]
#                 if e_names != ['']:
#                     for e_name in e_names:
#                         predicted_entities.append({'e_type':e_type,'e_name':e_name})
#             example['NER'] = predicted_entities
            
#             data.append(example)

#     hard_res, soft_res = report_metric(data)
#     print('hard match res:')
#     print(hard_res)
#     print('soft match res:')
#     print(soft_res)
#     print()
    
    