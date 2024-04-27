import os
import json
import ast
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

type_list = ['Catastrophe','Attack','Hostile_encounter','Causation','Process_start','Competition','Motion','Social_event','Killing','Conquering']
# num_type_dict={0: 'Catastrophe', 1: 'Attack', ...}
num_type_dict = dict(zip(range(len(type_list)), type_list))

def get_metrics(tp, fp, fn):
    p, r, f1 = 0.0, 0.0, 0.0

    if tp + fp != 0:
        p = 1.0 * tp / (tp + fp)
    if tp + fn != 0:
        r = 1.0 * tp / (tp + fn)
    if p + r != 0.0:
        f1 = 2.0 * p * r / (p + r)
        
    return p, r, f1

def convert_to_list(predict, format_type):
    # 将各种format的输出转化为[[ , ], [ , ]...]
    # convert the output of various formats to [[ , ], [ , ]...]
    format_error_flag = 0
    empty_flag = 0
    processed_list = []
    
    # empty output
    if predict == "":
        format_error_flag = 1
        empty_flag = 1
        return processed_list, format_error_flag, empty_flag
    
    if 'ou' in format_type:
        pred_list = predict.split(';')
        for pred_tuple in pred_list:
            e_items = pred_tuple.split(':')
            if len(e_items)==1:
                processed_list.append([e_items[0].strip().lower(), ""])
            else:
                e_type = e_items[0]
                e_trigger_list = e_items[1].split(',')
                for t in e_trigger_list:
                    processed_list.append([e_type.strip().lower(), t.strip().lower()])

    elif 'heuristic' in format_type:
        pred_list = predict.split('\n')
        for item in pred_list:
            try:
                pred_tuple = ast.literal_eval(item.strip())
            except:
                pred_tuple = []
            if pred_tuple != []:
                processed_list.append([target_item.strip().lower() for target_item in pred_tuple])
    
    elif 'numerical' in format_type:
        pred_list = predict.split(';')
        for pred_tuple in pred_list:
            e_items = pred_tuple.split(':')
            try:
                e_type = num_type_dict[int(e_items[0])]
            except:
                e_type = e_items
                print(predict)
                print(pred_tuple)
            e_trigger_list = e_items[1].split(',')
            for t in e_trigger_list:
                if t.strip() != 'NONE':
                    processed_list.append([e_type.strip().lower(), t.strip().lower()])
        
    elif 'lines' in format_type:
        pred_list = predict.split('\n')
        for pred_tuple in pred_list:
            e_items = pred_tuple.split(':')
            if len(e_items)==1:
                processed_list.append([e_items[0].strip().lower(), ""])
            else:
                e_type = e_items[0].strip()
                e_trigger = e_items[1].strip()
                if e_trigger == 'NONE':
                    continue
                else:
                    e_trigger_list = e_trigger.split(',')
                    for t in e_trigger_list:
                        processed_list.append([e_type.strip().lower(), t.strip().lower()])
    
    elif 'json' in format_type:
        pred_list = predict.split('\n')
        for item in pred_list:
            try:
                pred_tuple = ast.literal_eval(item.strip())
            except:
                pred_tuple = {}
                format_error_flag = 1
                continue
            # print(pred_tuple)
            e_type = pred_tuple['event_type']
            e_trigger = pred_tuple['trigger word']
            if e_trigger == []:
                continue
            else:
                if type(e_trigger) == str:
                    e_trigger = e_trigger.strip().split(',')
                    
                if type(e_trigger) == list:
                    for t in e_trigger:
                        processed_list.append([e_type.strip().lower(), t.strip().lower()])
                else:
                    print('trigger format error')
                    format_error_flag = 1

    return processed_list, format_error_flag, empty_flag


def get_correct_list(predict_list, label_list, target_trigger_list, is_soft=False, threshold=0.5):
    '''
    predict_list: LLM predict of one sample
    label_list: label list of the sample  [[event_type, event_trigger], [ , ]...]
    target_trigger_list: true trigger list
    is_soft: set True --- modify trigger word by similarity 
    '''
    # Deduplicate the prediction
    pred_list = [list(t) for t in set(tuple(i) for i in predict_list)]

    # modify trigger word by similarity
    if is_soft == True:
        for pred_item in pred_list:
            pred_t = pred_item[1]
            similarity_list = [SequenceMatcher(a=pred_t, b=item).ratio() for item in target_trigger_list]
            max_score = max(similarity_list)
            if max_score > threshold:
                max_index = similarity_list.index(max_score)
                pred_t = target_trigger_list[max_index].lower().strip()
            pred_item[1] = pred_t
            
    res = [item for item in pred_list if item in label_list]

    return res
    
if __name__=='__main__':
    model = 'llama2'
    folder = os.path.join(os.path.dirname(__file__), 'maven_eval_' + model)

    # whether to record wrong formatted samples
    is_write = False
    # whether to use soft matching on trigger words
    is_soft = True
    result_folder = os.path.join(os.path.dirname(__file__), model + '_result')
    os.makedirs(os.path.dirname(result_folder), exist_ok=True)
    format_list = ['heuristic']
    
    # get labels from heuristic_test.json
    with open(os.path.join(folder, 'heuristic_test.json'), 'r') as f:
        label_lines = f.readlines()
    test_labels = {}
    for i in range(len(label_lines)):
        target_tuple_list = []
        label = json.loads(label_lines[i])['output']
        target_list = label.split('\n')
        for item in target_list:
            try:
                target_tuple = ast.literal_eval(item.strip())
            except:
                target_tuple = []
            if target_tuple != []:
                target_tuple_list.append([target_item.lower() for target_item in target_tuple])
        test_labels[i] = target_tuple_list
    
    for format in format_list:                           
        test_data_name = 'prediction_' + format + '_' + model
        print('test data name: ' + test_data_name)
        
        # statistics
        data_num = 0
        # count wrong formatted samples
        error_format_num = 0
        # count empty output
        empty_num = 0

        num_undefined_type = 0
        tp = 0
        fp = 0
        fn = 0
        
        error_sample = []

        # read test json file
        with open(os.path.join(folder, test_data_name) + '.json', 'r', encoding='utf8') as f:
            test_lines = f.readlines()
        data_num = len(test_lines)
        for i in range(len(test_lines)):
            test_predict = json.loads(test_lines[i])['prediction']
            test_label = test_labels[i]
            # ground truth trigger list of one sample
            test_triggers = [e_item[1] for e_item in test_label]
            # process model predict
            processed_pred_list, format_error_flag, empty_flag = convert_to_list(test_predict, format)
            error_format_num += format_error_flag
            empty_num += empty_flag
            if format_error_flag != 0:
                error_sample.append(json.loads(test_lines[i]))
            
            # get correct event tuples
            correct_list = get_correct_list(processed_pred_list, test_label, test_triggers, is_soft)
            tp += len(correct_list)
            fp += len(processed_pred_list) - len(correct_list)
            fn += len(test_label) - len(correct_list)

        print('格式不符合数量：', len(error_sample))
        if is_write == True:
            with open(os.path.join(result_folder, test_data_name) + '_error_samples.json', 'w') as f:
                for sample in error_sample:
                    f.write(json.dumps(sample, ensure_ascii=False))
                    f.write('\n')
                    
        # statistic
        print('format ----- ' + format)
        print('sample number: ' + str(data_num))
        precision, recall, f1 = get_metrics(tp, fp, fn)
        print('precision: ' + str(precision))
        print('recall: ' + str(recall))
        print('f1: ' + str(f1))
        metrics_f = os.path.join(result_folder, test_data_name)
        if is_soft == True:
            metrics_f += '_soft'
        with open(metrics_f + '_f1.txt', 'w', encoding='utf-8') as f:
            f.write(test_data_name+'\n')
            f.write('样本数量：' + str(data_num) + '\n')
            f.write('格式不符合数量：'+ str(error_format_num) + '\n')
            f.write('输出为空数量：'+ str(empty_num) + '\n')
            f.write('格式不符合数量占比：'+str(round(error_format_num/data_num, 4)) + '\n')
            f.write('precision: ' + str(precision) + '\n')
            f.write('recall: ' + str(recall) + '\n')
            f.write('f1: ' + str(f1) + '\n')
            

