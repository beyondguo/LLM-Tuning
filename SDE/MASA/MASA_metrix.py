import json
import pandas as pd
import numpy as np
import os
from format_parser import line_processor
from sentiment_analysis import get_confusion_matrix


aspect_name = {'1': ['食品评价', '饮品评价', '价格水平', '卫生情况', '服务人员态度', '停车方便程度'],
    '2': ['交通便利程度', '排队等候情况', '点菜上菜速度', '装修情况', '嘈杂情况'],
    'all': ['食品评价', '饮品评价', '价格水平', '卫生情况', '服务人员态度', '停车方便程度', '交通便利程度', '排队等候情况', '点菜上菜速度', '装修情况', '嘈杂情况'] }
# Relax the matching condition for aspects and labels to calculate sentiment performance
# For example: for 'traffic convenience', the model output 'traffic situation' is acceptable
soft_aspect_name = {'1': ['食品','饮品','价格','卫生','服务','停车'],
    '2': ['交通',['排队','等候'],'点菜','装修','嘈杂'],
    'all': ['食品','饮品','价格','卫生','服务','停车','交通',['排队','等候'],'点菜','装修','嘈杂']}
soft_senti = {'积极':'正面','正向':'正面','正面':'正面','中性':'中性','消极':'负面','负向':'负面','负面':'负面','未提及':'未提及'}   

mode_list = ['instruction-last','instruction-first','no-instruction','input-modeling', 'lines-output','json-output','numerical-label','omit-unmentioned','cot','rev-cot']   
mode_list = ['instruction-last']

if __name__ == '__main__':
    model = 'internlm-chat'
    folder = os.path.join(os.path.dirname(__file__), 'eval_' + model)
    result_folder = os.path.join(os.path.dirname(__file__), 'result_' + model)  
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # whether to record wrong format samples
    is_write = True 
    # whether to output a csv file
    is_table = True
    # whether to relax aspects and labels
    is_soft = True

    #for train_num in ['500','1000']:#train_size
    for train_num in ['500']:
        table = pd.DataFrame(columns=['aspect','D11_k','D22_k','D12_k','D21_k','D11_wf','D22_wf','D12_wf','D21_wf','D11_a','D22_a','D12_a','D21_a'])
        
        for mode in mode_list:
            result_table = {'aspect': mode}           
            #for train_domain, test_domain in [('1','1'),('1','2'),('2','1'),('2','2')]:  
            for train_domain, test_domain in [('1','1')]:
                test_data_name = mode + '_d' + train_domain + '_' + train_num + '_' + model + '_d' + test_domain + '_predictions'
                # read test data
                with open(os.path.join(folder,test_data_name) + '.json', 'r', encoding='utf8') as f:
                    lines = f.readlines()

                # skip 'no_instruction' method on out-of-domain tasks
                if mode == 'no_instruction' and train_domain != test_domain:
                    acc_all = kappa_all = num_format_error = '\\'
                
                # experimental results 
                # i)format adherence 
                else:
                    num_format_error = 0  # count samples with wrong format
                    processed_lines = [] 
                    processed_error_lines = []  # wrong formatted samples
                    for i in range(len(lines)):
                        line = lines[i]
                        processed_line, is_error, processed_error_line = line_processor(line, i , mode, aspect_name[test_domain], soft_aspect_name[test_domain], soft_senti, is_write, is_soft, punc_replace=True)
                        # For some LLMs, The COT-related approach leads to more punctuation errors, so we run twice with different punc_replace parameters
                        # punc_replace=True --- to process outputs into a unified format for sentiment analysis(ignore punctuation errors)
                        # punc_replace=False --- to get wrong formatted samples(count punctuation errors)
                        if mode in ['cot','rev-cot']:
                            _, is_error, processed_error_line = line_processor(line, i , mode, aspect_name[test_domain], soft_aspect_name[test_domain], soft_senti, is_write, is_soft, punc_replace=False) 
                        num_format_error += is_error
                        processed_lines.append(processed_line)
                        if processed_error_line != '':
                            processed_error_lines.append(processed_error_line)
                    print('格式不符合数量：', num_format_error)
                    soft = '_soft' if is_soft == True else ''
                    # record the wrong format samples
                    if is_write == True and processed_error_lines != []:
                        wf_file_name='wrong_format_' + test_data_name + soft + '.json'
                        wf_samples_file = os.path.join(result_folder, wf_file_name)
                        with open(wf_samples_file, 'w', encoding='utf-8') as f:
                            for l in processed_error_lines:
                                f.write(l + '\n')  
                    # record the format error rate
                    result_file_name='experimental_results_' + test_data_name + soft + '.txt'
                    write_f = os.path.join(result_folder, result_file_name)
                    with open(write_f, 'w', encoding='utf-8') as f:
                        f.write(test_data_name + '\n')
                        f.write('样本数量：' + str(len(lines)) + '\n')
                        f.write('格式不符合数量：' + str(num_format_error) + '\n')
                        f.write('格式不符合数量占比：' + str(round(num_format_error/len(lines),4)) + '\n')

                    # ii)sentiment analysis
                    if len(processed_lines) != num_format_error:
                        # calculate and record the sentiment analysis performance
                        acc_all,kappa_all = get_confusion_matrix(write_f, processed_lines, aspect_name[test_domain])
                    else:
                        acc_all=kappa_all = '\\'        

                # save the table 
                # for each domain
                result_table['D' + train_domain + test_domain+'_k'] = kappa_all
                result_table['D' + train_domain + test_domain+'_a'] = acc_all
                result_table['D' + train_domain + test_domain+'_wf'] = num_format_error           
            #for each mode
            table=pd.concat([table,pd.DataFrame(result_table,index=[0])], ignore_index=True)
        #for each train_size
        if is_table == True:
            table_name = train_num + soft + '_table.csv'
            table.to_csv(os.path.join(result_folder, table_name), index=False, float_format='%.4f')
