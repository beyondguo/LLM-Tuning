import json 

def JSON_text(text, mode, aspects_list, punc_replace):
    '''
    transfer different predictions to a unified JSON format
    punc_replace: if True, replace the incorrect punctuations with correct ones
    '''
    if mode in ['instruction-last','no-instruction','instruction-first','input-modeling']:
        dict_text = {}
        error_type = ''
        if text == '':
            error_type = 'empty_output'
        else:
            text = text.strip('。')
            for l in text.split('，'):
                try:
                    l = l.split('：')
                    dict_text[l[0]] = l[1]
                except:
                    error_type = 'split_error'
        return dict_text, error_type
    if mode == ['omit-unmentioned', 'EW-SDE']:
        dict_text = {}
        error_type = ''
        if text == '':
            error_typ = 'empty_output'
        else:
            text = text.strip('。')
            for l in text.split('，'):
                if l != '':
                    try:
                        l = l.split('：')
                        dict_text[l[0]] = l[1]
                    except:
                        error_type = 'split_error'
            for i in aspects_list:
                if i not in [*dict_text]:
                    dict_text[i] = '未提及'
        return dict_text, error_type
    if mode == 'numerical-label':
        text_num = {'1':'正面','0':'中性','-1':'负面','-2':'未提及'}
        dict_text = {}
        error_type = ''
        if text == '':
            error_type = 'empty_output'
        else:
            text = text.strip('。')
            for l in text.split('，'):
                try:
                    l=l.split('：')
                    dict_text[l[0]] = text_num[l[1]]
                except:
                    error_type = 'split_error'
        return dict_text, error_type
    if mode in ['lines-output', 'ES-SDE']:
        error_type = ''
        dict_text = {}
        if text == '':
            error_type = 'empty_output'
        else:
            text = text.strip('\n')
            for l in text.split('\n'):
                try:
                    l = l.split('：')
                    dict_text[l[0]] = l[1]
                except:
                    error_type = 'split_error'
        return dict_text,error_type
    
    if mode in ['json-output','cot','rev-cot']:  
        error_type = ''
        dict_text = {}
        if text == '':
            error_type = 'empty_output'
        else:              
            if mode != 'json-output' and punc_replace == True:
                text=text.replace("}, {","}\n{").replace("},\n{","}\n{").replace("}, \n{","}\n{")
                text=text.replace("\"， \"","\", \"").replace("\"; \"","\", \"").replace("\" ; \"","\", \"").replace("\"}, \"","\", \"").replace("\"},\"","\", \"")   
            text = text.strip('\n').split('\n')
            length = len(text)  
            for i in range(length):
                l = text[i]
                try:
                    eval_l = eval(l)
                    dict_text[eval_l["方面"]] = eval_l["情感"]
                except:
                    error_type = 'eval_error'
                    l = l.strip('"{').strip('}"').strip('{').strip('}').split('", "')  # ['aspect': '...', 'description': '...', 'sentiment': '...']
                    try:
                        for ls in l:
                            ls = ls.split('": "')
                            if ls[0] == '方面':
                                asp = ls[1]
                            if ls[0] == '情感':
                                senti = ls[1]
                        dict_text[asp] = senti
                    except:
                        error_type = 'split_error'
                        pass   
        return dict_text, error_type


def line_processor(line, i , mode, aspects_list, soft_keys, soft_senti, is_write, is_soft, punc_replace):
    d = json.loads(line)
    label_text = d['target']
    output_text = d['prediction']
    prompt = d['prompt']
    is_error = False
    label, _ = JSON_text(label_text, mode, aspects_list, punc_replace)
    if _ != '':
        label = {}
        print('label error')
    output,error_type = JSON_text(output_text, mode, aspects_list, punc_replace)   
    
    if is_soft == True :
        #i) relax aspects
        new_dict = {}
        for key_ori in [*output]:
            for k in range(len(soft_keys)):
                soft_key = soft_keys[k]
                aspect_key = aspects_list[k]
                if type(soft_key) == list:
                    for s in soft_key:
                        if s in str(key_ori) and aspect_key not in new_dict:
                            new_dict[aspect_key] = output[key_ori]
                else:
                    if soft_key in str(key_ori) and aspect_key not in new_dict:
                        new_dict[aspect_key] = output[key_ori]  
        output=new_dict     

        # ii) relax sentiment labels
        senti = [*soft_senti]
        senti_keys = [*output]
        for k in senti_keys:
            v = output[k]
            for s in senti:
                if s in v:
                    output[k] = soft_senti[s]
                    break
        # iii) aspects not appearing in the predictions are regarded as 'unmentioned'
        for asp in aspects_list:
            if asp not in [*output]:
                output[asp] = '未提及'            
            if output[asp] not in ['正面','中性','负面','未提及']:
                output[asp] = '未提及'  

    # check aspects and sentiment labels                            
    try:           
        assert set([*output]) == set(aspects_list)
        assert set(output.values()).issubset(['正面','中性','负面','未提及']) == True                     
    except:
        error_type = 'check_error'
        output = {}
        print('格式检验未通过：' + output_text)
    
    if error_type != '':
        is_error = True   
        
    processed_error_line = ''  
    if is_write == True and error_type != '':
        processed_error_line = {'id': str(i), 'error_type': error_type, 'prompt': str(prompt), 'label': str(label_text), 'prediction': str(output_text), 'processed_pre': output}
        processed_error_line = json.dumps(processed_error_line, ensure_ascii=False)

    processed_line = {'prompt': prompt, 'label': label, 'prediction': output}
    
    return processed_line, is_error, processed_error_line
