import json
from tqdm import tqdm
    
tp_list = ["DNA","RNA","protein","cell_type","cell_line"]

desc1 = "Read the given sentence carefully, identify all named entities of type \"DNA\", \"RNA\", \"protein\", \"cell_type\" or \"cell_line\"."

desc2 = "Carefully review the provided statement and detect any specific named entities that belong to the categories of \"DNA\", \"RNA\", \"protein\", \"cell_type\" or \"cell_line\""

desc3 = "This is a nested entity recognition problem, with these entity types: \"DNA\", \"RNA\", \"protein\", \"cell_type\" or \"cell_line\". Please thoroughly examine the provided sentence and accurately identify all instances of named entities."
desc_list = [desc1,desc2,desc3]

def generate_sample(example, mode='default', desc_num=0):
    desc = desc_list[desc_num]
    sent = example['seq']
    entities = example['entities']
    answer_d = {}
    for tp in tp_list:
        answer_d[tp] = []
    for e in entities:
        answer_d[e['e_type_verbose']].append(e['e_name'])
      
    if mode == 'default':
        prompt = f"[INST]{desc} \
Answer in the format [\"entity_type\", \"entity_name\"]. If no entity exists, then just answer \"[]\".\
\nGiven sentence: \"{sent}\"[/INST]"
        output = '\n'.join(['["'+e['e_type']+'","'+e['e_name']+'"]' for e in entities])
    
    if mode == 'weak1': # ins-first, natural concat, mentioned-only
        prompt = f"[INST]Given sentence: \"{sent}\"\n{desc} \
For each entity type, answer in the format like \"'entity_type': 'entity_name_1', 'entity_name_2'...\", then concat answer for each type with ';'. \
Only output entity types that contain entities.[/INST]"
        output = '; '.join([f"'{tp}': {str(answer_d[tp]).replace('[','').replace(']','')}" for tp in tp_list if answer_d[tp] != []])

    if mode == 'good1': # ins-last, lines output
        prompt = f"[INST]{desc} \
For each entity type, answer in a line in the format like \"'entity_type': 'entity_name_1', 'entity_name_2'...\" (when no entities exist, answer  \"'entity_type': ''\").\
\nGiven sentence: \"{sent}\"[/INST]"
        output = '\n'.join([f"'{tp}': {str(answer_d[tp]).replace('[]','None').replace('[','').replace(']','')}" for tp in tp_list])
        output = output.replace('None','\'\'')
        
    if mode == 'good2': # ins-last, dict output
        tmp = '{"entity_type": "DNA","entity_names": ["name1", "name2"]}'
        prompt = f"[INST]Read the given sentence carefully, identify all named entities of type \"DNA\", \"RNA\", \"protein\", \"cell_type\" or \"cell_line\". \
For each entity type, answer in a line with a dictionary where the key is the entity type and value a list of entity names: \'{tmp}\' (output empty list if no entities for a certain type).\
\nGiven sentence: \"{sent}\"[/INST]"
        output = []
        for tp in tp_list:
            line = json.dumps({"entity_type":tp, "entity_names":answer_d[tp]})
            output.append(line)
        output = '\n'.join(output)
    
    return {"prompt":prompt, "output":output}


# ------------ train data ----------
with open('data/ner/genia/ner_train.json','r') as fr:
    data = json.load(fr)
print('~~~~total size:', len(data))

N = 4500
desc_num = 0
for mode in ['default','weak1','good1']:
    with open(f'design_data/ner/NER_train_val_{str(N)}_{mode}_desc{desc_num}_design.json','w') as fw:
        for i in range(N):
            sample = json.dumps(generate_sample(data[i], mode, desc_num))
            fw.write(sample+'\n')
            
    print(mode, 'finished!')
    
# ------------ test data ----------
# with open('data/ner/genia/ner_test.json','r') as fr:
#     data = json.load(fr)
    
# for mode in ['default','weak1','good1','good2']:
#     with open(f'design_data/ner/NER_test_{mode}_desc{desc_num}_design.json','w') as fw:
#         for i in tqdm(range(len(data))):
#             sample = json.dumps(generate_sample(data[i], mode, desc_num))
#             fw.write(sample+'\n')
            
#     print(mode, 'finished!')