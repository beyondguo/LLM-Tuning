

import json
import os
import random

raw_data_dir = "./raw_data"
output_dir = "./data"


## NER dataset pre-processing
def ner_data_process(dataset, input_file, output_file, type_file, separator=" "):
    # # type file
    type_file_name = os.path.join(raw_data_dir, os.path.join(dataset, type_file))
    print("load type file: ", type_file_name)
    with open(type_file_name, 'r', encoding='utf-8') as fr_t:
        type_data = json.load(fr_t)

    # # output dir
    output_path = os.path.join(output_dir, dataset) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # # input file
    in_file_name = os.path.join(raw_data_dir, os.path.join(dataset, input_file))
    print("begin processing: ", in_file_name)
    with open(in_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_num_entity = 0
    total_length_sentence = 0.0
    max_num_entity_per_sentence = 0
    num_nested_sentence = 0
    num_nested_entity = 0
    
    new_data = []
    unique_sample_list = []
    for example in data:
        seq_tokens = example['tokens']
        # # 去重
        # if seq_tokens in unique_sample_list:
        #     # print(seq_tokens)
        #     continue
        # else:
        #     unique_sample_list.append(seq_tokens)
        total_length_sentence += len(seq_tokens)
        sent = separator.join(seq_tokens)

        entities = example['entities']
        if len(entities) > max_num_entity_per_sentence:
            max_num_entity_per_sentence = len(entities)

        entity_list = []
        for entity in entities:
            total_num_entity += 1

            e_seq = separator.join(seq_tokens[entity['start']: entity['end']])
            e_type = entity['type']
            e_type_verbose = type_data["entities"][e_type]["verbose"]
            entity_list.append(
                {
                "e_name": e_seq,
                "e_type": e_type,
                "e_type_verbose": e_type_verbose,
                "start": entity['start'],
                "end": entity["end"]
                }
            )

        entity_list = sorted(entity_list, key=lambda e: e['start'])

        ### judge nested entities
        nested_entity_list = []
        for ii in range(len(entity_list)):
            entity = entity_list[ii]
            start = entity["start"]
            end = entity["end"]

            for jj in range(len(entity_list)):
                if ii != jj:
                    entity_1 = entity_list[jj]
                    tar_start = entity_1["start"]
                    tar_end = entity_1["end"]
                    if (start >= tar_start and end <= tar_end) or (start <= tar_start and end >= tar_end):
                        if entity not in nested_entity_list:
                            nested_entity_list.append(entity)
                        if entity_1 not in nested_entity_list:
                            nested_entity_list.append(entity_1)
        ###

        new_example = dict()
        if len(nested_entity_list) != 0:
            new_example['mode'] = "nested"
            num_nested_sentence += 1
            num_nested_entity += len(nested_entity_list)
            nested_entity_list = sorted(nested_entity_list, key=lambda e: e['start'])
        else:
            new_example['mode'] = "flat"
        
        new_example['seq'] = sent
        new_example['entities'] = entity_list
        new_data.append(new_example)

    print("#sentences: ", len(new_data))
    print("#entities : ", total_num_entity)
    print("avg. sentence length       : ", total_length_sentence/len(new_data))
    print("max. #entities per sentence: ", max_num_entity_per_sentence)
    print("avg. #entities per sentence: ", total_num_entity*1.0/len(new_data))
    print("# nested sentences : ", num_nested_sentence)
    print("# nested entities  : ", num_nested_entity)
    print("nesting ratio      : ", num_nested_entity*1.0/total_num_entity)
    
    with open(os.path.join(output_path, output_file), 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(new_data, indent=4, ensure_ascii=False))
    with open(os.path.join(output_path, "types.json"), 'w', encoding='utf-8') as fw_t:
        fw_t.write(json.dumps(type_data, indent=4, ensure_ascii=False))


def ner_nested_data_process(dataset, input_file, output_file, type_file, num_type, separator=" "):
    ## ace2004 and ace2005 types
    type_file_name = os.path.join(raw_data_dir, os.path.join(dataset, type_file))
    print("load type file: ", type_file_name)
    with open(type_file_name, 'r', encoding='utf-8') as fr_t:
        type_data = json.load(fr_t)

    ## output dir
    output_path = os.path.join(output_dir, dataset) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## input file
    in_file_name = os.path.join(raw_data_dir, os.path.join(dataset, input_file))
    print("begin processing: ", in_file_name)
    with open(in_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_num_entity = 0
    total_length_sentence = 0.0
    max_num_entity_per_sentence = 0
    num_nested_sentence = 0
    num_nested_entity = 0

    i = 0
    new_data = []
    unique_sample_list = []
    while i < len(data):
        sent = data[i]["context"]
        # # 去重
        # if sent in unique_sample_list:
        #     i += num_type
        #     # print(sent)
        #     continue
        # else:
        #     unique_sample_list.append(sent)
        seq_tokens = sent.split(separator)
        total_length_sentence += len(seq_tokens)

        entity_list = []
        for j in range(num_type):
            example = data[i + j]
            spans = example["span_position"]
            e_type = example["entity_label"]
            e_type_verbose = type_data["entities"][e_type]["verbose"]

            for span in spans:
                start_id = int(span.split(";")[0])
                end_id = int(span.split(";")[1]) + 1
                e_name = separator.join(seq_tokens[start_id: end_id])
                entity_list.append(
                    {
                    "e_name": e_name,
                    "e_type": e_type,
                    "e_type_verbose": e_type_verbose,
                    "start": start_id,
                    "end": end_id
                    }
                )
        
        total_num_entity += len(entity_list)
        if len(entity_list) > max_num_entity_per_sentence:
            max_num_entity_per_sentence = len(entity_list)

        entity_list = sorted(entity_list, key=lambda e: e['start'])

        ### judge nested entities
        nested_entity_list = []
        for ii in range(len(entity_list)):
            entity = entity_list[ii]
            start = entity["start"]
            end = entity["end"]

            for jj in range(len(entity_list)):
                if ii != jj:
                    entity_1 = entity_list[jj]
                    tar_start = entity_1["start"]
                    tar_end = entity_1["end"]
                    if (start >= tar_start and end <= tar_end) or (start <= tar_start and end >= tar_end):
                        if entity not in nested_entity_list:
                            nested_entity_list.append(entity)

                        if entity_1 not in nested_entity_list:
                            nested_entity_list.append(entity_1)
        ###

        new_example = dict()
        if len(nested_entity_list) != 0:
            new_example['mode'] = "nested"
            num_nested_sentence += 1
            num_nested_entity += len(nested_entity_list)
            nested_entity_list = sorted(nested_entity_list, key=lambda e: e['start'])
        else:
            new_example['mode'] = "flat"
        new_example['seq'] = sent
        new_example['entities'] = entity_list
        new_data.append(new_example)
        i += num_type

    print("#sentences: ", len(new_data))
    print("#entities : ", total_num_entity)
    print("avg. sentence length       : ", total_length_sentence/len(new_data))
    print("max. #entities per sentence: ", max_num_entity_per_sentence)
    print("avg. #entities per sentence: ", total_num_entity*1.0/len(new_data))
    print("# nested sentences : ", num_nested_sentence)
    print("# nested entities  : ", num_nested_entity)
    print("nesting ratio      : ", num_nested_entity*1.0/total_num_entity)

    # json.dump(new_data, open(os.path.join(output_path, output_file), 'w'))
    with open(os.path.join(output_path, output_file), 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(new_data, indent=4, ensure_ascii=False))

    with open(os.path.join(output_path, "types.json"), 'w', encoding='utf-8') as fw_t:
        fw_t.write(json.dumps(type_data, indent=4, ensure_ascii=False))



## sample 3k
def sample_data(dataset, input_file, output_file, k):
    in_file_name = os.path.join(output_dir, os.path.join(dataset, input_file))
    # print("begin processing: ", in_file_name)
    with open(in_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out_file = os.path.join(output_dir, os.path.join(dataset, output_file))
    new_data = []
    with open(out_file, 'w', encoding='utf-8') as fw:
        selected_idx = random.sample(list(range(0, len(data))), k)
        selected_idx.sort()
        for idx, example in enumerate(data):
            if idx in selected_idx:
                new_data.append(example)

        print(len(new_data))
        json.dump(new_data, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    # NER
    # 3453 5648 13.45 31 1.64 0 0 0
    # 3184 5450 14.25 31 1.71 0 0 0
    # ner_data_process("ner/conll03/", "conll032_test_context.json", "ner_test.json", "conll032_types.json")
    # ner_data_process("ner/conll03/", "conll032_train_dev_context.json", "ner_train.json", "conll032_types.json")
    print()
    # 37648 96902 24.47 49 2.57 0 0 0
    # 37060 95439 24.48 49 2.57 0 0 0
    # ner_data_process("ner/fewnerd/", "fewnerd_test_context.json", "ner_test.json", "fewnerd_types.json")
    # ner_data_process("ner/fewnerd/", "fewnerd_train_context.json", "ner_train.json", "fewnerd_types.json")
    print()
    # 4365 6181 39.54 461 1.42 0 0 0
    # 4316 6160 39.88 461 1.43 0 0 0
    # ner_data_process("ner/zhmsra/", "zhmsra_test_context.json", "ner_test.json", "zhmsra_types.json", separator="")
    # 1854 5506 25.99 14 2.97 446 1199 0.2178
    # 1850 5506 26.03 14 2.98 446 1199 0.2178
    ner_data_process("ner/genia/", "genia_test_context.json", "ner_test.json", "genia_types.json")
    ner_data_process("ner/genia/", "genia_train_dev_context.json", "ner_train.json", "genia_types.json")
    print()
    # 812 3035 23.05 20 3.74 388 1415 0.4662   -->  1417
    # 808 3035 23.15 20 3.76 388 1415 0.4662  
    # ner_nested_data_process("ner/ace2004", "mrc-ner.test", "ner_test.json", "ace2004_types.json", 7)
    # ner_nested_data_process("ner/ace2004", "mrc-ner.train", "ner_train.json", "ace2004_types.json", 7)
    print()
    # 1060 3042 17.90 20 2.87 345 1189 0.3909  -->  均对不上
    # 1050 3041 18.03 20 2.90 345 1189 0.3910
    # ner_nested_data_process("ner/ace2005", "mrc-ner.test", "ner_test.json", "ace2005_types.json", 7)
    # ner_nested_data_process("ner/ace2005", "mrc-ner.train", "ner_train.json", "ace2005_types.json", 7)
    print()
    
    ## sample
    # sample_data("ner/fewnerd/", "ner_test.json", "ner_test_3k.json", 3000)
    