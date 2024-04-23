"""
Copied from https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction/blob/main/utils.py
Acknowledgment to repo: https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction
Paper: Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors (https://arxiv.org/pdf/2305.14450.pdf)

"""
import json
import os, sys
# import openai
# import backoff
import ast
# import threading


## multi thread start
## Read One Sample
class ReadSample(object):
    """read one sample from the data list """
    def __init__(self, data_list, data_idx_list, start_line=0):
        self.data = data_list
        self.data_idx = data_idx_list
        self.num_data = len(self.data_idx)
        self.lock = threading.RLock()  # 同步锁
        self.cur_index = start_line  # 当前读取位置
 
    def get_item(self):
        self.lock.acquire()
        try:
            if self.cur_index < self.num_data:
                sample = self.data[self.data_idx[self.cur_index]]
                self.cur_index += 1
                return True, sample
            else:
                return False, None
            
        except Exception as e:
            return False, "error:" + e.args
        finally:
            self.lock.release()

## Write One Sample
class WriteSample(object): 
    def __init__(self, file_name, mode):
        self.file_name = file_name
        self.mode = mode
        self.lock = threading.RLock()
        self.clear()

    def clear(self):
        with open(self.file_name, 'a', encoding='utf-8') as fw:
            fw.seek(0)  #定位
            fw.truncate()   #清空文件
 
    def write(self, sample):
        self.lock.acquire()
        with open(self.file_name, self.mode, encoding='utf-8') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        self.lock.release()

## multi thread end


## Logger
class Logger(object):
    def __init__(self, file_name = 'chatgpt_eval.log', stream = sys.stdout) -> None:
        self.terminal = stream
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log = open(os.path.join(log_dir, file_name), "a", encoding='utf-8')
        self.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.seek(0)	# 定位
        self.log.truncate()


# ## connect OpenAI API
# @backoff.on_exception(backoff.expo, \
#                       (openai.error.RateLimitError, 
#                        openai.error.APIConnectionError, 
#                        openai.error.APIError,
#                        openai.error.ServiceUnavailableError))
# def bot_create(bot, para):
#     return bot.create(**para).choices[0].message

# def bot_run(bot, prompt, model="gpt-3.5-turbo-0301"):
#     para = {
#         "model": model,
#         "temperature": 0.0,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ]
#     }
#     response = bot_create(bot, para)
#     response = response["content"].strip().strip("\n")

#     return response


## Eval
def response_string_to_list(response):
    """return 
        1) string 列表
        2) list  列表
    """
    def get_list_by_string(list_str, ori_response):
        try:
            res_list = ast.literal_eval(list_str) 
            flag = True
        except:
            res_list = []
            flag = False
            # print(ori_response)
        finally:
            return res_list, flag
    
    ori_response = response
    response = response.replace("(", "[").replace(")", "]")
    response = response.lower()
    num_left = response.count("[")

    res_list = []

    if num_left == 0:
        # print(ori_response)
        return res_list, False
    
    if num_left == 1:
        start_idx = response.find('[')
        response = response[start_idx:]
        num_right = response.count("]")
        if num_right < 1:
            # print(ori_response)
            return res_list, False
        else:
            start_idx = response.find('[')
            end_idx = response.find(']')
            span = response[start_idx: end_idx+1]
            res_list, flag = get_list_by_string(span, ori_response)
            res_list = [res.strip() for res in res_list] 
            return res_list, flag

    # "['a', 'b'], ['c', 'd']"
    start_idx = -1
    end_idx = -1

    res_flag = True
    for i, ch in enumerate(response):
        if ch == '[':
            start_idx = i
        if ch == ']':
            end_idx = i
        # print(start_idx, end_idx)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            span = response[start_idx: end_idx+1]
            # print(span)
            tmp_list, flag = get_list_by_string(span, ori_response)
            if not flag:
                res_flag = False
                # print(response)
            tmp_list = [str(res).strip() for res in tmp_list] 
            res_list.append(tmp_list)
            start_idx = -1
            end_idx = -1
        elif  start_idx != -1 and end_idx != -1 and start_idx > end_idx:
            res_flag = False
            # print(response)


    return res_list, res_flag


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


def print_metrics(tp, fp, fn, logger, task, align=8):
    p, r, f1 = 0.0, 0.0, 0.0

    if tp + fp != 0:
        p = 1.0 * tp / (tp + fp)
    if tp + fn != 0:
        r = 1.0 * tp / (tp + fn)
    if p + r != 0.0:
        f1 = 2.0 * p * r / (p + r)
        
    logger.write("{} | p: {:.4f}, r: {:.4f}, f1: {:.4f} | tp: {:4d}, fp: {:4d}, fn: {:4d}, tp+fn: {:4d}\n".format(
        task.ljust(align),
        round(p, 4),
        round(r, 4),
        round(f1, 4),
        tp,
        fp,
        fn,
        tp+fn,
        )
    )
    return f1

    