"""
将数据集中所有样本变成1000个token序列。
输出字典 candi_Data{
    "CWE-1":{
        "Embeddings":[[1000个tokens],[]]
        "Labels""[]
    },...
    "Benign":{
        "Embeddings":[]
        "Labels""[]
    }
}

python prep.py --Dataset SARD
               --Dataset Data_six
               --Dataset SARD_4

(The above operations are only suitable for system supported datasets. Identify all the required parameters for customized dataset.)

"""


import re
import os
import json, random
from DataLoader import LoadPickleData
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import argparse

random.seed(445)


# ============================================  1. 数据集中所有文件程序向量化 ====================================================
def load_dataset_json(json_path):
    """
    载入待处理数据集的 JSON 信息
    ------------------------
    E.g., Data_six 数据集：
        将 static.json 载入 result{}
        Data_six/Six_project_info/static.json文件中有各个文件CWE类型的信息。
        可在命令行通过以下指令查看：
        $ cat static.json|python -m json.tool
    E.g., SARD 数据集：
        将 CWE_info.json 载入 result{}
    """
    result = {}
    with open(json_path) as f:
        result = json.load(f)
        # print("JSON file Loaded.")
        # print(result)
        # print("---------------------------------")
    return result


def SplitCharacters(str_to_split):
    """
    源码 --> token sequence，分割标点
    -----------------------------------------------
    函数输入：   “str_to_split” —— 待序列化的程序字符串。
    函数操作：   1）把标点前后都加空格
                2）按空格分割各个token
                3）然后把各个token重新连接成 token 序列。
    函数返回值： “str_list_str” —— token sequence
    """

    # Character_sets = ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',']

    str_list_str = ''

    if '(' in str_to_split:
        str_to_split = str_to_split.replace(
            '(', ' ( '
        )  # Add the space before and after the '(', so that it can be
        # split by space.把“（“替换成” （ “。
        str_list = str_to_split.split(' ')  # 按空格分割成各个token
        str_list_str = ' '.join(str_list)  # 把各个token重新拼接，中间用空格分开

    if ')' in str_to_split:
        str_to_split = str_to_split.replace(
            ')', ' ) '
        )  # Add the space before and after the ')', so that it can be
        # split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '{' in str_to_split:
        str_to_split = str_to_split.replace('{', ' { ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '}' in str_to_split:
        str_to_split = str_to_split.replace('}', ' } ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '*' in str_to_split:
        str_to_split = str_to_split.replace('*', ' * ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '/' in str_to_split:
        str_to_split = str_to_split.replace('/', ' / ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '+' in str_to_split:
        str_to_split = str_to_split.replace('+', ' + ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '-' in str_to_split:
        str_to_split = str_to_split.replace('-', ' - ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '=' in str_to_split:
        str_to_split = str_to_split.replace('=', ' = ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if ';' in str_to_split:
        str_to_split = str_to_split.replace(';', ' ; ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '[' in str_to_split:
        str_to_split = str_to_split.replace('[', ' [ ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if ']' in str_to_split:
        str_to_split = str_to_split.replace(']', ' ] ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '>' in str_to_split:
        str_to_split = str_to_split.replace('>', ' > ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '<' in str_to_split:
        str_to_split = str_to_split.replace('<', ' < ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '"' in str_to_split:
        str_to_split = str_to_split.replace('"', ' " ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '->' in str_to_split:
        str_to_split = str_to_split.replace('->', ' -> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '>>' in str_to_split:
        str_to_split = str_to_split.replace('>>', ' >> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if '<<' in str_to_split:
        str_to_split = str_to_split.replace('<<', ' << ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if ',' in str_to_split:
        str_to_split = str_to_split.replace(',', ' , ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if str_list_str != '':
        return str_list_str
    else:
        return str_to_split


def tokenize(fpath, f):
    """
    将单个文件转化成token列表
    """
    with open(os.path.join(fpath, f), encoding='utf-8', errors="ignore") as file:
        lines = file.readlines()
        file_list = []
        for line in lines:
            if line != ' ' and line != '\n':  # Remove sapce and line-change characters
                sub_line = line.split()
                new_sub_line = []
                for element in sub_line:
                    new_element = SplitCharacters(element)
                    new_sub_line.append(new_element)
                new_line = ' '.join(new_sub_line)
                file_list.append(new_line)
        new_file_list = ' '.join(file_list)
        tokens = new_file_list.split()
    return tokens


def get_all_files_and_ids(path):
    '''获取数据集所有文件文本的token序列和文件名(专门给word2vec模型训练使用)'''
    total_list = []  # 所有.c, .cpp程序token序列的列表
    total_list_id = []  # 所有.c, .cpp文件名的列表
    if not os.path.isdir(path):
        print("Error! Path is not a dir.")
    else:
        for fpath, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.c') or f.endswith('.cpp'):
                    tokens = tokenize(fpath, f)
                    total_list.append(tokens)
                    total_list_id.append(f)
        return total_list, total_list_id


def get_files_and_ids(path):
    """
    只用于 Data_six 数据集，因为没有指明无漏洞文件名。
    获取有、无漏洞的文件名及相应token序列的列表 为后续向量化做准备。
    -----------------------------------------------------
    程序输入:
        path —— 数据集所在路径。e.g. Datasets/Data_six/Six_project/
    程序输出：
        1）把有漏洞的.c, .cpp文件放入           flawFile_id_list[]列表
        2）把无漏洞的.c, .cpp文件放入           file_id_list[]列表
        3）各个有漏洞.c, .cpp程序token序列的列表   flawFiles_list
        4）各个无漏洞.c, .cpp程序token序列的列表   files_list
    """
    files_list = []  # 各无漏洞文件的 token 列表
    file_id_list = []
    flawFiles_list = []  # 有漏洞文件的 token 列表
    flawFile_id_list = []  # 有CVE编号（即有漏洞）的.c文件列表

    if not os.path.isdir(path):
        print("Error! Path is not a dir.")
    else:
        for fpath, dirs, fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1] == '.c' or os.path.splitext(f)[1] == '.cpp':
                    tokens = tokenize(fpath, f)
                    if re.findall('CVE-(\d+)', os.path.splitext(f)[0]) or re.findall(
                        'cve-(\d+)', os.path.splitext(f)[0]
                    ):
                        flawFile_id_list.append(f)
                        flawFiles_list.append(tokens)
                    else:
                        file_id_list.append(f)
                        files_list.append(tokens)
        return files_list, file_id_list, flawFiles_list, flawFile_id_list


def get_Dicts(
    result, files_list, file_list_id, flawFiles_list, flawFile_id_list, tokenizer
):
    """
    Data_six数据集

    获取有、无漏洞文件的 token 及embedding字典
    --------------------------------------------
    flaw_Dict{}           有漏洞程序的文件的token字典
    flaw_embed_dict{}     有漏洞程序的 Embedding 字典
    all_embed_dict{}      包含有、无漏洞程序 Embedding 的完整 Embedding 字典
    """
    # ---------------------- Step 1 ---------------------------
    # 有漏洞程序的文件的token字典，e.g.,
    # flaw_Dict{
    #     "CWE-1": [
    #         "_CVE-2011-2943.c 的 token sequence"
    #         "_CVE-2011-2944.c 的 token sequence"
    #         ]
    #     "CWE-2": [
    #         "_CVE-2011-2945.c 的 token sequence"
    #         "_CVE-2011-2946.c 的 token sequence"
    #     ]
    # }
    #
    flaw_Dict = {}  # 字典， key：cwe类型，value：每类CWE所包含的各cve程序的token序列
    for k, v in result.items():
        # 键 key：General, VLC, Asterisk, FFmpeg, LibPNG, LibTIFF, Pidgin
        if k == "General":
            continue
        info = v
        for cwe, vv in info.items():
            # cwe: e.g. "CWE-134", "CWE-119"...
            # vv: "count" "files"..
            if flaw_Dict.get(cwe) == None:  # .get()用于返回指定键的值
                flaw_Dict[cwe] = []
            for j in vv["files"]:
                # 对于“files”中的信息，即每一个cve文件。 e.g.：CVE-2008-1489.c
                for i in range(len(flawFile_id_list)):
                    if j == flawFile_id_list[i]:
                        # 判断数据集中是否有该json文件中记录的这个cve文件
                        flaw_Dict[cwe].append(flawFiles_list[i])
                        break
    # print("CWE,\tNumber")
    # for k, v in flaw_Dict.items():
    #     print(k, len(v))
    #     # 输出有哪些CWE类型，每一类有几个cve程序的token sequence。
    # print("=========================")

    # ---------------------- Step 2 ---------------------------
    # 有漏洞程序的 Embedding 字典. e.g.,
    #     flaw_embed_dict{
    #         "CWE-1": [
    #             "_CVE-2011-2943.c 的 Embedding (1000-d token 数字，非向量矩阵)",
    #             "_CVE-2011-2944.c 的 Embedding (1000-d token 数字，非向量矩阵)"
    #         ],
    #         "CWE-2": [
    #             "_CVE-2011-2945.c 的 Embedding (1000-d token 数字，非向量矩阵)",
    #             "_CVE-2011-2946.c 的 Embedding (1000-d token 数字，非向量矩阵)"
    #         ]
    #     }
    #
    flaw_embed_dict = {}
    # TODO: 该部分在 helper_for_metaL.py 中 Helper 类的 tokenization 函数中有定义，后期可以优化！！！！！！！
    # 这个文件应该是是运行Word_to_vec_embedding.py后生成的 tokenizer
    # tokenizer = LoadPickleData("result/tokenizer.pickle")
    # 各个word的索引index，如 "something to eat", something: 1, to: 2, eat: 3
    word_index = tokenizer.word_index
    for k, v in flaw_Dict.items():
        cur_seq = tokenizer.texts_to_sequences(v)  # 把原来的代码文本序列转化为对应的数字序列
        padded_seq = pad_sequences(
            cur_seq, maxlen=1000, padding='post'
        )  # 不足1000的在后面补零，超了的截断
        if flaw_embed_dict.get(k) == None:
            flaw_embed_dict[k] = []
        for i in range(len(padded_seq)):
            flaw_embed_dict[k].append(padded_seq[i])

    # ---------------------- Step 3 ---------------------------
    # 包含无漏洞程序 Embedding 的完整 Embedding 字典
    # all_embed_dict =
    #     {
    #         "CWE-1": [
    #             "_CVE-2011-2943.c 的 Embedding (1000-d token 数字，非向量矩阵)",
    #             "_CVE-2011-2944.c 的 Embedding (1000-d token 数字，非向量矩阵)"
    #         ],
    #         "CWE-2": [
    #             "_CVE-2011-2945.c 的 Embedding (1000-d token 数字，非向量矩阵)",
    #             "_CVE-2011-2946.c 的 Embedding (1000-d token 数字，非向量矩阵)"
    #         ],
    #        ...
    #         "Benign":[
    #             "program_1.c 的 Embedding (1000-d token 数字，非向量矩阵)",
    #             "program_2.c 的 Embedding (1000-d token 数字，非向量矩阵)"
    #         ],
    #     }
    #
    all_embed_dict = flaw_embed_dict
    cur_seq = tokenizer.texts_to_sequences(files_list)
    padded_seq = pad_sequences(cur_seq, maxlen=1000, padding='post')
    all_embed_dict["Benign"] = padded_seq

    if list(all_embed_dict.keys())[-1] != "Benign":
        print("Error! The last key should be Benign!")

    return flaw_Dict, flaw_embed_dict, all_embed_dict


def SARD_all_embed_dict(json, data_path, tokenizer):
    """
    SARD 数据集
    获取 all_embed_dict 的方法
    注意无漏洞的key是 Benign
    把 benign 放最后
    """
    token_dict = {}
    all_embed_dict = {}
    for k in json.keys():
        if k == "General":
            continue
        for file in json[k]["files"]:
            token_file = tokenize(data_path, file)
            try:
                token_dict[k].append(token_file)
            except KeyError:
                token_dict[k] = []
                token_dict[k].append(token_file)
    # 将 Benign 放最后
    value = token_dict.pop('Benign')
    token_dict['Benign'] = value
    if list(token_dict.keys())[-1] != "Benign":
        print("Error! The last key should be Benign!")

    # TODO: 这里需要用数据集自己训练的word2vec模型
    # tokenizer = LoadPickleData("result/tokenizer.pickle")
    # 各个word的索引index，如 "something to eat", something: 1, to: 2, eat: 3
    word_index = tokenizer.word_index
    for k, v in token_dict.items():
        cur_seq = tokenizer.texts_to_sequences(v)  # 把原来的代码文本序列转化为对应的数字序列
        padded_seq = pad_sequences(cur_seq, maxlen=1000, padding='post')
        all_embed_dict[k] = padded_seq
    return all_embed_dict


def SARD_4_all_embed_dict(json, data_path, tokenizer):
    """
    SARD_4 数据集
    获取 all_embed_dict (每个文件的代码先变成token sequence 后把每个token映射成一个数字，每个file的代码用1000个数字表示)
    注意无漏洞的key是 Benign
    把 benign 放最后
    """
    token_dict = {}
    all_embed_dict = {}

    # 将数据集中的文件代码转换为token序列，保存成字典形式
    # {'API function call':{
    #   'CWE1':[token1, token2, ...],
    #   'CWE2':[token1, token2, ...],
    #   ...
    #   'CWEN':[token1, token2, ...]
    #   'Benign':[token1, token2, ...],
    #   },
    # 'Arithmetic expression':{
    #   'CWE1':[token1, token2, ...],
    #   'CWE2':[token1, token2, ...],
    #   ...
    #   'CWEN':[token1, token2, ...]
    #   'Benign':[token1, token2, ...],
    #   },
    # 'Array usage':{
    #   'CWE1':[token1, token2, ...],
    #   'CWE2':[token1, token2, ...],
    #   ...
    #   'CWEN':[token1, token2, ...]
    #   'Benign':[token1, token2, ...],
    #   },
    # 'Pointer usage':{
    #   'CWE1':[token1, token2, ...],
    #   'CWE2':[token1, token2, ...],
    #   ...
    #   'CWEN':[token1, token2, ...]
    #   'Benign':[token1, token2, ...],
    #   },
    # }
    for k in json.keys():
        if k == "General":
            continue
        token_dict[k] = {}
        new_data_path = os.path.join(data_path, k)
        for k2 in json[k].keys():  # 'Benign', 'CWE1', ..., 'CWEN'
            for file in json[k][k2]["files"]:
                token_file = tokenize(new_data_path, file)
                try:
                    token_dict[k][k2].append(token_file)
                except KeyError:
                    token_dict[k][k2] = []
                    token_dict[k][k2].append(token_file)
        # 将 Benign 放最后
        value = token_dict[k].pop('Benign')
        token_dict[k]['Benign'] = value
        if list(token_dict[k].keys())[-1] != "Benign":
            print("Error! The last key should be Benign!")

    # 这里需要用数据集自己训练的word2vec模型
    # 各个word的索引index，如 "something to eat", something: 1, to: 2, eat: 3
    word_index = tokenizer.word_index
    for syntaxfeatures, cwes in token_dict.items():
        all_embed_dict[syntaxfeatures] = {}
        for cwe, files in cwes.items():
            cur_seq = tokenizer.texts_to_sequences(files)
            padded_seq = pad_sequences(cur_seq, maxlen=1000, padding='post')
            all_embed_dict[syntaxfeatures][cwe] = padded_seq
    return all_embed_dict


def GenerateLabels(all_embed_dict):
    """
    Add labels to the candidate datasets' dictionaries.
    labeled_dict =
    {
        'CWE-1':{
            'Embeddings':[],
            'Labels':[],
        },
        ...
        'Benign':{
            'Embeddings':[],
            'Labels':[],
        }
    }
    """
    labeled_dict = {}
    for k, v in all_embed_dict.items():
        # k: Non_Vulnerability, CWE-1, CWE-2,...;   v: embedding1, embedding2,...
        labeled_dict[k] = {}
        labeled_dict[k]["Embeddings"] = v
        labeled_dict[k]["Labels"] = []

        if k == "Benign":
            labeled_dict[k]["Labels"] = [0 for i in range(len(v))]  # 对于无漏洞的样本，每个标签都是0
        else:
            labeled_dict[k]["Labels"] = [1 for i in range(len(v))]

    #  识别字典labeled_dict最后一个key是不是“benign”
    if list(labeled_dict.keys())[-1] != "Benign":
        print("Error! The last key should be Benign!")
    # print("Add labels to the dict:")
    # print(dict)
    # print("=============================")
    return labeled_dict


def GenerateLabels_SARD_4(all_embed_dict):
    """使用于 SARD_4 数据集结构的 GenerateLabels"""
    labeled_dict = {}
    for syntaxfeatures, cwes in all_embed_dict.items():
        labeled_dict[syntaxfeatures] = {}
        for cwe, num_tokens in cwes.items():
            labeled_dict[syntaxfeatures][cwe] = {}
            labeled_dict[syntaxfeatures][cwe]["Embeddings"] = num_tokens
            labeled_dict[syntaxfeatures][cwe]["Labels"] = []
            if cwe == "Benign":
                labeled_dict[syntaxfeatures][cwe]["Labels"] = [
                    0 for i in range(len(num_tokens))
                ]
            else:
                labeled_dict[syntaxfeatures][cwe]["Labels"] = [
                    1 for i in range(len(num_tokens))
                ]
    return labeled_dict


# # ============================================  2. 元学习训练、测试阶段数据集划分 ====================================================
# """
# 按0.8:0.2 划分训练、验证集
#     # {
#     #   'CWE1':{
#             'Embeddings':[],
#             'Labels':[],
#             'train_x':[],
#             'train_y':[],
#             'vali_x':[],
#             'vali_y':[],
#             },
#         ...
#         ,
#         'Non_Vulnerability':{
#         ...
#         }
#     # }
# """


# def sep_train_vali(dict, flag):
#     # 处理有漏洞样本
#     if flag == 1:
#         for k, v in dict.items():  # k: CWE types; v: embeddings:[], labels:[]
#             dict[k]["train_x"] = []
#             dict[k]["vali_x"] = []
#             dict[k]["train_y"] = []
#             dict[k]["vali_y"] = []
#             (
#                 dict[k]["train_x"],
#                 dict[k]["vali_x"],
#                 dict[k]["train_y"],
#                 dict[k]["vali_y"],
#             ) = train_test_split(
#                 v["Embeddings"], v["Labels"], test_size=0.2, random_state=445
#             )
#     # 处理无漏洞样本
#     elif flag == 0:
#         tmp_train_x = []
#         tmp_train_y = []
#         train_x = []
#         train_y = []
#         vali_x = []
#         vali_y = []
#         test_x = []
#         test_y = []
#         dict["Non_Vulnerability"]["train_x"] = []
#         dict["Non_Vulnerability"]["train_y"] = []
#         dict["Non_Vulnerability"]["vali_x"] = []
#         dict["Non_Vulnerability"]["vali_y"] = []
#         dict["Non_Vulnerability"]["test_x"] = []
#         dict["Non_Vulnerability"]["test_y"] = []

#         tmp_train_x, test_x, tmp_train_y, test_y = train_test_split(
#             dict["Non_Vulnerability"]["Embeddings"],
#             dict["Non_Vulnerability"]["Labels"],
#             test_size=0.2,
#             random_state=445,
#         )
#         train_x, vali_x, train_y, vali_y = train_test_split(
#             tmp_train_x, tmp_train_y, test_size=0.25, random_state=445
#         )
#         dict["Non_Vulnerability"]["train_x"] = train_x
#         dict["Non_Vulnerability"]["train_y"] = train_y
#         dict["Non_Vulnerability"]["vali_x"] = vali_x
#         dict["Non_Vulnerability"]["vali_y"] = vali_y
#         dict["Non_Vulnerability"]["test_x"] = test_x
#         dict["Non_Vulnerability"]["test_y"] = test_y

#     # print("Dict with train and vali:")
#     # print("----------------------------")
#     # print(dict)
#     # print("==============================")

#     return dict


# """
# 数据集初步划分
# Candidate sub-datasets for meta-training and meta-testing phases.
# Dataset: Data_six
# ---------------------------------------------------------
# 有漏洞样本：
#     待选的meta-training set:    样本数量不小于10的漏洞类型
#     待选的meta-testing set:     样本数量大于1且小于10的漏洞类型
#     return                     candi_meta_train, candi_meta_test
# 无漏洞样本:
#     按0.6:0.2:0.2划分

# 返回：
# candi_Data_six ={
#     "meta_training":{
#         "CWE-1":{
#             "train_x":[],
#             "train_y":[],
#             "vali_x":[],
#             "vali_y":[],
#         },
#         "CWE-2":{
#             "train_x":[],
#             "train_y":[],
#             "vali_x":[],
#             "vali_y":[],
#         },
#         ...
#         ,
#         "Non_Vulnerability":{
#             "train_x":[],
#             "train_y":[],
#             "vali_x":[],
#             "vali_y":[],
#         },
#     },
#     "meta_testing":{
#         "CWE-3":{
#             "test_x":[],
#             "test_y":[],
#         },
#         "CWE-4":{
#             "test_x":[],
#             "test_y":[],
#         },
#         ...
#         ,
#         "Non_Vulnerability":{
#             "test_x":[]
#             "test_y":[]
#         }
#     }
# }
# """


# def candidate_Data_six(labeled_dict):
#     candi_Data_six = {}
#     candi_Data_six["meta_training"] = {}
#     candi_Data_six["meta_testing"] = {}

#     # 有漏洞样本
#     flag = 1
#     vul_dict = {}
#     for k in labeled_dict.keys():
#         if k == "Non_Vulnerability":
#             continue
#         else:
#             vul_dict[k] = {}
#             vul_dict[k] = labeled_dict[k]

#     vul_candi_meta_train = {}

#     for k in vul_dict.keys():
#         if len(vul_dict[k]["Embeddings"]) >= 10:
#             vul_candi_meta_train[k] = {}
#             vul_candi_meta_train[k] = vul_dict[k]
#         elif len(vul_dict[k]["Embeddings"]) < 10 and len(vul_dict[k]["Embeddings"]) > 1:
#             candi_Data_six["meta_testing"][k] = {}
#             candi_Data_six["meta_testing"][k]["test_x"] = []
#             candi_Data_six["meta_testing"][k]["test_y"] = []
#             candi_Data_six["meta_testing"][k]["test_x"] = vul_dict[k]["Embeddings"]
#             candi_Data_six["meta_testing"][k]["test_y"] = vul_dict[k]["Labels"]

#     # 删除候选 meta-train 中的 NVD-CWE-other 和 NVD-CWE-noinfo 类
#     del vul_candi_meta_train['NVD-CWE-Other']
#     del vul_candi_meta_train['NVD-CWE-noinfo']

#     # 按0.8:0.2 划分训练验证集
#     vul_candi_meta_train_v2 = sep_train_vali(vul_candi_meta_train, flag)
#     # 存入字典candi_Data_six
#     for k, v in vul_candi_meta_train_v2.items():
#         candi_Data_six["meta_training"][k] = {}
#         for vv in v:
#             if vv == "train_x" or vv == "train_y" or vv == "vali_x" or vv == "vali_y":
#                 candi_Data_six["meta_training"][k][vv] = []
#                 candi_Data_six["meta_training"][k][vv] = vul_candi_meta_train_v2[k][vv]

#     # 无漏洞样本
#     flag = 0
#     benign_dict = {}
#     benign_dict["Non_Vulnerability"] = {}
#     benign_dict["Non_Vulnerability"] = labeled_dict["Non_Vulnerability"]

#     benign_dict_v2 = sep_train_vali(benign_dict, flag)
#     candi_Data_six["meta_training"]["Non_Vulnerability"] = {}
#     candi_Data_six["meta_training"]["Non_Vulnerability"]["train_x"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["train_x"]
#     candi_Data_six["meta_training"]["Non_Vulnerability"]["train_y"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["train_y"]
#     candi_Data_six["meta_training"]["Non_Vulnerability"]["vali_x"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["vali_x"]
#     candi_Data_six["meta_training"]["Non_Vulnerability"]["vali_y"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["vali_y"]

#     candi_Data_six["meta_testing"]["Non_Vulnerability"] = {}
#     candi_Data_six["meta_testing"]["Non_Vulnerability"]["test_x"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["test_x"]
#     candi_Data_six["meta_testing"]["Non_Vulnerability"]["test_y"] = benign_dict_v2[
#         "Non_Vulnerability"
#     ]["test_y"]

#     return candi_Data_six


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vectorize all the programs in the given dataset.'
    )
    # 待处理数据集的名称，如果是 Data_six 或 SARD 则不需要额外处理，数据集路径和json文件路径是写死的。否则需要指定路径。
    parser.add_argument(
        '--Dataset', type=str, help='Name of the dataset. E.g.,Data_six, SARD, SARD_4'
    )
    # 数据集描述文件 JSON file 的路径
    # data_six 数据集   --json_path = "../Datasets/Data_six/Six_project_info/static.json"
    # DARD 数据集       --json_path = "../Datasets/SARD/Pre_SARD/CWE_info/sard_info.json"
    parser.add_argument(
        '--json_path',
        type=str,
        default=None,
        help='Path to the JSON file of the dataset.',
    )
    # 数据集所在路径
    # data_six 数据集   --data_path = "../Datasets/Data_six/Six_project/"
    # SARD 数据集       --data_path = "../Datasets/SARD/Pre_SARD/SARD_samples/"
    parser.add_argument(
        '--data_path', type=str, default=None, help='Path to the dataset.'
    )
    # 处理后数据集的保存路径
    parser.add_argument(
        '--save_path',
        type=str,
        default='../Datasets/',
        help='Path to save the processed dataset.',
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Path to the tokenizer.',
    )
    paras = parser.parse_args()

    # 加载数据集的 JSON 文件
    # 加载数据集
    if paras.Dataset == "Data_six":
        json_path = "../Datasets/Data_six/Six_project_info/static.json"
        # data_path = "../Datasets/Data_six/Six_project/"
        # 改为去注释且向量化后的数据集
        data_path = "../Datasets/Data_six/Pre_Six_project/symbols"
        save_path = "../Datasets/Data_six/"
        tokenizer_path = "../w2v/Data_six/tokenizer.pickle"

    # 这里用SARD训练，测试用Data_six
    # TODO: 老版本待确认
    elif paras.Dataset == "Data_six_only_for_test":
        json_path = "Datasets/Data_six/Six_project_info/static.json"
        data_path = "Datasets/Data_six/Pre_Six_project/symbols"
        save_path = "Datasets/Data_six/"
        tokenizer_path = "w2v/SARD/tokenizer.pickle"

    elif paras.Dataset == "SARD":
        json_path = "../Datasets/SARD/Pre_SARD/CWE_info/sard_info.json"
        data_path = "../Datasets/SARD/Pre_SARD/SARD_samples/"
        save_path = "../Datasets/SARD/"
        tokenizer_path = "../w2v/SARD/tokenizer.pickle"

    elif paras.Dataset == "SARD_4":
        json_path = "../Datasets/SARD_4/Program_Samples/CWE_info/sard_info.json"
        data_path = "../Datasets/SARD_4/Pre_Program_Samples/symbols"
        save_path = "../Datasets/SARD_4"
        tokenizer_path = "../w2v/SARD_4/tokenizer.pickle"

    elif paras.json_path != None and paras.data_path != None:
        print("Customized dataset information Loaded!")
        json_path = paras.json_path
        data_path = paras.data_path
        save_path = paras.save_path
        tokenizer_path = paras.tokenizer_path

    else:
        print(
            "Error! If you want to preprocess system supported datasets (Data_six and SARD), please do this:"
        )
        print("\tpython prep.py --dataset Data_six")
        print("\tor")
        print("\tpython prep.py --dataset SARD")
        print("\tor")
        print("\tpython prep.py --dataset SARD_4")
        print()

        print(
            "If you want to preprocess other datasets, please provide the path of the dataset, tokenizer, and its JSON file using parameters --data_path, --tokenizer_path, and --json_path. E.g.,:"
        )
        print(
            "\tpython prep.py --data_path <PATH to your Dataset> --tokenizer_path <PATH to the tokenizer obtained from Word_to_vec_embedding.py> --json_path <PATH to the JSON file> (--save_path <PATH to save processed dataset>)"
        )
        exit(1)

    if tokenizer_path == None:
        print(
            "Error! Please provide the path of the tokenizer unless using system supported dataset 'SARD_4', 'SARD' and 'Data_six'!"
        )
        exit(1)

    result = load_dataset_json(json_path)

    if paras.Dataset == "Data_six" or paras.Dataset == "Data_six_only_for_test":
        print("Processing Data_six...")
        tokenizer = LoadPickleData(tokenizer_path)

        # 获取有、无漏洞程序名及token序列列表，为后续向量化做准备。
        files_list, file_list_id, flawFiles_list, flawFile_id_list = get_files_and_ids(
            data_path
        )
        # 获取有、无漏洞文件的 token及embedding字典
        flaw_Dict, flaw_embed_dict, all_embed_dict = get_Dicts(
            result,
            files_list,
            file_list_id,
            flawFiles_list,
            flawFile_id_list,
            tokenizer,
        )
    elif paras.Dataset == "SARD":
        print("Processing SARD...")
        tokenizer = LoadPickleData(tokenizer_path)
        all_embed_dict = SARD_all_embed_dict(result, data_path, tokenizer)
    elif paras.Dataset == "SARD_4":
        print("Processing SARD_4...")
        tokenizer = LoadPickleData(tokenizer_path)
        all_embed_dict = SARD_4_all_embed_dict(result, data_path, tokenizer)
        # Check
        # print("the first layer of keys in all_embed_dict:")
        # print(all_embed_dict.keys())
        # print("the second layer of keys in all_embed_dict:")
        # print("keys of {}:".format(list(all_embed_dict.keys())[0]))
        # for k in all_embed_dict[list(all_embed_dict.keys())[0]].keys():
        #     print(k)
        # print("elements of the first cwe:")
        # cwes = list(all_embed_dict["API function call"].keys())
        # print(all_embed_dict["API function call"][cwes[0]])

    else:
        print(
            "User defined dataset should satisfy the preprocess standard like the processed SARD in our project."
        )
        print(
            "It requires to provide a JSON file that has the exact same format with SARD."
        )
        print("Please check the format to avoid any potential mistake.")
        print("Besides, it needs user's own tokenizer.")
        print(
            "We are going to use the method for precessing SARD to process the dataset."
        )
        print("Processing User Defined Dataset...")
        tokenizer = LoadPickleData(tokenizer_path)

        all_embed_dict = SARD_all_embed_dict(result, data_path, tokenizer)

    # 给 all_embed_dict 加上各个样本的标签
    if paras.Dataset == "SARD_4":
        labeled_dict = GenerateLabels_SARD_4(all_embed_dict)
    else:
        labeled_dict = GenerateLabels(all_embed_dict)

    # # 若是 Data_six 数据集，则可以直接划分训练、验证、测试集。其中样本数量大于10的CWE类型被用于meta-training，小于10的用于meta-testing
    # # 划分 Meta-training， meta-testing 阶段的训练、验证、测试集
    # if paras.Dataset == "Data_six":
    #     candi_Data = candidate_Data_six(labeled_dict)
    # # 若是 SARD 数据集，则直接交付 labeled dict，等到后续在 helper 中选定了实验方案再处理。
    # elif paras.Dataset == "SARD":
    #     candi_Data = labeled_dict
    # else:
    #     candi_Data = labeled_dict
    #     print("User defined dataset has been processed into embeddings.")

    candi_Data = labeled_dict
    print(candi_Data)

    # 把处理好的数据集保存下来
    save_path = os.path.join(save_path, "processed_data")
    with open(save_path, 'wb') as f:
        pickle.dump(candi_Data, f)
    print("The processed dataset has been dumped into: " + save_path)
    print("Done!")
    print()
