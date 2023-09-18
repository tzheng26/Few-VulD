# -*- coding: utf-8 -*-

import pickle
import csv
import os
import pandas as pd
import random
import numpy as np
import re


def SplitCharacters(str_to_split):
    """
    函数功能：   源码 --> token sequence，分割标点
    函数输入：   “str_to_split” —— 待序列化的程序字符串。
    函数操作：   1）把标点前后都加空格
                2）按空格分割各个token
                3）然后把各个token重新连接成 token 序列。
    函数返回值： “str_list_str” —— token sequence

    Separate '(', ')', '{', '}', '*', '/', '+', '-', '=', ';', '[', ']' characters.
    """
    # Character_sets = ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',']

    str_list_str = ''

    if '(' in str_to_split:
        str_to_split = str_to_split.replace(
            '(', ' ( '
        )  # Add the space before and after the '(', so that it can be split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)

    if ')' in str_to_split:
        str_to_split = str_to_split.replace(
            ')', ' ) '
        )  # Add the space before and after the ')', so that it can be split by space.
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


def SavedPickle(path, file_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(file_to_save, handle)


def Save3DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(list_to_save)


def Save2DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_save)


def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)


def LoadPickleData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def removeSemicolon(input_list):
    """Remove ';' from the list."""
    new_list = []
    for line in input_list:
        new_line = []
        for item in line:
            if item != ';' and item != ',':
                new_line.append(item)
        new_list.append(new_line)

    return new_list


def ProcessList(list_to_process):
    """Split the elements such as "const int *" into "const", "int" and "*" """
    token_list = []
    for sub_list_to_process in list_to_process:
        sub_token_list = []
        if len(sub_list_to_process) != 0:
            for each_word in sub_list_to_process:  # Remove the empty row
                each_word = str(each_word)
                sub_word = each_word.split()
                for element in sub_word:
                    sub_token_list.append(element)
            token_list.append(sub_token_list)
    return token_list


# TODO: 得把.c 改成 .c or .cpp
def getCFilesFromText(path):
    """
    读取目录下的.c文件
    返回：
        files_list      -- 文件内容列表(去掉所有空格、换行符、tab后的token化列表)
        file_id_list    -- c、cpp文件名列表
    """
    files_list = []
    file_id_list = []
    if os.path.isdir(path):
        for fpath, dirs, fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1] == '.c':  # 有问题，SARD数据集还有 cpp 文件
                    file_id_list.append(f)
                if os.path.splitext(f)[1] == '.c':  # 可以去掉！
                    # print(os.path.splitext(f)[0])
                    # print(fs)
                    with open(
                        fpath + os.sep + f, encoding='utf-8', errors="ignore"
                    ) as file:
                        lines = file.readlines()
                        file_list = []
                        for line in lines:
                            if (
                                line != ' ' and line != '\n'
                            ):  # Remove space and line-change characters 删除空行
                                sub_line = line.split()
                                new_sub_line = []
                                for element in sub_line:
                                    new_element = SplitCharacters(element)
                                    new_sub_line.append(
                                        new_element
                                    )  # 将每一行的所有词、标点分开成一个个token
                                new_line = ' '.join(new_sub_line)
                                file_list.append(new_line)
                        new_file_list = ' '.join(file_list)
                        split_by_space = new_file_list.split()
                    files_list.append(split_by_space)
        return files_list, file_id_list


# 得把.c 改成 .c or .cpp
def getCFilesFromTextRevise(path):
    """
    程序输入:
    path —— 数据集所在路径。e.g. Data_six/Six_project/

    程序输出：
    1）把有漏洞的.c文件放入           flawFile_id_list[]列表
    2）把无漏洞的.c文件放入           file_id_list[]列表
    3）各个有漏洞.c程序token序列的列表   flawFiles_list
    4）各个无漏洞.c程序token序列的列表   files_list
    """
    files_list = []
    file_id_list = []
    flawFiles_list = []
    flawFile_id_list = []
    if os.path.isdir(path):
        # print(fs)
        for fpath, dirs, fs in os.walk(path):
            for f in fs:
                flawFlag = False
                if re.findall('CVE-(\d+)', os.path.splitext(f)[0]) or re.findall(
                    'cve-(\d+)', os.path.splitext(f)[0]
                ):
                    # print(f)
                    flawFlag = True
                if os.path.splitext(f)[1] == '.c':
                    if flawFlag:
                        flawFile_id_list.append(f)
                    else:
                        file_id_list.append(f)
                if os.path.splitext(f)[1] == '.c':
                    if os.path.splitext(f)[0] == "puzzle_pce.c_puzzle_rotate_pce":
                        with open(
                            fpath + os.sep + f, encoding='utf-8', errors='ignore'
                        ) as file:
                            lines = file.readlines()
                            file_list = []
                            for line in lines:
                                if (
                                    line != ' ' and line != '\n'
                                ):  # Remove sapce and line-change characters
                                    sub_line = line.split()
                                    new_sub_line = []
                                    for element in sub_line:
                                        new_element = SplitCharacters(element)
                                        new_sub_line.append(new_element)
                                    new_line = ' '.join(new_sub_line)
                                    file_list.append(new_line)
                            new_file_list = ' '.join(file_list)
                            split_by_space = new_file_list.split()
                        files_list.append(split_by_space)
                        continue
                    with open(fpath + os.sep + f, encoding='utf-8') as file:
                        lines = file.readlines()
                        file_list = []
                        for line in lines:
                            if (
                                line != ' ' and line != '\n'
                            ):  # Remove sapce and line-change characters
                                sub_line = line.split()
                                new_sub_line = []
                                for element in sub_line:
                                    new_element = SplitCharacters(element)
                                    new_sub_line.append(new_element)
                                new_line = ' '.join(new_sub_line)
                                file_list.append(new_line)
                        new_file_list = ' '.join(file_list)
                        split_by_space = new_file_list.split()
                    if flawFlag:
                        flawFiles_list.append(split_by_space)
                    else:
                        files_list.append(split_by_space)
                    # files_list.append(split_by_space)
        return files_list, file_id_list, flawFiles_list, flawFile_id_list


# 需要修改！！！！！！！！！！！！！！！！！！
def GenerateLabels(input_arr):
    """
    根据文件名给标签。一个列表
    Data labels are generated based on the sample IDs. All the vulnerable function samples are named with CVE IDs.
    """

    temp_arr = []
    for func_id in input_arr:
        temp_sub_arr = []
        if "cve" in func_id or "CVE" in func_id:
            temp_sub_arr.append(1)
        else:
            temp_sub_arr.append(0)
        temp_arr.append(temp_sub_arr)
    return temp_arr


######################################################################################################
"""
MAML 数据集划分
"""


# 重点 应该是在预处理的训练、测试数据集的基础上，进一步划分每个元学习batch的训练、测试数据。
class MAMLDataLoader:
    """
    MAML数据读取
    """

    def __init__(self, data_x, data_y):
        self.file_list = data_x
        self.steps = 50  # steps 的解释
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.steps

    def get_one_metabatch(self, batch_size, k):
        self.meta_batch_size = batch_size
        k_shot = k
        query_num = k_shot

        batch_support_code = []
        batch_support_label = []
        batch_query_code = []
        batch_query_label = []

        # 对于一个batch中的每个任务，T0,T1,T2,T3
        for index in range(self.meta_batch_size):
            support_code = []
            support_label = []
            query_code = []
            query_label = []

            # 从选中的CWE类型的所有样本中，随机选2k个，k个做sample set，k个做query set
            vul_ind = random.sample(range(len(self.data_x[index])), k_shot + query_num)
            # 从无漏洞的所有样本中，随机选2k个。
            # data_x[-1] 是无漏洞 benign 类型
            non_vul_ind = random.sample(range(len(self.data_x[-1])), k_shot + query_num)

            support_data = []
            query_data = []
            # support data，query data 有漏洞、无漏洞
            for i in vul_ind[:k_shot]:
                support_data.append((self.data_x[index][i], self.data_y[index][i]))
            for i in non_vul_ind[:k_shot]:
                support_data.append((self.data_x[-1][i], self.data_y[-1][i]))
            for i in vul_ind[k_shot:]:
                query_data.append((self.data_x[index][i], self.data_y[index][i]))
            for i in non_vul_ind[k_shot:]:
                query_data.append((self.data_x[-1][i], self.data_y[-1][i]))

            # 分出 support、query data 的样本、标签
            random.shuffle(support_data)
            for i in range(len(support_data)):
                support_code.append(support_data[i][0])
                support_label.append(support_data[i][1])
            random.shuffle(query_data)
            for i in range(len(query_data)):
                query_code.append(query_data[i][0])
                query_label.append(query_data[i][1])

            np.array(support_code)
            np.array(support_label)
            np.array(query_code)
            np.array(query_label)

            batch_support_code.append(support_code)
            batch_support_label.append(support_label)
            batch_query_code.append(query_code)
            batch_query_label.append(query_label)

        return (
            np.array(batch_support_code),
            np.array(batch_support_label),
            np.array(batch_query_code),
            np.array(batch_query_label),
        )
