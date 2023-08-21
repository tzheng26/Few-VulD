# -*- coding: utf-8 -*-
import os
import sys
import pickle, random
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras import optimizers, utils
from tensorflow.compat.v1.keras import losses
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.compat.v1.keras.callbacks import TensorBoard, CSVLogger
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.compat.v1.keras.models import load_model
from sklearn.model_selection import train_test_split
from src.DataLoader import (
    MAMLDataLoader,
    getCFilesFromText,
    GenerateLabels,
    LoadPickleData,
    SavedPickle,
    ListToCSV,
    getCFilesFromTextRevise,
)
from tensorflow import test
from src.models.Deep_model import Deep_model
from src.models.textCNN import textCNN
import time
import json


class Helper:
    '''Super class Solver for all kinds of tasks'''

    def __init__(self, config, paras):
        self.config = config
        self.paras = paras

        now = datetime.datetime.now()

        """config 定义的参数"""
        # path of the pre-trained tokenizer and w2v model
        self.tokenizer_path = self.config['training_settings']['tokenizer_path']
        self.embed_path = self.config['training_settings']['embedding_model_path']
        self.verbose("-------------------------------------------------------")
        self.verbose("Word2vec Info:")
        self.verbose("Path to the tokenizer: " + self.tokenizer_path)
        self.verbose("Path to the embeding model: " + self.embed_path)

        # By default, transfer tokens to 100-d vectors
        self.embed_dim = self.config['model_settings']['model_para']['embedding_dim']

        # # 模型保存路径
        # # TODO: 没用上，看和下面的怎么合并
        # if not os.path.exists(self.config['training_settings']['model_save_path']):
        #     os.makedirs(self.config['training_settings']['model_save_path'])

        # 保存训练模型的路径和名称
        if not os.path.exists(self.config['training_settings']['model_save_path']):
            os.makedirs(self.config['training_settings']['model_save_path'])
        self.model_name = (
            self.config['training_settings']['model_save_path']
            + os.sep
            + self.config['model_settings']['model']
        )

        # date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # self.model_name = (
        #     "Models/"
        #     + self.config['model_settings']['model']
        #     + "/"
        #     + date
        #     + "/mamlMetaFormal"
        #     + self.config['model_settings']['model']
        # )
        # # check model save folder
        # if not os.path.exists(
        #     "Models/" + self.config['model_settings']['model'] + "/" + date
        # ):
        #     os.makedirs("Models/" + self.config['model_settings']['model'] + "/" + date)
        # self.verbose("-------------------------------------------------------")
        # self.verbose(
        #     "Model save path: "
        #     + "Models/"
        #     + self.config['model_settings']['model']
        #     + "/"
        #     + date
        # )

        # 一个batch几个任务 (即，元学习中使用的CWE类型数量)
        self.batch_s = self.config['training_settings']['network_config']['batch_size']
        self.verbose("-------------------------------------------------------")
        self.verbose("Batch size: " + str(self.batch_s))

        """para 定义的参数"""
        # The output path of the trained network model. 输出到 result/里
        # TODO: 没用上,后续可以改成和上面设置的模型保存路径一致
        # if not os.path.exists(self.paras.output_dir):
        #     os.makedirs(self.paras.output_dir)

        # 日志
        if not os.path.exists(self.paras.logdir):
            os.makedirs(self.paras.logdir)

        # 载入预处理好的数据集，输出数据集所在目录：
        if self.paras.dataset != None:
            # 数据集名
            if "Data_six" in self.paras.dataset:
                self.type_dataset = "Data_six"
            elif "SARD_4" in self.paras.dataset:
                self.type_dataset = "SARD_4"
            elif "SARD" in self.paras.dataset:
                self.type_dataset = "SARD"
            else:
                self.type_dataset = "Not_Data_six"
            # 读取数据集
            with open(self.paras.dataset, "rb") as f:
                self.candi_data = pickle.load(f)
            self.verbose("-------------------------------------------------------")
            self.verbose("Preprocessed Data Loaded!")
            self.verbose("Path to the dataset: " + self.paras.dataset)
            self.verbose("Type of dataset: " + self.type_dataset)
            # 数据集包含的漏洞类型（不适用于SARD_4）
            if self.type_dataset != "SARD_4":
                list_CWE_types = list(self.candi_data.keys())[:-1]  # 最后一个key是benign
                self.verbose(
                    "The dataset has totally "
                    + str(len(list_CWE_types))
                    + " types of CWEs, which are:"
                )
                self.verbose(list_CWE_types)

        # 根据 main_meta.py 的参数，载入实验方案
        self.experiment_scheme = self.paras.experiment_scheme
        if (
            self.paras.train_or_test == None
        ):  # 如果是训练、测试就不输出实验方案了，因为参数里并不需要指定实验方案，如果输出则是默认的实验方案参数，容易误导
            self.verbose("-------------------------------------------------------")
            self.verbose("Experiment Scheme: " + self.experiment_scheme)

        # 设定随机种子
        nu = self.paras.seed
        random.seed(nu)

    """
    TODO: 可删除
    分割训练、验证、测试数据集；比例：0.6:0.2:0.2
    后面还要再看！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    """

    def patitionData(self, data_list_pad, data_list_id):
        test_size = self.config['training_settings']['dataset_config']['Test_set_ratio']
        validation_size = self.config['training_settings']['dataset_config'][
            'Test_set_ratio'
        ]

        # 调用 DataLoader.py 中的 GenerateLabels 函数，根据文件名中是否包含cve，分配标签 1 或 0
        data_list_label = GenerateLabels(data_list_id)
        # print(data_list_label[:5])

        # 从数据集中划分测试集。
        if not self.config['training_settings']['using_separate_test_set']:
            # The value of the seed for testing should be the same to that was used during the training phase.
            (
                train_vali_set_x,
                test_set_x,
                train_vali_set_y,
                test_set_y,
                train_vali_set_id,
                test_set_id,
            ) = train_test_split(
                data_list_pad,
                data_list_label,
                data_list_id,
                test_size=test_size,
                random_state=self.paras.seed,
            )
            (
                train_set_x,
                validation_set_x,
                train_set_y,
                validation_set_y,
                train_set_id,
                validation_set_id,
            ) = train_test_split(
                train_vali_set_x,
                train_vali_set_y,
                train_vali_set_id,
                test_size=validation_size,
                random_state=self.paras.seed,
            )

            tuple_with_test = (
                train_set_x,
                train_set_y,
                train_set_id,
                validation_set_x,
                validation_set_y,
                validation_set_id,
                test_set_x,
                test_set_y,
                test_set_id,
            )
            setattr(self, 'patitioned_data', tuple_with_test)
            return tuple_with_test
        else:
            (
                train_set_x,
                validation_set_x,
                train_set_y,
                validation_set_y,
                train_set_id,
                validation_set_id,
            ) = train_test_split(
                train_vali_set_x,
                train_vali_set_y,
                train_vali_set_id,
                test_size=validation_size,
                random_state=self.paras.seed,
            )
            tuple_without_test = (
                train_set_x,
                train_set_y,
                train_set_id,
                validation_set_x,
                validation_set_y,
                validation_set_id,
            )
            setattr(self, 'patitioned_data', tuple_without_test)
            return tuple_without_test

    """标准化输出显示"""

    def verbose(self, msg):
        '''Verbose function for print information to stdout'''
        if self.paras.verbose == 1:
            print('[INFO]', msg)

    """
    TODO: 看后面还要不要
    运行Word_to_vec_embedding.py 后生成的 tokenizer，实现了文本的序列化
    """

    def tokenization(self, data_list):
        # 这个文件应该是是运行Word_to_vec_embedding.py后生成的 tokenizer
        tokenizer = LoadPickleData(self.tokenizer_path)
        # 把原来的代码文本序列转化为对应的数字序列
        total_sequences = tokenizer.texts_to_sequences(data_list)
        # 各个word的索引index，如 "something to eat", something: 1, to: 2, eat: 3
        word_index = tokenizer.word_index

        return total_sequences, word_index

    """
    TODO: 后面也没用到
    补零
    补到 max_sequence_length (default 1000)
    最后应该每个程序是1000*100的向量，即1000个词，每个词用100维的向量表示。
    """

    def padding(self, sequences_to_pad):
        padded_seq = pad_sequences(
            sequences_to_pad,
            maxlen=self.config['model_settings']['model_para']['max_sequence_length'],
            padding='post',
        )
        return padded_seq

    """
    TODO: 看还要不要
    调用 DataLoader.py 中的 getCFilesFromText() 函数，获取 .c 文件名与文件内容列表
    应该需要更改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    """

    def loadData(self, data_path):
        '''Load data for training/validation'''
        self.verbose('Loading data from ' + os.getcwd() + os.sep + data_path + '....')
        total_list, total_list_id = getCFilesFromText(data_path)
        self.verbose("The length of the loaded data list is : " + str(len(total_list)))
        # print(total_list[0])
        # print(total_list_id[:5])
        return total_list, total_list_id

    """
    apply Embedding
    word_index 表示训练文本字典中的每个词对应的编号
    返回一个嵌入矩阵（数组）-- embedding_matrix: 
        词典中每个词的编号及对应的 100-d embedding vector
        len(word_index) + 1 行, self.embed_dim (100)列
    """

    def applyEmbedding(self, w2v_model, word_index):
        # a dictionary, key: word i.e. 'int', value: its corresponding 100 dimension embedding.
        embeddings_index = {}
        # Use the loaded w2v_model
        # TODO: 这里应该删除w2v_model文件中的第一行，第一行是词典的大小和维度，检查word vector 的大小是不是多了一个
        for line in w2v_model:
            if not line.isspace():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        w2v_model.close()
        self.verbose('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros(
            (len(word_index) + 1, self.embed_dim)
        )  # word_index: 词和其在字典中对应的 index; self.embed_dim = 100(default)
        for word, i in word_index.items():
            # 词对应的embedding向量
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    """
    载入word2vec
    applyEmbedding
    载入学习模型
    """

    def load_w2v_embed_model(self):
        # tokenizer 指运行 Word_to_vec_embedding.py 后生成的 tokenizer
        # tokenzier的路径应当在config.yaml文件中定义
        # TODO 这部分在父类中定义了单独的函数 tokenization（），应优化！！！！！！！！！
        tokenizer = LoadPickleData(self.tokenizer_path)
        # 各个word的索引index，如 "something to eat", something: 1, to: 2, eat: 3
        self.word_index = tokenizer.word_index
        word_index = self.word_index  # 每个词的数字表示

        # 载入word2vec模型
        self.verbose("-------------------------------------------------------")
        self.verbose("Loading trained Word2vec model. ")
        # 配置文件中的 embedding_model_path: "w2v/SARD/w2v_model.txt"
        w2v_model = open(self.embed_path)
        self.verbose("The trained word2vec model: ")
        self.verbose(w2v_model)

        # applyEmbedding:
        # 返回一个嵌入矩阵（数组）-- embedding_matrix:
        # 词典中每个词的编号及对应的 100-d embedding vector
        # len(word_index) + 1 行, self.embed_dim (100)列
        self.verbose("-------------------------------------------------------")
        self.verbose("Embedding Matrix:")
        embedding_matrix = self.applyEmbedding(w2v_model, word_index)  # 向量表示
        self.verbose("The embedding matrix of the token dictionary: ")
        print(embedding_matrix)
        self.verbose("")

        # 根据 config 的设定，载入相关模型。
        # Initialize the model class here.
        deep_model = Deep_model(self.config, word_index, embedding_matrix)
        test_CNN = textCNN(self.config)
        model_func = deep_model.meta_model  # 相关的模型
        # textCNN 应该暂不支持，需要特殊声明
        if self.config['model_settings']['model'] == 'textCNN':
            self.verbose(
                "Loading the " + self.config['model_settings']['model'] + " model."
            )
            model_func = test_CNN.buildModel(word_index, embedding_matrix)
        self.verbose("Model structure loaded.")
        model_func.summary()  # 应该是tensorflow的一个方法，能够保存训练过程以及参数分布图并在tensorboard显示

        return w2v_model, word_index, embedding_matrix, deep_model, model_func

    """
    实验方案1:  随机选择训练任务 (适用 Data_six)
    ------------------------------------------------------------------------------------
    训练任务:   从数据集 meta_training 的漏洞类型中随机选 meta_batch 类个CWE

    测试任务:   样本少的其他漏洞类型作为测试任务
    """

    def Random_Scheme(self, meta_batch, type_dataset):
        if type_dataset == "Data_six":
            threshold_1 = 10
            threshold_2 = 2
        else:
            threshold_1 = 100
            threshold_2 = 100

        # 候选CWE类型（样本数大于等于threshold_1）
        candi_train_CWEs = []
        # [:-1]的作用是排除最后一个 Benign 类型
        for k in list(self.candi_data.keys())[:-1]:
            num_single_cwe_samples = len(self.candi_data[k]["Embeddings"])
            if num_single_cwe_samples >= threshold_1:
                candi_train_CWEs.append(k)
        if type_dataset == "Data_six":
            try:
                candi_train_CWEs.remove('NVD-CWE-Other')
            except ValueError:
                pass
            try:
                candi_train_CWEs.remove('NVD-CWE-noinfo')
            except ValueError:
                pass

        # 从候选 CWE 类型中随机选 meta_batch 个作为训练任务
        if meta_batch > len(candi_train_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)
        Mtrain_CWE_types = random.sample(candi_train_CWEs, meta_batch)
        # self.verbose("-------------------------------------------------------")
        # self.verbose(
        #     "Randomly selected " + str(meta_batch) + " types of CWEs for meta-training:"
        # )
        # self.verbose(Mtrain_CWE_types)

        # 测试 CWE 类别 从数据集的所有漏洞类型中去掉用于训练的类型和样本数量少于threshold_2的类别
        Mtest_CWE_types = list(self.candi_data.keys())[:-1]

        for CWE in Mtrain_CWE_types:
            Mtest_CWE_types.remove(CWE)
        for CWE in Mtest_CWE_types:
            num_single_cwe_samples = len(self.candi_data[CWE]["Embeddings"])
            if num_single_cwe_samples < threshold_2:
                Mtest_CWE_types.remove(CWE)
        if type_dataset == "Data_six":
            try:
                Mtest_CWE_types.remove('NVD-CWE-Other')
            except ValueError:
                pass
            try:
                Mtest_CWE_types.remove('NVD-CWE-noinfo')
            except ValueError:
                pass
        # self.verbose("-------------------------------------------------------")
        # self.verbose("CWE types for meta-testing are:")
        # self.verbose(Mtest_CWE_types)

        return Mtrain_CWE_types, Mtest_CWE_types

    # TODO
    def Scheme2(self, meta_batch):
        """
        实验方案2:  用相近机理的CWE进行训练  1) 测试不同漏洞机理的CWE 或 2)相似机理CWE (适用 Data_six)
        ------------------------------------------------------------------------------------
        训练任务:   缓冲区溢出漏洞  CWE-119、125、787、20

        不同机理：
            测试任务a:  数字相关错误    CWE-369 (Divided By Zero), CWE-189 (Numberic Errors)
            测试任务b:  资源管理错误    CWE-399 (Resource Management Errors), 772 (Missing Release of Resource after Effective Lifetime)
            测试任务c:  指针类         CWE-476 (NULL Pointer Dereference)

        相似机理:
            测试任务d:  整数溢出       CWE-190 (Integer Overflow or Wraparound)
        """
        Mtrain_CWE_types = []
        Mtrain_CWE_types.append("CWE-119")
        Mtrain_CWE_types.append("CWE-125")
        Mtrain_CWE_types.append("CWE-787")
        Mtrain_CWE_types.append("CWE-20")

        if len(Mtrain_CWE_types) != meta_batch:
            self.verbose(
                "Error! The number of training tasks is different with the meta-training's batch size."
            )
            exit(0)

        Mtest_CWE_types = [
            "CWE-369",
            "CWE-189",
            "CWE-399",
            "CWE-772",
            "CWE-476",
            "CWE-190",
        ]
        return Mtrain_CWE_types, Mtest_CWE_types

    # TODO
    def Scheme3(self, meta_batch):
        """
        实验方案3:  训练各种不同机理的CWE  测试其中某种相似机理或全新漏洞机理的CWE (适用 Data_six)
        ------------------------------------------------------------------------------------
        训练任务:   CWE-119、20、189、125、399

        测试任务a:  CWE-787、476、200、190、369、834、617、134、264、772、835、400
        """
        Mtrain_CWE_types = []
        Mtrain_CWE_types.append("CWE-119")
        Mtrain_CWE_types.append("CWE-20")
        Mtrain_CWE_types.append("CWE-189")
        Mtrain_CWE_types.append("CWE-125")

        if len(Mtrain_CWE_types) != meta_batch:
            self.verbose(
                "Error! The number of training tasks is different with the meta-training's batch size."
            )
            exit(0)

        Mtest_CWE_types = [
            "CWE-787",
            "CWE-476",
            "CWE-200",
            "CWE-190",
            "CWE-369",
            "CWE-834",
            "CWE-617",
            "CWE-134",
            "CWE-264",
            "CWE-772",
            "CWE-835",
            "CWE-400",
        ]
        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_Random_SARD_4(self, meta_batch, type_dataset):
        """针对 SARD_4 数据集，从数量大于一定数量的 CWE 中随机选择 meta_batch 个 CWE 作为训练集，剩下的作为测试集"""

        if type_dataset != "SARD_4":
            self.verbose(
                "Error! The Experiment Scheme is only suitable for dataset--SARD_4."
            )
            exit(0)

        Mtrain_CWE_types = []
        Mtest_CWE_types = []

        # 获取各cwe类型样本数量和
        dict_tmp_cwe_count = {}
        for syntax_feature in self.candi_data.keys():
            for k in list(self.candi_data[syntax_feature].keys())[:-1]:  # 各 cwe 类型
                if k not in dict_tmp_cwe_count.keys():
                    dict_tmp_cwe_count[k] = len(
                        self.candi_data[syntax_feature][k]["Embeddings"]
                    )
                else:
                    dict_tmp_cwe_count[k] = dict_tmp_cwe_count[k] + len(
                        self.candi_data[syntax_feature][k]["Embeddings"]
                    )

        # 选取样本数大于等于threshold的cwe类型
        threshold = 200
        candi_CWEs = []
        for k in dict_tmp_cwe_count.keys():
            if dict_tmp_cwe_count[k] >= threshold:
                candi_CWEs.append(k)

        # 检查meta_batch是否大于candi_CWEs的长度
        if meta_batch > len(candi_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)

        # 随机选择meta_batch个cwe类型作为训练集
        Mtrain_CWE_types = random.sample(candi_CWEs, meta_batch)

        # 从 candi_CWEs 中去掉已选中用于训练的类型
        for CWE in Mtrain_CWE_types:
            candi_CWEs.remove(CWE)

        # 剩下的作为测试集
        Mtest_CWE_types = candi_CWEs
        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_API_SARD_4(self, meta_batch, type_dataset):
        """
        针对SARD_4数据集，训练、测试拥有 API function call 特征的CWE类型
        """
        if type_dataset != "SARD_4":
            self.verbose(
                "Error! The Experiment Scheme is only suitable for dataset--SARD_4."
            )
            exit(0)
        Mtrain_CWE_types = []
        Mtest_CWE_types = []
        # API function call 中的 CWE 类型
        list_API_CWEs = list(self.candi_data["API function call"].keys())[:-1]
        # 获取各cwe类型样本数量和
        dict_tmp_cwe_count = {}
        for k in list_API_CWEs:
            dict_tmp_cwe_count[k] = len(
                self.candi_data["API function call"][k]["Embeddings"]
            )
        # 选取样本数大于等于threshold的cwe类型
        threshold = 200
        candi_CWEs = []
        for k in dict_tmp_cwe_count.keys():
            if dict_tmp_cwe_count[k] >= threshold:
                candi_CWEs.append(k)
        # 检查meta_batch是否大于candi_CWEs的长度
        if meta_batch > len(candi_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)
        # 随机选择meta_batch个cwe类型作为训练集
        Mtrain_CWE_types = random.sample(candi_CWEs, meta_batch)
        # 从 candi_CWEs 中去掉已选中用于训练的类型
        for CWE in Mtrain_CWE_types:
            candi_CWEs.remove(CWE)
        # 剩下的作为测试集
        Mtest_CWE_types = candi_CWEs

        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_Array_SARD_4(self, meta_batch, type_dataset):
        """
        针对SARD_4数据集，训练、测试拥有 Array usage 特征的CWE类型
        """
        if type_dataset != "SARD_4":
            self.verbose(
                "Error! The Experiment Scheme is only suitable for dataset--SARD_4."
            )
            exit(0)
        Mtrain_CWE_types = []
        Mtest_CWE_types = []
        # Array usage 中的 CWE 类型
        list_Array_CWEs = list(self.candi_data["Array usage"].keys())[:-1]
        # 获取各cwe类型样本数量和
        dict_tmp_cwe_count = {}
        for k in list_Array_CWEs:
            dict_tmp_cwe_count[k] = len(self.candi_data["Array usage"][k]["Embeddings"])
        # 选取样本数大于等于threshold的cwe类型
        threshold = 200
        candi_CWEs = []
        for k in dict_tmp_cwe_count.keys():
            if dict_tmp_cwe_count[k] >= threshold:
                candi_CWEs.append(k)
        # 检查meta_batch是否大于candi_CWEs的长度
        if meta_batch > len(candi_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)
        # 随机选择meta_batch个cwe类型作为训练集
        Mtrain_CWE_types = random.sample(candi_CWEs, meta_batch)
        # 从 candi_CWEs 中去掉已选中用于训练的类型
        for CWE in Mtrain_CWE_types:
            candi_CWEs.remove(CWE)
        # 剩下的作为测试集
        Mtest_CWE_types = candi_CWEs

        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_Arithmetic_SARD_4(self, meta_batch, type_dataset):
        """
        针对SARD_4数据集，训练、测试拥有 Arithmetic expression 特征的CWE类型
        """
        if type_dataset != "SARD_4":
            self.verbose(
                "Error! The Experiment Scheme is only suitable for dataset--SARD_4."
            )
            exit(0)
        Mtrain_CWE_types = []
        Mtest_CWE_types = []
        # Arithmetic expression 中的 CWE 类型
        list_Arithmetic_CWEs = list(self.candi_data["Arithmetic expression"].keys())[
            :-1
        ]
        # 获取各cwe类型样本数量和
        dict_tmp_cwe_count = {}
        for k in list_Arithmetic_CWEs:
            dict_tmp_cwe_count[k] = len(
                self.candi_data["Arithmetic expression"][k]["Embeddings"]
            )
        # 选取样本数大于等于threshold的cwe类型
        threshold = 200
        candi_CWEs = []
        for k in dict_tmp_cwe_count.keys():
            if dict_tmp_cwe_count[k] >= threshold:
                candi_CWEs.append(k)
        # 检查meta_batch是否大于candi_CWEs的长度
        if meta_batch > len(candi_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)
        # 随机选择meta_batch个cwe类型作为训练集
        Mtrain_CWE_types = random.sample(candi_CWEs, meta_batch)
        # 从 candi_CWEs 中去掉已选中用于训练的类型
        for CWE in Mtrain_CWE_types:
            candi_CWEs.remove(CWE)
        # 剩下的作为测试集
        Mtest_CWE_types = candi_CWEs

        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_Pointer_SARD_4(self, meta_batch, type_dataset):
        """
        针对SARD_4数据集，训练、测试拥有 Pointer usage 特征的CWE类型
        """
        if type_dataset != "SARD_4":
            self.verbose(
                "Error! The Experiment Scheme is only suitable for dataset--SARD_4."
            )
            exit(0)
        Mtrain_CWE_types = []
        Mtest_CWE_types = []
        # Pointer usage 中的 CWE 类型
        list_Pointer_CWEs = list(self.candi_data["Pointer usage"].keys())[:-1]
        # 获取各cwe类型样本数量和
        dict_tmp_cwe_count = {}
        for k in list_Pointer_CWEs:
            dict_tmp_cwe_count[k] = len(
                self.candi_data["Pointer usage"][k]["Embeddings"]
            )
        # 选取样本数大于等于threshold的cwe类型
        threshold = 200
        candi_CWEs = []
        for k in dict_tmp_cwe_count.keys():
            if dict_tmp_cwe_count[k] >= threshold:
                candi_CWEs.append(k)
        # 检查meta_batch是否大于candi_CWEs的长度
        if meta_batch > len(candi_CWEs):
            self.verbose("-------------------------------------------------------")
            self.verbose(
                "Error! Meta batch size is larger than the number of available candidate CWE types!"
            )
            self.verbose(
                "Make sure it is smaller than the total number of available CWE types for meta_training stage."
            )
            sys.exit(1)
        # 随机选择meta_batch个cwe类型作为训练集
        Mtrain_CWE_types = random.sample(candi_CWEs, meta_batch)
        # 从 candi_CWEs 中去掉已选中用于训练的类型
        for CWE in Mtrain_CWE_types:
            candi_CWEs.remove(CWE)
        # 剩下的作为测试集
        Mtest_CWE_types = candi_CWEs

        return Mtrain_CWE_types, Mtest_CWE_types

    def Scheme_Data_six_test(self, meta_batch, type_dataset):
        # 拿真实数据纯测试
        Mtrain_CWE_types = []
        Mtest_CWE_types = []
        if type_dataset == "Data_six":
            threshold_1 = 10
            threshold_2 = 2
        else:
            print("This experiment scheme is only for Data_six dataset.")
            exit(0)
        # 测试 CWE 类别 从数据集的所有漏洞类型中去掉用于训练的类型和样本数量少于threshold_2的类别
        Mtest_CWE_types = list(self.candi_data.keys())[:-1]
        for CWE in Mtest_CWE_types:
            num_single_cwe_samples = len(self.candi_data[CWE]["Embeddings"])
            if num_single_cwe_samples < threshold_2:
                Mtest_CWE_types.remove(CWE)
        if type_dataset == "Data_six":
            try:
                Mtest_CWE_types.remove('NVD-CWE-Other')
            except ValueError:
                pass
            try:
                Mtest_CWE_types.remove('NVD-CWE-noinfo')
            except ValueError:
                pass
        return Mtrain_CWE_types, Mtest_CWE_types

    def train_vali_test(self, Mtrain_CWE_types, Mtest_CWE_types):
        """对选中的meta-training CWE types 划分训练、验证集, 对候选的测试集，存入测试数据"""
        train_x = []
        train_y = []
        vali_x = []
        vali_y = []
        test_x = []
        test_y = []

        # 由于 SARD_4 数据集中，每个 CWE 类的样本存放在不同的 syntax feature 目录下，因此需要对系统输入的 candi_data 进行更新
        update_candi_data = {}
        if self.type_dataset == "SARD_4":
            for syntax_feature in self.candi_data.keys():
                for CWE in self.candi_data[syntax_feature].keys():
                    if CWE not in update_candi_data.keys():
                        update_candi_data[CWE] = {}
                        update_candi_data[CWE] = self.candi_data[syntax_feature][CWE]
                    else:
                        update_candi_data[CWE]["Embeddings"] = np.concatenate(
                            (
                                update_candi_data[CWE]["Embeddings"],
                                self.candi_data[syntax_feature][CWE]["Embeddings"],
                            ),
                            axis=0,
                        )
                        # update_candi_data[CWE]["Embeddings"].extend(
                        #     self.candi_data[syntax_feature][CWE]["Embeddings"]
                        # )
                        update_candi_data[CWE]["Labels"] = np.concatenate(
                            (
                                update_candi_data[CWE]["Labels"],
                                self.candi_data[syntax_feature][CWE]["Labels"],
                            ),
                            axis=0,
                        )
                        # update_candi_data[CWE]["Labels"].extend(
                        #     self.candi_data[syntax_feature][CWE]["Labels"]
                        # )
            self.candi_data = update_candi_data
            # print("update_candi_data:")
            # print(self.candi_data.keys())

        # 有漏洞样本:
        # 如果实验方案选定了用于 Meta-training 的CWE
        # TODO：训练、验证集比例可更改
        # 按0.8:0.2 划分训练验证集
        if Mtrain_CWE_types != []:
            for CWE in Mtrain_CWE_types:
                CWE_train_x = []
                CWE_vali_x = []
                CWE_train_y = []
                CWE_vali_y = []
                (
                    CWE_train_x,
                    CWE_vali_x,
                    CWE_train_y,
                    CWE_vali_y,
                ) = train_test_split(
                    self.candi_data[CWE]["Embeddings"],
                    self.candi_data[CWE]["Labels"],
                    test_size=0.2,
                    random_state=445,
                )
                train_x.append(CWE_train_x)
                train_y.append(CWE_train_y)
                vali_x.append(CWE_vali_x)
                vali_y.append(CWE_vali_y)
        # 如果实验方案选定了用于 Meta-testing 的CWE
        if Mtest_CWE_types != []:
            for CWE in Mtest_CWE_types:
                test_x.append(self.candi_data[CWE]["Embeddings"])
                test_y.append(self.candi_data[CWE]["Labels"])

        # 无漏洞样本:
        # 数据集同时用于元学习训练、测试
        if Mtrain_CWE_types != [] and Mtest_CWE_types != []:
            # 无漏洞样本加入训练、验证、测试集 0.6:0.2:0.2
            tmp_Benign_train_x = []
            tmp_Benign_train_y = []
            Benign_train_x = []
            Benign_train_y = []
            Benign_vali_x = []
            Benign_vali_y = []
            Benign_test_x = []
            Benign_test_y = []

            (
                tmp_Benign_train_x,
                Benign_test_x,
                tmp_Benign_train_y,
                Benign_test_y,
            ) = train_test_split(
                self.candi_data["Benign"]["Embeddings"],
                self.candi_data["Benign"]["Labels"],
                test_size=0.2,
                random_state=445,
            )
            (
                Benign_train_x,
                Benign_vali_x,
                Benign_train_y,
                Benign_vali_y,
            ) = train_test_split(
                tmp_Benign_train_x, tmp_Benign_train_y, test_size=0.25, random_state=445
            )

            train_x.append(Benign_train_x)
            train_y.append(Benign_train_y)
            vali_x.append(Benign_vali_x)
            vali_y.append(Benign_vali_y)
            test_x.append(Benign_test_x)
            test_y.append(Benign_test_y)

        # 数据集仅用于测试
        elif Mtrain_CWE_types == [] and Mtest_CWE_types != []:
            test_x.append(self.candi_data["Benign"]["Embeddings"])
            test_y.append(self.candi_data["Benign"]["Labels"])

        # 数据集仅用于训练
        elif Mtrain_CWE_types != [] and Mtest_CWE_types == []:
            (
                Benign_train_x,
                Benign_vali_x,
                Benign_train_y,
                Benign_vali_y,
            ) = train_test_split(
                self.candi_data["Benign"]["Embeddings"],
                self.candi_data["Benign"]["Labels"],
                test_size=0.2,
                random_state=445,
            )
            train_x.append(Benign_train_x)
            train_y.append(Benign_train_y)
            vali_x.append(Benign_vali_x)
            vali_y.append(Benign_vali_y)

        return train_x, train_y, vali_x, vali_y, test_x, test_y

    def choose_scheme_sep_data(self):
        """调用实验方案选择与数据划分"""
        self.verbose("-------------------------------------------------------")
        self.verbose(
            "According to the experiment scheme, spliting the dataset into training, validation, and test lists."
        )
        self.verbose("")
        # 在helper类和此处增加其他实验方案
        if self.experiment_scheme == "Random_test":
            # 随机选择的用于 meta-training 的 CWE 数据类型，以及所有用于 meta-testing 的 CWE 类型。
            Mtrain_CWE_types, Mtest_CWE_types = self.Random_Scheme(
                self.batch_s, self.type_dataset
            )
        # TODO
        elif self.experiment_scheme == "Scheme2":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme2(self.batch_s)
        # TODO
        elif self.experiment_scheme == "Scheme3":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme3(self.batch_s)
        elif self.experiment_scheme == "Scheme_Data_six_test":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_Data_six_test(
                self.batch_s, self.type_dataset
            )
        elif self.experiment_scheme == "Scheme_Random_SARD_4":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_Random_SARD_4(
                self.batch_s, self.type_dataset
            )
        elif self.experiment_scheme == "Scheme_API_SARD_4":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_API_SARD_4(
                self.batch_s, self.type_dataset
            )
        elif self.experiment_scheme == "Scheme_Array_SARD_4":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_Array_SARD_4(
                self.batch_s, self.type_dataset
            )
        elif self.experiment_scheme == "Scheme_Arithmetic_SARD_4":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_Arithmetic_SARD_4(
                self.batch_s, self.type_dataset
            )
        elif self.experiment_scheme == "Scheme_Pointer_SARD_4":
            Mtrain_CWE_types, Mtest_CWE_types = self.Scheme_Pointer_SARD_4(
                self.batch_s, self.type_dataset
            )

        self.verbose(
            "There are "
            + str(len(Mtrain_CWE_types))
            + " types of CWEs for meta_training stage."
        )
        self.verbose(Mtrain_CWE_types)
        self.verbose("")
        self.verbose(
            "There are "
            + str(len(Mtest_CWE_types))
            + " types of CWEs for meta_testing stage."
        )
        self.verbose(Mtest_CWE_types)
        self.verbose("")

        self.verbose("-------------------------------------------------------")
        self.verbose("Split the dataset into train, vali, and test sets.")
        train_x, train_y, vali_x, vali_y, test_x, test_y = self.train_vali_test(
            Mtrain_CWE_types, Mtest_CWE_types
        )
        return (
            train_x,
            train_y,
            vali_x,
            vali_y,
            test_x,
            test_y,
            Mtrain_CWE_types,
            Mtest_CWE_types,
        )


class Trainer(Helper):
    '''Handler for complete training progress'''

    def __init__(
        self, config, paras, train_x, train_y, vali_x, vali_y, Mtrain_CWE_types
    ):
        super(Trainer, self).__init__(config, paras)
        self.verbose("-------------------------------------------------------")
        self.verbose("******** Prepare for Meta-Training Process ********")
        self.model_save_path = config['training_settings']['model_save_path']
        self.model_save_name = config['training_settings']['model_saved_name']
        self.log_path = config['training_settings']['log_path']
        self.train_x = train_x
        self.train_y = train_y
        self.vali_x = vali_x
        self.vali_y = vali_y
        self.Mtrain_CWE_types = Mtrain_CWE_types
        # random.seed(445)

    """ 
    TODO: 暂时没用，后面可以用，放到Helper 类中
    将测试集预测结果和真实标签进行对比，求准确率 = (TP+TN)/(TP+FP+TN+FN)
    """

    def getAccuracy(self, probs, test_set_y):
        predicted_classes = []
        for item in probs:
            if item[0] > 0.5:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0)
        test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))
        return test_accuracy, predicted_classes

    """
    TODO: 暂时没用到，后面看怎么用，可以放到Helper类中
    画图，epochs-loss of Training and Validation
    """

    def plot_history(self, network_history):
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(network_history.history['loss'])
        plt.plot(network_history.history['val_loss'])
        plt.legend(['Training', 'Validation'])
        plt.savefig(
            self.config['training_settings']['model_save_path']
            + os.sep
            + self.config['training_settings']['model_saved_name']
            + '_Epoch_loss'
            + '.jpg'
        )

    """
    对一个特殊文件作处理
    TODO: 可删除
    """

    def preprocess_data(self, file_list_id, files_list):
        for i in range(len(file_list_id)):
            if file_list_id[i] == "puzzle_pce.c_puzzle_rotate_pce.c":
                print(files_list[i])
                while files_list[i][0] == '*':
                    files_list[i].remove('*')
                return files_list
        return files_list

    """Train 类真正要执行的部分"""

    def exec(self):
        # 载入word2vec; applyEmbedding; 载入深度学习模型
        (
            w2v_model,
            word_index,
            embedding_matrix,
            deep_model,
            model_func,
        ) = self.load_w2v_embed_model()

        # 保存训练好的模型的路径
        model_name = self.model_name
        date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # check model save folder
        if not os.path.exists(model_name + "/" + date):
            os.makedirs(model_name + "/" + date)
        self.verbose("-------------------------------------------------------")
        self.verbose("Models saved to folder: " + model_name + "/" + date)

        model_name = (
            model_name
            + "/"
            + date
            + "/mamlMetaFormal"
            + self.config['model_settings']['model']
        )

        # Loss Function
        loss_func = losses.BinaryCrossentropy(from_logits=False)
        self.verbose("-------------------------------------------------------")
        self.verbose("Loss function has been set.")

        # TODO:内外层优化器设定（后面又设定了一遍，选一个删掉）
        inner_optimizer = optimizers.SGD(learning_rate=0.00001)
        inner_optimizer = optimizers.Adam(learning_rate=0.001)  # 选用Adam
        outer_optimizer = optimizers.SGD(learning_rate=0.00001)
        outer_optimizer = optimizers.Adam(learning_rate=0.001)  # 选用Adam
        self.verbose("-------------------------------------------------------")
        self.verbose("Optimizers has been loaded.")

        # 训练轮数
        epo = 50  # 训练50轮，有50个.h5文件。
        self.verbose("-------------------------------------------------------")
        self.verbose("Training epoes has been set: " + str(epo))

        # Meta-training 阶段伪代码
        # ---------------------------------------
        # 当前训练数据：candi_data
        # candi_data["meta_training"]
        # # 实验方案 1:随机测试——即从数据集中样本多的漏洞类型中随机选四类CWE作为训练任务，样本少的其他漏洞类型作为测试任务
        #
        # if experiment_scheme == "Random_test":
        #     # meta_training 中随机选中的CWE类型，及 meta_testing中待测试的CWE类型
        #     self.Random_Scheme_Data_six
        #
        #     # epo = 50
        #     # steps = 50
        #     for e in range(epo):
        #         # meta_training 中的 train
        #         for i in range(steps):
        #             get_one_meta_batch(T1,T2,T3,T4)
        #             train_on_one_batch
        #         # 综合训练结果
        #         print("train result:")

        #         # meta_training 中的 valificatjion
        #         for i in range(steps/5*2):
        #             get_one_meta_batch(T1,T2,T3,T4)
        #             train_on_one_batch
        #         # 综合验证结果
        #         print("vali result:")
        train_x = self.train_x
        train_y = self.train_y
        vali_x = self.vali_x
        vali_y = self.vali_y
        Mtrain_CWE_types = self.Mtrain_CWE_types

        # 按元学习的任务和batch划分
        self.verbose("-------------------------------------------------------")
        self.verbose("Process the dataset into meta batches.")
        train_data = MAMLDataLoader(train_x, train_y)
        vali_data = MAMLDataLoader(vali_x, vali_y)

        self.verbose("-------------------------------------------------------")
        self.verbose("Start Meta-Training...")

        epo_time = []

        # 将每个epo训练、验证模型的表现保存到数组
        epos_train_loss = []
        epos_train_acc = []
        epos_train_rec = []
        epos_train_pre = []
        epos_train_TNR = []
        epos_train_FPR = []
        epos_train_FNR = []
        epos_train_F1 = []

        epos_vali_loss = []
        epos_vali_acc = []
        epos_vali_rec = []
        epos_vali_pre = []
        epos_vali_TNR = []
        epos_vali_FPR = []
        epos_vali_FNR = []
        epos_vali_F1 = []

        for e in range(epo):
            # 显示是否使用了GPU
            # self.verbose("-------------------------------------------------------")
            # self.verbose("Is GPU usded? " + str(test.is_gpu_available()))

            # 根据训练轮数调整学习率，加快收敛速度
            origin_lr = 0.0001
            if e <= 7:
                lr = 10 * origin_lr
            elif e <= 20:
                lr = origin_lr
            else:
                lr = 0.4 * origin_lr

            # TODO:内、外层优化器设定（前面设定过了，选一个位置删掉）
            inner_optimizer = optimizers.Adam(learning_rate=lr)
            outer_optimizer = optimizers.Adam(learning_rate=lr)

            # 显示训练进度
            # stateful_metrics中一定要指定显示的参数名，否则后面他会做一个时间上的平均，显示的值就不对。
            train_progbar = utils.Progbar(
                train_data.steps,
                stateful_metrics=[
                    "loss",
                    "accuracy",
                    "recall",
                    "precision",
                    "TNR",
                    "FPR",
                    "FNR",
                    "F1",
                ],
            )
            val_progbar = utils.Progbar(
                vali_data.steps,
                stateful_metrics=[
                    "loss",
                    "accuracy",
                    "recall",
                    "precision",
                    "TNR",
                    "FPR",
                    "FNR",
                    "F1",
                ],
            )
            print('\nEpoch {}/{}'.format(e + 1, 100))

            train_meta_loss = []
            train_meta_acc = []
            train_meta_pre = []
            train_meta_rec = []
            train_meta_TNR = []
            train_meta_FPR = []
            train_meta_FNR = []
            train_meta_F1 = []
            train_meta_interval = []

            val_meta_loss = []
            val_meta_acc = []
            val_meta_pre = []
            val_meta_rec = []
            val_meta_TNR = []
            val_meta_FPR = []
            val_meta_FNR = []
            val_meta_F1 = []
            val_meta_interval = []

            epo_start_time = datetime.datetime.now()

            # 训练集，每个batch完了更新外层参数
            self.verbose("-------------------------------------------------------")
            self.verbose("Meta-Training: Train")
            for i in range(train_data.steps):
                train_start_time = datetime.datetime.now()
                (
                    batch_train_loss,
                    acc,
                    rec,
                    pre,
                    TNR,
                    FPR,
                    FNR,
                    F1,
                ) = deep_model.train_on_metabatch(
                    train_data.get_one_metabatch(self.batch_s),
                    inner_optimizer,
                    inner_step=1,
                    outer_optimizer=outer_optimizer,
                    losss=loss_func,
                )
                train_end_time = datetime.datetime.now()
                train_interval = (train_end_time - train_start_time).seconds
                train_meta_interval.append(train_interval)

                train_meta_loss.append(batch_train_loss)
                # print("For batches, train_meta_loss:", train_meta_loss)

                train_meta_acc.append(acc)
                # print("For batches, train_meta_acc:", train_meta_acc)
                train_meta_rec.append(rec)
                train_meta_pre.append(pre)
                train_meta_TNR.append(TNR)
                train_meta_FPR.append(FPR)
                train_meta_FNR.append(FNR)
                train_meta_F1.append(F1)

                # # recall (TPR) = TP/(TP+FN) 检测出来的漏洞样本的个数（TP）占样本中所有漏洞样本个数的比例
                # num = 0.0  # TP+FN的数量，即一个batch所有样本中所有标签为1的样本数量
                # t = 0.0  # TP的数量，即模型预测正确且是1的数量
                # for ii in range(len(rec)):
                #     num += len(rec[ii])
                #     for j in range(len(rec[ii])):
                #         t += rec[ii][j]
                # recall = 0
                # if num != 0:
                #     recall = t / num
                # train_meta_rec.append(recall)

                # # precision = TP/(TP+FP) 被认定为漏洞程序的样本中，有多少是真正的漏洞程序
                # num = 0.0  # 模型做出所有预测为1的数量
                # t = 0.0  # 其中真的是1的数量
                # for ii in range(len(pre)):
                #     num += len(pre[ii])
                #     for j in range(len(pre[ii])):
                #         t += pre[ii][j]
                # precision = 0
                # if num != 0:
                #     precision = t / num
                # # print(precision)
                # train_meta_pre.append(precision)
                # # print(pre)

                # 输出显示，显示的是 50 steps (1个epo) 中的均值
                train_progbar.update(
                    i + 1,
                    [
                        ('loss', np.mean(train_meta_loss)),
                        ('accuracy', np.mean(train_meta_acc)),
                        ('recall', np.mean(train_meta_rec)),
                        ('precision', np.mean(train_meta_pre)),
                        ('TNR', np.mean(train_meta_TNR)),
                        ('FPR', np.mean(train_meta_FPR)),
                        ('FNR', np.mean(train_meta_FNR)),
                        ('F1', np.mean(train_meta_F1)),
                        ('Average time per step', np.mean(train_meta_interval)),
                    ],
                )
                print()

            # 验证集，不更新外层参数
            self.verbose("Meta-Training: Validation")
            for i in range(int(vali_data.steps / 5 * 2)):
                vali_start_time = datetime.datetime.now()
                (
                    batch_val_loss,
                    val_acc,
                    rec,
                    pre,
                    TNR,
                    FPR,
                    FNR,
                    F1,
                ) = deep_model.train_on_metabatch(
                    vali_data.get_one_metabatch(self.batch_s),
                    inner_optimizer,
                    inner_step=1,
                    losss=loss_func,
                )
                vali_end_time = datetime.datetime.now()
                vali_interval = (vali_end_time - vali_start_time).seconds

                val_meta_interval.append(vali_interval)
                val_meta_loss.append(batch_val_loss)
                val_meta_acc.append(val_acc)
                val_meta_rec.append(rec)
                val_meta_pre.append(pre)
                val_meta_TNR.append(TNR)
                val_meta_FPR.append(FPR)
                val_meta_FNR.append(FNR)
                val_meta_F1.append(F1)

                # # recall
                # num = 0.0
                # t = 0.0
                # for ii in range(len(rec)):
                #     num += len(rec[ii])
                #     for j in range(len(rec[ii])):
                #         t += rec[ii][j]
                # recall = 0
                # if num != 0:
                #     recall = t / num
                # val_meta_rec.append(recall)

                # # precision
                # num = 0.0
                # t = 0.0
                # for ii in range(len(pre)):
                #     num += len(pre[ii])
                #     for j in range(len(pre[ii])):
                #         t += pre[ii][j]
                # precision = 0
                # if num != 0:
                #     precision = t / num
                # val_meta_pre.append(precision)

                val_progbar.update(
                    i + 1,
                    [
                        ('val_loss', np.mean(val_meta_loss)),
                        ('val_accuracy', np.mean(val_meta_acc)),
                        ('val_rec', np.mean(val_meta_rec)),
                        ('val_precision', np.mean(val_meta_pre)),
                        ('val_TNR', np.mean(val_meta_TNR)),
                        ('val_FPR', np.mean(val_meta_FPR)),
                        ('val_FNR', np.mean(val_meta_FNR)),
                        ('val_F1', np.mean(val_meta_F1)),
                        ('Average time per step', np.mean(val_meta_interval)),
                    ],
                )
                print()

            # 保存模型
            deep_model.meta_model.save_weights(model_name + str(e) + ".h5")

            print()
            epo_end_time = datetime.datetime.now()
            epo_interval = (epo_end_time - epo_start_time).seconds
            print("Time for this epo:")
            print(epo_interval)
            epo_time.append(epo_interval)
            print()

            # 将每个epo训练模型的表现保存到数组
            epos_train_loss.append(np.mean(train_meta_loss))
            epos_train_acc.append(np.mean(train_meta_acc))
            epos_train_rec.append(np.mean(train_meta_rec))
            epos_train_pre.append(np.mean(train_meta_pre))
            epos_train_TNR.append(np.mean(train_meta_TNR))
            epos_train_FPR.append(np.mean(train_meta_FPR))
            epos_train_FNR.append(np.mean(train_meta_FNR))
            epos_train_F1.append(np.mean(train_meta_F1))

            epos_vali_loss.append(np.mean(val_meta_loss))
            epos_vali_acc.append(np.mean(val_meta_acc))
            epos_vali_rec.append(np.mean(val_meta_rec))
            epos_vali_pre.append(np.mean(val_meta_pre))
            epos_vali_TNR.append(np.mean(val_meta_TNR))
            epos_vali_FPR.append(np.mean(val_meta_FPR))
            epos_vali_FNR.append(np.mean(val_meta_FNR))
            epos_vali_F1.append(np.mean(val_meta_F1))

        self.verbose("-------------------------------------------------------")
        print('Average time per epo', np.mean(epo_time))
        print()

        # 根据F1-score的值，从 epos_train_F1 中找到前5个最大值的索引
        epos_train_loss = np.array(epos_train_loss)
        epos_train_acc = np.array(epos_train_acc)
        epos_train_rec = np.array(epos_train_rec)
        epos_train_pre = np.array(epos_train_pre)
        epos_train_TNR = np.array(epos_train_TNR)
        epos_train_FPR = np.array(epos_train_FPR)
        epos_train_FNR = np.array(epos_train_FNR)
        epos_train_F1 = np.array(epos_train_F1)
        index = np.argsort(epos_train_F1)[-5:]

        # 从 index 中找到对应的值
        print("The best five models according to the F1-score:")
        print(index)
        print()
        for i in range(len(index)):
            print("epo_loss:", epos_train_loss[index[i]])
            print("epo_acc:", epos_train_acc[index[i]])
            print("epo_rec:", epos_train_rec[index[i]])
            print("epo_pre:", epos_train_pre[index[i]])
            print("epo_TNR:", epos_train_TNR[index[i]])
            print("epo_FPR:", epos_train_FPR[index[i]])
            print("epo_FNR:", epos_train_FNR[index[i]])
            print("epo_F1:", epos_train_F1[index[i]])
            print()
        print()

        print("The best model:")
        print("epo_loss:", epos_train_loss[index[-1]])
        print("epo_acc:", epos_train_acc[index[-1]])
        print("epo_rec:", epos_train_rec[index[-1]])
        print("epo_pre:", epos_train_pre[index[-1]])
        print("epo_TNR:", epos_train_TNR[index[-1]])
        print("epo_FPR:", epos_train_FPR[index[-1]])
        print("epo_FNR:", epos_train_FNR[index[-1]])
        print("epo_F1:", epos_train_F1[index[-1]])
        print()

        # 多子图画出模型训练、验证过程中的各实验结果随epo的变化图
        # plt.figure()
        metrics = [
            ("Loss", "train_loss", "vali_loss"),
            ("Accuracy", "train_acc", "vali_acc"),
            ("Recall", "train_rec", "vali_rec"),
            ("Precision", "train_pre", "vali_pre"),
            ("TNR", "train_TNR", "vali_TNR"),
            ("FPR", "train_FPR", "vali_FPR"),
            ("FNR", "train_FNR", "vali_FNR"),
            ("F1-score", "train_F1", "vali_F1"),
        ]
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
        for i, (metric, train_metric, vali_metric) in enumerate(metrics):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            ax.plot(eval(f"epos_{train_metric}"), label=train_metric)
            ax.plot(eval(f"epos_{vali_metric}"), label=vali_metric)
            ax.set_xlabel("Epochs")
            ax.set_ylabel(metric)
            ax.legend(loc="best")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        graph_save_path = (
            "result_analysis"
            + os.sep
            + self.config['model_settings']['model']
            + os.sep
            + "Meta-training"
            + os.sep
            + date
            + os.sep
            + "metrics_epos"
        )
        if not os.path.exists(graph_save_path):
            os.makedirs(graph_save_path)
        plt.savefig(graph_save_path + os.sep + "metrics.png")
        plt.close()

        print(
            "Model training process graph saved to "
            + graph_save_path
            + os.sep
            + "metrics.png"
        )

        # fig, axes = plt.subplots(nrows=2, ncols=4)
        # axes[0, 0].plot(epos_train_loss, label="train_loss")
        # axes[0, 0].plot(epos_vali_loss, label="vali_loss")
        # axes[0, 0].set_xlabel("epos")
        # axes[0, 0].set_ylabel("loss")
        # axes[0, 0].legend()

        # axes[0, 1].plot(epos_train_acc, label="train_acc")
        # axes[0, 1].plot(epos_vali_acc, label="vali_acc")
        # axes[0, 1].set_xlabel("epos")
        # axes[0, 1].set_ylabel("acc")
        # axes[0, 1].legend()

        # axes[0, 2].plot(epos_train_rec, label="train_rec")
        # axes[0, 2].plot(epos_vali_rec, label="vali_rec")
        # axes[0, 2].set_xlabel("epos")
        # axes[0, 2].set_ylabel("rec")
        # axes[0, 2].legend()

        # axes[0, 3].plot(epos_train_pre, label="train_pre")
        # axes[0, 3].plot(epos_vali_pre, label="vali_pre")
        # axes[0, 3].set_xlabel("epos")
        # axes[0, 3].set_ylabel("pre")
        # axes[0, 3].legend()

        # axes[1, 0].plot(epos_train_TNR, label="train_TNR")
        # axes[1, 0].plot(epos_vali_TNR, label="vali_TNR")
        # axes[1, 0].set_xlabel("epos")
        # axes[1, 0].set_ylabel("TNR")
        # axes[1, 0].legend()

        # axes[1, 1].plot(epos_train_FPR, label="train_FPR")
        # axes[1, 1].plot(epos_vali_FPR, label="vali_FPR")
        # axes[1, 1].set_xlabel("epos")
        # axes[1, 1].set_ylabel("FPR")
        # axes[1, 1].legend()

        # axes[1, 2].plot(epos_train_FNR, label="train_FNR")
        # axes[1, 2].plot(epos_vali_FNR, label="vali_FNR")
        # axes[1, 2].set_xlabel("epos")
        # axes[1, 2].set_ylabel("FNR")
        # axes[1, 2].legend()

        # axes[1, 3].plot(epos_train_F1, label="train_F1")
        # axes[1, 3].plot(epos_vali_F1, label="vali_F1")
        # axes[1, 3].set_xlabel("epos")
        # axes[1, 3].set_ylabel("F1")

        # graph_save_path = (
        #     "result_analysis/Graphs"
        #     + os.sep
        #     + self.config['model_settings']['model']
        #     + os.sep
        #     + "Meta-training"
        #     + os.sep
        #     + date
        #     + os.sep
        #     + "metrics_epos"
        # )
        # if not os.path.exists(graph_save_path):
        #     os.makedirs(graph_save_path)
        # plt.savefig(graph_save_path + os.sep + "metrics.png")
        # plt.close()

        # # 画出模型训练、验证过程中的loss随epo的变化图
        # plt.figure()
        # plt.plot(epos_train_loss, label="train_loss")
        # plt.plot(epos_vali_loss, label="vali_loss")
        # plt.xlabel("epos")
        # plt.ylabel("loss")
        # plt.legend()
        # graph_save_path = (
        #     "result_analysis/Graphs"
        #     + os.sep
        #     + self.config['model_settings']['model']
        #     + os.sep
        #     + "Meta-training"
        #     + os.sep
        #     + date
        #     + os.sep
        #     + "loss_epos"
        # )
        # if not os.path.exists(graph_save_path):
        #     os.makedirs(graph_save_path)
        # plt.savefig(graph_save_path + os.sep + "loss.png")
        # plt.close()

        # # 画出模型训练、验证过程中的F1随epo的变化图
        # plt.figure()
        # plt.plot(epos_train_F1, label="train_F1")
        # plt.plot(epos_vali_F1, label="vali_F1")
        # plt.xlabel("epos")
        # plt.ylabel("F1")
        # plt.legend()
        # graph_save_path = (
        #     "result_analysis/Graphs"
        #     + os.sep
        #     + self.config['model_settings']['model']
        #     + os.sep
        #     + "Meta-training"
        #     + os.sep
        #     + date
        #     + os.sep
        #     + "F1_epos"
        # )
        # if not os.path.exists(graph_save_path):
        #     os.makedirs(graph_save_path)
        # plt.savefig(graph_save_path + os.sep + "F1.png")
        # plt.close()

        return

        # TODO!!!!!!!!
        # if self.config['model_settings']['model_para']['handle_data_imbalance']:
        #    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',classes = np.unique(np.array(train_y).flatten()), y = np.array(train_y).flatten())
        #    print(class_weights)
        #    #class_weights = dict(enumerate(class_weights))

        # else:
        #    class_weights = None


# ===========================================================================================
class Tester(Helper):
    '''Handler for complete inference progress'''

    def __init__(
        self, config, paras, test_x, test_y, Mtrain_CWE_types, Mtest_CWE_types
    ):
        super(Tester, self).__init__(config, paras)
        self.verbose("-------------------------------------------------------")
        self.verbose("********** Prepare for Meta-Testing Process **********")
        self.test_x = test_x
        self.test_y = test_y
        self.Mtrain_CWE_types = Mtrain_CWE_types
        self.Mtest_CWE_types = Mtest_CWE_types

        # 输出实验日期
        now = datetime.datetime.now()
        # self.verbose("Experiment Date: " + now.strftime("%m/%d/%Y %H:%M:%S"))

    # def modelLoader(self):
    #     trained_model_path = self.paras.trained_model
    #     if os.path.isfile(trained_model_path):
    #         # Load the model and print the model details.
    #         trained_model = load_model(trained_model_path)
    #         trained_model.summary()
    #         return trained_model
    #     else:
    #         self.verbose("Failed to load the trained model!")

    # def loadTestSet(self):
    #     if not self.config['training_settings']['using_separate_test_set']:
    #         total_list, total_list_id = self.loadData(self.paras.data_dir)
    #         self.verbose("Perform tokenization ....")
    #         total_sequences, word_index = self.tokenization(total_list)
    #         self.verbose("Pad the sequence to unified length...")
    #         total_list_pad = self.padding(total_sequences)
    #         self.verbose("Patition the data ....")
    #         tuple_with_test = self.patitionData(total_list_pad, total_list_id)
    #         test_set_x = tuple_with_test[6]
    #         test_set_y = np.asarray(tuple_with_test[7]).flatten()
    #         test_set_id = tuple_with_test[8]
    #         self.verbose(
    #             "There are "
    #             + str(len(test_set_x))
    #             + " total samples in the test set. "
    #             + str(np.count_nonzero(test_set_y))
    #             + " vulnerable samples. "
    #         )

    #     else:
    #         self.verbose(
    #             "Loading test data from "
    #             + os.getcwd()
    #             + os.sep
    #             + self.config['training_settings']['test_set_path']
    #         )
    #         test_list, test_list_id = self.loadData(
    #             self.config['training_settings']['test_set_path']
    #         )
    #         self.verbose("Perform tokenization ....")
    #         test_sequences, word_index = self.tokenization(test_list)
    #         self.verbose("Pad the sequence to unified length...")
    #         test_list_pad = self.padding(test_sequences)
    #         test_list_label = GenerateLabels(test_list_id)
    #         test_set_x = test_list_pad
    #         test_set_y = test_list_label

    #     return test_set_x, test_set_y, test_set_id

    # def getAccuracy(self, probs, test_set_y):
    #     predicted_classes = []
    #     for item in probs:
    #         if item[0] > 0.5:
    #             predicted_classes.append(1)
    #         else:
    #             predicted_classes.append(0)
    #     test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))
    #     return test_accuracy, predicted_classes

    def exec(self):
        # test_set_x, test_set_y, test_set_id = self.loadTestSet()
        # model = self.modelLoader()
        # probs = model.predict(
        #     test_set_x,
        #     batch_size=self.config['training_settings']['network_config']['batch_size'],
        #     verbose=self.paras.verbose,
        # )
        # accuracy, predicted_classes = self.getAccuracy(probs, test_set_y)
        # self.verbose(
        #     self.config['model_settings']['model'] + " classification result: \n"
        # )
        # self.verbose("Total accuracy: " + str(accuracy))
        # self.verbose("----------------------------------------------------")
        # self.verbose("The confusion matrix: \n")
        # target_names = [
        #     "Non-vulnerable",
        #     "Vulnerable",
        # ]  # non-vulnerable->0, vulnerable->1
        # print(confusion_matrix(test_set_y, predicted_classes, labels=[0, 1]))
        # print("\r\n")
        # print(
        #     classification_report(
        #         test_set_y, predicted_classes, target_names=target_names
        #     )
        # )
        # # Wrap the result to a CSV file.
        # if not isinstance(test_set_x, list):
        #     test_set_x = test_set_x.tolist()
        # if not isinstance(probs, list):
        #     probs = probs.tolist()
        # if not isinstance(test_set_id, list):
        #     test_set_id = test_set_id.tolist()
        # zippedlist = list(zip(test_set_id, probs, test_set_y))
        # result_set = pd.DataFrame(
        #     zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label']
        # )
        # # print(result_set,self.paras.output_dir)
        # if not os.path.exists(
        #     self.paras.output_dir + os.sep + self.config['model_settings']['model']
        # ):
        #     os.mkdir(
        #         self.paras.output_dir + os.sep + self.config['model_settings']['model']
        #     )

        # ListToCSV(
        #     result_set,
        #     self.paras.output_dir
        #     + os.sep
        #     + self.config['model_settings']['model']
        #     + os.sep
        #     + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #     + '_result.csv',
        # )

        # 实验结果保存路径（保存训练、测试漏洞类型，k值，各漏洞类型的实验结果，使用的哪一个模型等）
        date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # experiment_result_path = "experiment_result"
        # if not os.path.isdir(experiment_result_path):
        #     os.makedirs(experiment_result_path)
        # experiment_result_filename = date + '.json'
        # if os.path.isfile(experiment_result_path + os.sep + experiment_result_filename):
        #     experiment_result_filename = (
        #         os.path.splitext(experiment_result_filename)[0]
        #         + "_new"
        #         + os.path.splitext(experiment_result_filename)[1]
        #     )
        # experiment_result_path = (
        #     experiment_result_path + os.sep + experiment_result_filename
        # )

        experiment_result_path = (
            "result_analysis"
            + os.sep
            + self.config['model_settings']['model']
            + os.sep
            + "Meta-test"
            + os.sep
            + date
        )
        if not os.path.isdir(experiment_result_path):
            os.makedirs(experiment_result_path)
        experiment_result_filename = "test_info" + '.json'
        experiment_result_path = (
            experiment_result_path + os.sep + experiment_result_filename
        )
        self.verbose("Saving experiment result to " + experiment_result_path)

        js_exp_result = {}
        js_exp_result["Experiment Settings"] = {}
        js_exp_result["Experiment Results"] = {}

        # 实验的训练、测试漏洞类型：
        self.verbose("Meta-training CWE types:")
        self.verbose(self.Mtrain_CWE_types)
        self.verbose("Meta-testing CWE types:")
        self.verbose(self.Mtest_CWE_types)

        js_exp_result["Experiment Settings"][
            "Meta-training CWE types"
        ] = self.Mtrain_CWE_types
        js_exp_result["Experiment Settings"][
            "Meta-testing CWE types"
        ] = self.Mtest_CWE_types

        (
            w2v_model,
            word_index,
            embedding_matrix,
            deep_model,
            model_func,
        ) = self.load_w2v_embed_model()

        # 加载测试模型
        self.verbose("-------------------------------------------------------")
        self.verbose("Loading deep learning model. ")
        # loadModelNum = 0
        # # 路径 e.g., Models/BiGRU/mamlMetaFormalBiGRU
        # model_name = self.model_name
        # # main.meta.py 是否指定了 --model_num 参数
        # # 可以选择特定的模型进行测试
        # if self.paras.model_num != None:
        #     loadModelNum = self.paras.model_num
        # else:
        #     self.verbose(
        #         "Error! Please name the index of model that you want to test using the --model_num parameter!"
        #     )
        #     exit(2)
        # load_name = model_name + str(loadModelNum) + ".h5"

        load_name = self.paras.trained_model
        try:
            deep_model.meta_model.load_weights(load_name)  # 仅读取权重
        except OSError:
            self.verbose(
                "Error! The given model does not exist. Please select another one!"
            )
            exit(3)

        js_exp_result["Experiment Settings"]["Loaded Model"] = load_name

        # Loss Function
        loss_func = losses.BinaryCrossentropy(from_logits=False)
        self.verbose("-------------------------------------------------------")
        self.verbose("Loss function has been set.")

        # TODO:内外层优化器设定（后面又设定了一遍，选一个删掉）
        inner_optimizer = optimizers.SGD(learning_rate=0.00001)
        inner_optimizer = optimizers.Adam(learning_rate=0.001)  # 选用Adam
        outer_optimizer = optimizers.SGD(learning_rate=0.00001)
        outer_optimizer = optimizers.Adam(learning_rate=0.001)  # 选用Adam
        self.verbose("-------------------------------------------------------")
        self.verbose("Optimizers has been loaded.")

        test_x = self.test_x
        test_y = self.test_y
        Mtest_CWE_types = self.Mtest_CWE_types

        # k shots for support set.
        k = 1
        js_exp_result["Experiment Settings"]["K"] = k

        # 测试各个 CWE 的分类效果
        p = 0
        for CWE in Mtest_CWE_types:
            self.verbose("=======================================================")
            self.verbose("Now is testing " + CWE + "!")
            js_exp_result["Experiment Results"][CWE] = {}
            print("value of p", p)

            # Support set
            # how many samples in this cwe type
            # sample_size = len(candi_data["meta_testing"][CWE]["test_x"])
            # Non_vul_sample_size = len(
            #     candi_data["meta_testing"]["Non_Vulnerability"]["test_x"]
            # )

            # 检查数据集中CWE样本和非CWE样本的数量是否至少大于 k 个
            sample_size = len(test_x[p])
            if sample_size < k:
                print(
                    "Error! For meta testing, the number of samples for this CWE is smaller than the value of k."
                )
                exit(5)
            Non_vul_sample_size = len(test_x[-1])
            if sample_size < k:
                print(
                    "Error! For meta testing, the number of benign samples is smaller than the value of k."
                )
                exit(5)

            # 统计多次实验的结果
            list_all_labels = []
            list_all_probs = []
            list_all_ppred = []
            list_all_loss = []
            list_all_acc = []
            list_all_recall = []
            list_all_precision = []
            list_all_tnr = []
            list_all_fpr = []
            list_all_fnr = []
            list_all_f1 = []
            all_confusion_matrix = np.zeros((2, 2))
            list_all_test_support_interval = []
            list_all_mean_test_query = []

            # Repeat the experiment 10 times
            for test_round in range(10):
                self.verbose("-------------------------------------------------------")
                self.verbose("Experiment round:" + str(test_round))

                # 随机选k个CWE样本和k个非CWE样本作为support set
                index = []
                index = random.sample(range(sample_size), k)
                index_Non_vul = random.sample(range(Non_vul_sample_size), k)
                support_tuples = []
                for i in index:
                    support_tuples.append(
                        (
                            # candi_data["meta_testing"][CWE]["test_x"][i],
                            # candi_data["meta_testing"][CWE]["test_y"][i],
                            test_x[p][i],
                            test_y[p][i],
                        )
                    )
                for j in index_Non_vul:
                    support_tuples.append(
                        (
                            # candi_data["meta_testing"]["Non_Vulnerability"]["test_x"][j],
                            # candi_data["meta_testing"]["Non_Vulnerability"]["test_y"][j],
                            test_x[-1][j],
                            test_y[-1][j],
                        )
                    )
                random.shuffle(support_tuples)
                support_image = []
                support_label = []
                for support_tuple in support_tuples:
                    support_image.append(support_tuple[0])
                    support_label.append(support_tuple[1])

                # Query set
                # 把cwe样本中没抽中的样本放到 query里，从中选max_size个样本作为query set (无漏洞的同理)
                query_tuples = []
                for m in range(sample_size):
                    if m not in index:
                        query_tuples.append(
                            (
                                # candi_data["meta_testing"][CWE]["test_x"][m],
                                # candi_data["meta_testing"][CWE]["test_y"][m],
                                test_x[p][m],
                                test_y[p][m],
                            )
                        )
                size = len(query_tuples)
                # 应该对测试集数据大小进行约束，避免测试数据量过大，导致GPU内存不足，报错OOM。
                max_size = 100
                if size > max_size:
                    # print("There are %d samples in the test set." % size)
                    # print("The size of test set is too large!")
                    size = max_size
                    reduced_query_tuples = random.sample(query_tuples, size)
                    # print("The size of test set is reduced to %d." % max_size)
                else:
                    print("The size of test set of the CWE is %d." % size)
                    print(
                        "The size is small. We will make them all be tested but it might be not enough for statistical conclusions."
                    )
                    reduced_query_tuples = query_tuples
                query_tuples = reduced_query_tuples

                # 采样相同数量的无漏洞样本放到query里
                index_Non = random.sample(range(Non_vul_sample_size), size)
                for n in index_Non:
                    query_tuples.append(
                        (
                            # candi_data["meta_testing"]["Non_Vulnerability"]["test_x"][n],
                            # candi_data["meta_testing"]["Non_Vulnerability"]["test_y"][n],
                            test_x[-1][n],
                            test_y[-1][n],
                        )
                    )
                random.shuffle(query_tuples)
                query_image = []
                query_label = []
                for query_tuple in query_tuples:
                    query_image.append(query_tuple[0])
                    query_label.append(query_tuple[1])

                # 计算模型对于 CWE-k 的测试集的损失、准确率、召回率、精度等。
                (
                    query_label,
                    prb_pred,  # 概率, 用于计算AUC
                    ppred,
                    type_loss,
                    type_acc,
                    rec,
                    pre,
                    TNR,
                    FPR,
                    FNR,
                    F1_score,
                    confusionMatrix,
                    test_support_interval,
                    mean_test_query,
                ) = deep_model.test_on_one_type(
                    support_image,
                    support_label,
                    query_image,
                    query_label,
                    inner_optimizer,
                    inner_step=3,
                    losss=loss_func,
                )

                print(
                    "query_label: ",
                    query_label,
                    "prb_pred: ",
                    prb_pred,
                    "ppred: ",
                    ppred,
                    "Loss: ",
                    type_loss,
                    "Accuracy: ",
                    type_acc,
                    "Recall: ",
                    rec,
                    "Precision: ",
                    pre,
                    "TNR: ",
                    TNR,
                    "FPR: ",
                    FPR,
                    "FNR: ",
                    FNR,
                    "F1_score: ",
                    F1_score,
                )
                print("Confusion Matrix:")
                print(confusionMatrix)
                print(
                    "Total time for training meta-test support data: ",
                    test_support_interval,
                )
                print("Average time for testing each test data: ", mean_test_query)

                list_all_labels = list_all_labels + list(query_label)
                list_all_probs = list_all_probs + list(prb_pred)
                list_all_ppred = list_all_ppred + list(ppred)
                list_all_loss.append(type_loss)
                list_all_acc.append(type_acc)
                list_all_recall.append(rec)
                list_all_precision.append(pre)
                list_all_tnr.append(TNR)
                list_all_fpr.append(FPR)
                list_all_fnr.append(FNR)
                list_all_f1.append(F1_score)
                all_confusion_matrix += confusionMatrix
                list_all_test_support_interval.append(test_support_interval)
                list_all_mean_test_query.append(mean_test_query)

            # 计算10次实验的平均值
            average_loss = np.mean(list_all_loss)
            average_acc = np.mean(list_all_acc)
            average_recall = np.mean(list_all_recall)
            average_precision = np.mean(list_all_precision)
            average_tnr = np.mean(list_all_tnr)
            average_fpr = np.mean(list_all_fpr)
            average_fnr = np.mean(list_all_fnr)
            average_f1 = np.mean(list_all_f1)
            average_test_support_interval = np.mean(list_all_test_support_interval)
            average_mean_test_query = np.mean(list_all_mean_test_query)
            self.verbose("-------------------------------------------------------")
            self.verbose("Conclusion -- The test result after 10 rounds:")
            print("Average loss: ", average_loss)
            print("Average accuracy: ", average_acc)
            print("Average recall: ", average_recall)
            print("Average precision: ", average_precision)
            print("Average TNR: ", average_tnr)
            print("Average FPR: ", average_fpr)
            print("Average FNR: ", average_fnr)
            print("Average F1_score: ", average_f1)
            print("Overall confusion matrix:")
            print(all_confusion_matrix)
            print(
                "Average time for training meta-test support data of each round: ",
                average_test_support_interval,
            )
            print("Average time for testing each test data: ", average_mean_test_query)
            print()
            print()

            # 把实验结果写入json文件
            js_exp_result["Experiment Results"][CWE]["Query labels"] = list(
                map(float, list_all_labels)
            )
            js_exp_result["Experiment Results"][CWE][
                "Probs. of being vulnerable"
            ] = list(map(float, list_all_probs))
            js_exp_result["Experiment Results"][CWE]["Predicted labels"] = list(
                map(float, list_all_ppred)
            )
            js_exp_result["Experiment Results"][CWE]["Average loss"] = float(
                average_loss
            )
            js_exp_result["Experiment Results"][CWE]["Average accuracy"] = float(
                average_acc
            )
            js_exp_result["Experiment Results"][CWE]["Average recall"] = float(
                average_recall
            )
            js_exp_result["Experiment Results"][CWE]["Average precision"] = float(
                average_precision
            )
            js_exp_result["Experiment Results"][CWE]["Average TNR"] = float(average_tnr)
            js_exp_result["Experiment Results"][CWE]["Average FPR"] = float(average_fpr)
            js_exp_result["Experiment Results"][CWE]["Average FNR"] = float(average_fnr)
            js_exp_result["Experiment Results"][CWE]["Average F1_score"] = float(
                average_f1
            )
            js_exp_result["Experiment Results"][CWE][
                "Overall confusion matrix"
            ] = all_confusion_matrix.tolist()

            p = p + 1

        experiment_result = json.dumps(js_exp_result)
        with open(experiment_result_path, 'w') as f:
            f.write(experiment_result)
            print("Experiment result loaded in: " + experiment_result_path)

        # 画图：各测试CWE类型的各项评估结果。图1: 一次性展示所有CWE的所有评估结果；图2: 分别表示所有CWE的各项评估结果
        test_CWE_types = js_exp_result["Experiment Settings"]["Meta-testing CWE types"]
        list_query_labels = []
        list_predict_probs = []
        list_predicted_labels = []
        list_loss = []
        list_acc = []
        list_recall = []
        list_precision = []
        list_TNR = []
        list_FPR = []
        list_FNR = []
        list_F1 = []
        for CWE in test_CWE_types:
            list_query_labels.append(
                js_exp_result["Experiment Results"][CWE]["Query labels"]
            )
            list_predict_probs.append(
                js_exp_result["Experiment Results"][CWE]["Probs. of being vulnerable"]
            )
            list_predicted_labels.append(
                js_exp_result["Experiment Results"][CWE]["Predicted labels"]
            )
            list_loss.append(js_exp_result["Experiment Results"][CWE]["Average loss"])
            list_acc.append(
                js_exp_result["Experiment Results"][CWE]["Average accuracy"]
            )
            list_recall.append(
                js_exp_result["Experiment Results"][CWE]["Average recall"]
            )
            list_precision.append(
                js_exp_result["Experiment Results"][CWE]["Average precision"]
            )
            list_TNR.append(js_exp_result["Experiment Results"][CWE]["Average TNR"])
            list_FPR.append(js_exp_result["Experiment Results"][CWE]["Average FPR"])
            list_FNR.append(js_exp_result["Experiment Results"][CWE]["Average FNR"])
            list_F1.append(js_exp_result["Experiment Results"][CWE]["Average F1_score"])
        data = [
            list_loss,
            list_acc,
            list_recall,
            list_precision,
            list_TNR,
            list_FPR,
            list_FNR,
            list_F1,
        ]
        metrices_labels = [
            "Loss",
            "Accuracy",
            "Recall",
            "Precision",
            "TNR",
            "FPR",
            "FNR",
            "F1-score",
        ]

        # 把所有测试CWE类型的所有评估结果画在一张图上
        def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0):
            '''
            labels : x轴坐标标签序列
            datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
            tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
            group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
            bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
            '''
            plt.figure(figsize=(30, 8), dpi=300)

            # x为每组柱子x轴的基准位置
            x = np.arange(len(labels)) * tick_step
            # group_num为数据的组数，即每组柱子的柱子个数
            group_num = len(datas)
            # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
            group_width = tick_step - group_gap
            # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
            bar_span = group_width / group_num
            # bar_width为每个柱子的实际宽度
            bar_width = bar_span - bar_gap
            # 绘制柱子
            for index, y in enumerate(datas):
                plt.bar(
                    x + index * bar_span, y, bar_width, label=metrices_labels[index]
                )
            plt.ylabel('Scores')
            plt.title('Meta-test Result')
            # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
            ticks = x + (group_width - bar_span) / 2
            plt.xticks(ticks, labels)
            plt.legend(loc='upper right')

            model_file = os.path.split(load_name)[-1]
            modelname = model_file.split('.')[0]
            graph_save_path = (
                "result_analysis"
                + os.sep
                + self.config['model_settings']['model']
                + os.sep
                + "Meta-test"
                + os.sep
                + date
            )
            if not os.path.exists(graph_save_path):
                os.makedirs(graph_save_path)
            plt.savefig(graph_save_path + os.sep + modelname + "_all_in_one.png")
            print(
                "All-in-one test result graph saved in: "
                + graph_save_path
                + os.sep
                + modelname
                + "_all_in_one.png"
            )
            plt.close()

        # 分别画出各实验指标与所有测试cwe类型的关系图(共享x坐标轴)
        def create_distinct_bars(x_labels, datas):
            fig, axes = plt.subplots(nrows=8, ncols=1, sharex=True, figsize=(14, 10))

            # 添加x,y轴刻度
            x_ticks_label = x_labels
            y_ticks = np.arange(0, 1.01, 0.1)

            for i in range(len(datas)):
                ax = axes[i]
                ax.bar(x_labels, datas[i], label=metrices_labels[i])
                if i == range(len(datas))[-1]:
                    ax.set_xticks(x_labels)
                    ax.set_xticklabels(labels=x_labels, rotation=45)  # 旋转45度
                ax.set_yticks(y_ticks[::5])
                ax.set_ylabel(metrices_labels[i])
                ax.legend(loc="upper right")
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

            model_file = os.path.split(load_name)[-1]
            modelname = model_file.split('.')[0]
            graph_save_path = (
                "result_analysis"
                + os.sep
                + self.config['model_settings']['model']
                + os.sep
                + "Meta-test"
                + os.sep
                + date
            )
            if not os.path.exists(graph_save_path):
                os.makedirs(graph_save_path)
            plt.savefig(graph_save_path + os.sep + modelname + "_subplots.png")
            print(
                "Subplots test result graph saved in: "
                + graph_save_path
                + os.sep
                + modelname
                + "_subplots.png"
            )
            plt.close()

        create_multi_bars(test_CWE_types, data, tick_step=10, group_gap=1.5, bar_gap=0)
        create_distinct_bars(test_CWE_types, data)

        # 多子图画出所有测试CWE类型的ROC、AUC曲线
        def create_multi_roc_curve(
            test_CWE_types, list_query_labels, list_predict_probs
        ):
            # 画图
            plt.figure(figsize=(10, 10), dpi=100)
            for i in range(len(test_CWE_types)):
                # print("List of query labels for " + test_CWE_types[i] + ":")
                # print(list_query_labels[i])
                # print("List of prediction probabilities for " + test_CWE_types[i] + ":")
                # print(list_predict_probs[i])

                fpr, tpr, thresholds = roc_curve(
                    list_query_labels[i], list_predict_probs[i]
                )
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    lw=1,
                    label=test_CWE_types[i] + " (area = %0.2f)" % roc_auc,
                )
            plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r")
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")

            model_file = os.path.split(load_name)[-1]
            modelname = model_file.split('.')[0]
            graph_save_path = (
                "result_analysis"
                + os.sep
                + self.config['model_settings']['model']
                + os.sep
                + "Meta-test"
                + os.sep
                + date
            )
            if not os.path.exists(graph_save_path):
                os.makedirs(graph_save_path)
            plt.savefig(graph_save_path + os.sep + modelname + "_ROC_curve.png")
            print(
                "ROC curve graph saved in: "
                + graph_save_path
                + os.sep
                + modelname
                + "_ROC_curve.png"
            )
            plt.close()

        create_multi_roc_curve(test_CWE_types, list_query_labels, list_predict_probs)
        return