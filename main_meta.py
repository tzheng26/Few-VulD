# -*- coding: utf-8 -*-
"""
1) Splite training, validation, and test data
--------------------
python main_meta.py --config config/config.yaml --dataset Datasets/Data_six/processed_data --experiment_scheme Scheme_Random_SARD_4 --seed 445 --sep_train_vali_test --train_vali_test_dump train_vali_test

2) Start meta-training
--------------------
python main_meta.py --config config/config.yaml --train_or_test 0 --train_vali_test_load train_vali_test/train_vali_test0 --seed 445

3) Start meta-testing
--------------------
python main_meta.py --config config/config.yaml --train_or_test 1 --train_vali_test_load train_vali_test/train_vali_test0 --model_num 5 --seed 445
"""
import os
import sys
import yaml
import argparse
from tensorflow.compat.v1.keras import backend as K
from src.helper_for_metaL import Helper, Trainer, Tester
import time
import pickle

import faulthandler

faulthandler.enable()


# 将程序运行命令行返回结果保存到日志文件夹中，以日期为文件名
log_path = "logs/main_meta"
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
if os.path.isfile(log_path + os.sep + log_filename):
    log_filename = (
        os.path.splitext(log_filename)[0] + "_new" + os.path.splitext(log_filename)[1]
    )
log_path = log_path + os.sep + log_filename
LogOutputFile = open(log_path, 'w')
sys.stdout = LogOutputFile
sys.stderr = LogOutputFile


def verbose(msg):
    '''Verbose function for print information to stdout'''
    print('[INFO]', msg)


# GPU support is recommended.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Arguments
parser = argparse.ArgumentParser(
    description='Run a few-shot software vulnerability detection system.'
)
parser.add_argument('--config', type=str, help='Path to the configuration file.')
parser.add_argument(
    '--logdir', default='logs/', type=str, help='Path to log files.', required=False
)
parser.add_argument(
    '--seed',
    default=42,
    type=int,
    help='Random seed for reproducable results.',
    required=False,
)
parser.add_argument('--verbose', default=1, help='Show all messages.')
parser.add_argument(
    '--dataset',
    default=None,
    help='Directory of the user specified dataset (Processded in prep.py as a dictionary with keys "Embeddings" and "Labels").',
)
parser.add_argument(
    '--experiment_scheme', default='Random_test', help='Choose the experiment scheme.'
)
parser.add_argument(
    '--sep_train_vali_test',
    action='store_true',
    help='If identified, seperate dataset into train, vali, test sets.',
)
parser.add_argument(
    '--train_vali_test_dump',
    default='train_vali_test',
    help='The path to dump processed train, vali, and test data.',
)
parser.add_argument(
    '--train_or_test',
    choices=[0, 1],
    default=None,
    type=int,
    help='0 --the dataset is used for training; 1 --the dataset is used for testing.',
)
parser.add_argument(
    '--train_vali_test_load',
    default='train_vali_test/train_vali_test0',
    help='Path to load the train, vali, test data file.',
)
parser.add_argument(
    '--output_dir',
    default='result/',
    type=str,
    help='The output path of the trained network model.',
)
parser.add_argument(
    '--trained_model', type=str, help='The path of the trained model for test.'
)

parser.add_argument('--model_num', default=None, help='num of loaded model')

paras = parser.parse_args()
config = yaml.safe_load(open(paras.config, 'r'))

# ---------------------------------------------------------------------------------------------------------- #
# 显示当前时间
verbose("================================================")
verbose(
    "Experiment time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
)

# 输出用户当前运行该python脚本的命令行参数
verbose("================================================")
verbose("Now, the user is running the python script named: main_meta.py")
verbose("The command line parameters of the current python script are as follows:")
verbose(paras)


# Step 1: Split training, validation, and test data
# 如果main.py指定了--sep_train_vali_test参数, 将数据集按实验方案划分训练、测试集
if paras.sep_train_vali_test == True:
    verbose("================================================")
    verbose("Operation: Split training, validation, and test data")
    helper = Helper(config, paras)
    (
        train_x,
        train_y,
        vali_x,
        vali_y,
        test_x,
        test_y,
        Mtrain_CWE_types,
        Mtest_CWE_types,
    ) = helper.choose_scheme_sep_data()

    Dict_tra_val_test = {
        "Dataset_path": paras.dataset,
        "train_x": train_x,
        "train_y": train_y,
        "vali_x": vali_x,
        "vali_y": vali_y,
        "test_x": test_x,
        "test_y": test_y,
        "Meta_train_CWE_types": Mtrain_CWE_types,
        "Meta_test_CWE_types": Mtest_CWE_types,
    }

    path = paras.train_vali_test_dump
    if not os.path.exists(path):
        print("The path to dump processed train, vali, and test data does not exit.")
        os.makedirs(path)

    for i in range(1000):
        filename = "train_vali_test" + str(i)
        if filename in os.listdir(path):
            i = i + 1
        else:
            break

    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(Dict_tra_val_test, f)
    print("The processed train, vali, test data has been dumped into: " + save_path)

# Step 2 & 3: Meta-Training or Meta-Testing
if paras.train_or_test != None:
    # 处理好的train, vali, test数据
    file = paras.train_vali_test_load
    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)
    # train
    if paras.train_or_test == 0:
        verbose("================================================")
        verbose("Operation: Meta-training")
        train_x = loaded_data["train_x"]
        train_y = loaded_data["train_y"]
        vali_x = loaded_data["vali_x"]
        vali_y = loaded_data["vali_y"]
        Mtrain_CWE_types = loaded_data["Meta_train_CWE_types"]

        trainer = Trainer(
            config, paras, train_x, train_y, vali_x, vali_y, Mtrain_CWE_types
        )
        trainer.exec()
    # test
    elif paras.train_or_test == 1:
        verbose("================================================")
        verbose("Operation: Meta-testing")
        test_x = loaded_data["test_x"]
        test_y = loaded_data["test_y"]
        Mtrain_CWE_types = loaded_data["Meta_train_CWE_types"]
        Mtest_CWE_types = loaded_data["Meta_test_CWE_types"]

        tester = Tester(
            config, paras, test_x, test_y, Mtrain_CWE_types, Mtest_CWE_types
        )
        tester.exec()


# # 根据脚本是否指定 --test 参数，判断是进行模型训练还是模型测试。
# if paras.test != True:
#     trainer = Trainer(config, paras, train_x, train_y, vali_x, vali_y, Mtrain_CWE_types)
#     trainer.exec()

#     tester = Tester(config, paras, test_x, test_y, Mtest_CWE_types)
#     tester.exec()
# else:
#     tester = Tester(config, paras, test_x, test_y, Mtest_CWE_types)
#     tester.exec()

K.clear_session()
LogOutputFile.close()
