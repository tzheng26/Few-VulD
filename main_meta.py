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


def verbose(msg):
    '''Verbose function for print information to stdout'''
    print('[INFO]', msg)


# GPU support is recommended.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    default=None,
    help='Path to load the train, vali, test data file.',
)
# parser.add_argument(
#     '--output_dir',
#     default='result/',
#     type=str,
#     help='The output path of the trained network model.',
# )
parser.add_argument(
    '--trained_model', type=str, help='The path of the trained model for test.'
)
parser.add_argument('--model_num', default=None, help='num of loaded model')
paras = parser.parse_args()
config = yaml.safe_load(open(paras.config, 'r'))


time_main = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# 将输出重定向到日志文件
log_path = os.path.join(paras.logdir, "main_meta")
# log_path = "logs/main_meta"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_filename = time_main + '.txt'
if os.path.isfile(log_path + os.sep + log_filename):
    log_filename = (
        os.path.splitext(log_filename)[0] + "_new" + os.path.splitext(log_filename)[1]
    )
log_path = os.path.join(log_path, log_filename)
LogOutputFile = open(log_path, 'w')
sys.stdout = LogOutputFile
sys.stderr = LogOutputFile


# 显示当前时间
verbose("===================================================================")
verbose("Experiment time: {}".format(time_main))

space = ' '

# Step 1: Split training, validation, and test data
# 如果main.py指定了--sep_train_vali_test参数, 将数据集按实验方案划分训练、测试集
if paras.sep_train_vali_test == True:
    verbose("===================================================================")
    verbose("Operation: Split training, validation, and test data")
    verbose("===================================================================")
    verbose("Running script 'main_meta.py' with parameters:")
    verbose("python main_meta.py " + "--config " + paras.config + " \\")
    verbose(space * 20 + "--dataset " + paras.dataset + " \\")
    verbose(space * 20 + "--experiment_scheme " + paras.experiment_scheme + " \\")
    verbose(space * 20 + "--seed " + str(paras.seed) + " \\")
    verbose(space * 20 + "--sep_train_vali_test " + " \\")
    verbose(space * 20 + "--train_vali_test_dump " + paras.train_vali_test_dump)

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

    # Save the processed train, vali, and test data
    path = paras.train_vali_test_dump
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        i = 0
        filename = "train_vali_test" + str(i)
        save_path = os.path.join(path, filename)
        while os.path.isfile(save_path):
            i = i + 1
            filename = "train_vali_test" + str(i)
            save_path = os.path.join(path, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(Dict_tra_val_test, f)
        verbose("Train, vali, and test sets have been dumped into: " + save_path)

# Step 2 & 3: Meta-Training & Meta-Testing
if paras.train_or_test != None:
    # # 处理好的train, vali, test数据
    # file = paras.train_vali_test_load
    # with open(file, 'rb') as f:
    #     loaded_data = pickle.load(f)

    # Meta-training
    if paras.train_or_test == 0:
        verbose("===================================================================")
        verbose("Operation: Meta-training")
        verbose("===================================================================")
        verbose("Running script 'main_meta.py' with parameters:")
        verbose("python main_meta.py " + "--config " + paras.config + " \\")
        verbose(space * 20 + "--train_or_test " + str(paras.train_or_test) + " \\")
        verbose(
            space * 20 + "--train_vali_test_load " + paras.train_vali_test_load + " \\"
        )
        verbose(space * 20 + "--seed " + str(paras.seed))

        # train_x = loaded_data["train_x"]
        # train_y = loaded_data["train_y"]
        # vali_x = loaded_data["vali_x"]
        # vali_y = loaded_data["vali_y"]
        # Mtrain_CWE_types = loaded_data["Meta_train_CWE_types"]

        # trainer = Trainer(
        #     config, paras, train_x, train_y, vali_x, vali_y, Mtrain_CWE_types
        # )
        trainer = Trainer(config, paras)
        trainer.exec()

    # Meta-testing
    elif paras.train_or_test == 1:
        verbose("===================================================================")
        verbose("Operation: Meta-testing")
        verbose("===================================================================")
        verbose("Running script 'main_meta.py' with parameters:")
        verbose("python main_meta.py " + "--config " + paras.config + " \\")
        verbose(space * 20 + "--train_or_test " + str(paras.train_or_test) + " \\")
        verbose(
            space * 20 + "--train_vali_test_load " + paras.train_vali_test_load + " \\"
        )
        verbose(space * 20 + "trained_model " + paras.trained_model + " \\")
        verbose(space * 20 + "--seed " + str(paras.seed))

        # test_x = loaded_data["test_x"]
        # test_y = loaded_data["test_y"]
        # Mtrain_CWE_types = loaded_data["Meta_train_CWE_types"]
        # Mtest_CWE_types = loaded_data["Meta_test_CWE_types"]

        # tester = Tester(
        #     config, paras, test_x, test_y, Mtrain_CWE_types, Mtest_CWE_types
        # )
        tester = Tester(config, paras)
        tester.exec()


K.clear_session()
LogOutputFile.close()
