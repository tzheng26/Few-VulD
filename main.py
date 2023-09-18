# -*- coding: utf-8 -*-
"""
Interface to the Few-VulD system.

Step 1) Splite training, validation, and test data
------------------------------------------------------------
python main.py  --config config/config.yaml 
                --dataset Datasets/Data_six/processed_data 
                --experiment_scheme Scheme_Random_SARD_4 
                --sep_train_vali_test 
                --train_vali_test_dump train_vali_test
                --seed 445 

Step 2) Start meta-training
------------------------------------------------------------
python main.py  --config config/config.yaml 
                --train_or_test 0 
                --train_vali_test_load train_vali_test/train_vali_test0 
                --seed 445

Step 3) Start meta-testing
------------------------------------------------------------
python main.py  --config config/config.yaml 
                --train_or_test 1 
                --train_vali_test_load train_vali_test/train_vali_test0 
                --trained_model 5 
                --seed 445
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
import random


def verbose(msg):
    '''Verbose function for print information to stdout'''
    print('[INFO]', msg)


def set_args():
    """Set program arguments"""
    parser = argparse.ArgumentParser(
        description='Run a few-shot software vulnerability detection system.'
    )
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    parser.add_argument(
        '--logdir', default='logs/', type=str, help='Path to log files.', required=False
    )
    parser.add_argument(
        '--seed',
        default=None,
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
        '--experiment_scheme',
        default='Random_test',
        help='Choose the experiment scheme.',
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
    parser.add_argument(
        '--trained_model', type=str, help='The path of the trained model for test.'
    )
    # parser.add_argument('--model_num', default=None, help='num of loaded model')
    parser.add_argument(
        '--automode', default=0, type=int, help='0 --manual mode; 1 --auto mode.'
    )  # 自动化执行数据分割、训练、测试

    paras = parser.parse_args()
    config = yaml.safe_load(open(paras.config, 'r'))
    return paras, config


def redir_log(var_time):
    """Redirect stdout and stderr to log files"""
    log_path = os.path.join(paras.logdir, "main")  # log_path = "logs/main"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = var_time + '.txt'
    if os.path.isfile(log_path + os.sep + log_filename):
        log_filename = (
            os.path.splitext(log_filename)[0]
            + "_new"
            + os.path.splitext(log_filename)[1]
        )
    log_path = os.path.join(log_path, log_filename)
    # LogOutputFile = open(log_path, 'w')
    # sys.stdout = LogOutputFile
    # sys.stderr = LogOutputFile
    return log_path


def show_automode(paras):
    """显示程序执行模式是手动还是自动：手动模式则需要一步步指定参数，进行数据划分、训练、测试；自动模式则直接执行所有步骤"""
    if paras.automode == 0:
        verbose("===================================================================")
        verbose("Script Execution State: Manual mode")
    elif paras.automode == 1:
        verbose("===================================================================")
        verbose("Script Execution State: Auto mode")


def paras_split_trai_vali_test(paras):
    """
    Step 1: Split training, validation, and test data
    Output command line parameters
    """
    paras.sep_train_vali_test = True
    if paras.seed == None:
        paras.seed = random.randint(0, 2000)
    space = ' '  # 用于输出对齐
    verbose("===================================================================")
    verbose("Operation: Split training, validation, and test data")
    verbose("===================================================================")
    verbose("Running script 'main.py' with parameters:")
    verbose("python main.py " + "--config " + paras.config + " \\")
    verbose(space * 15 + "--dataset " + paras.dataset + " \\")
    verbose(space * 15 + "--experiment_scheme " + paras.experiment_scheme + " \\")
    verbose(space * 15 + "--sep_train_vali_test " + " \\")
    verbose(space * 15 + "--train_vali_test_dump " + paras.train_vali_test_dump + " \\")
    verbose(space * 15 + "--seed " + str(paras.seed))
    return paras


def get_and_store_trai_vali_test_data(config, paras):
    """Get meta-training and meta-testing data based on the experiment scheme"""
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
    # Save the processed train, vali, and test data
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
        os.makedirs(path)
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
    if paras.automode == 1:
        paras.train_vali_test_load = save_path
    paras.seed = None
    return paras


def paras_meta_training(paras):
    """
    Step 2: Meta-training
    Output command line parameters
    """
    paras.train_or_test = 0
    space = ' '  # 用于输出对齐
    if paras.seed == None:
        paras.seed = random.randint(0, 2000)
    verbose("===================================================================")
    verbose("Operation: Meta-training")
    verbose("===================================================================")
    verbose("Running script 'main.py' with parameters:")
    verbose("python main.py " + "--config " + paras.config + " \\")
    verbose(space * 15 + "--train_or_test " + str(paras.train_or_test) + " \\")
    verbose(space * 15 + "--train_vali_test_load " + paras.train_vali_test_load + " \\")
    verbose(space * 15 + "--seed " + str(paras.seed))
    return paras


def meta_training(config, paras):
    """Start meta-training"""
    trainer = Trainer(config, paras)
    best_model = trainer.exec()
    # 如果是automode，这里需要返回训练好的模型位置，用于自动衔接后续测试
    if paras.automode == 1:
        paras.trained_model = best_model
    paras.seed = None
    return paras


def paras_meta_testing(paras):
    """
    Step 3: Meta-testing
    Output command line parameters
    """
    paras.train_or_test = 1
    space = ' '  # 用于输出对齐
    if paras.seed == None:
        paras.seed = random.randint(0, 2000)
    verbose("===================================================================")
    verbose("Operation: Meta-testing")
    verbose("===================================================================")
    verbose("Running script 'main.py' with parameters:")
    verbose("python main.py " + "--config " + paras.config + " \\")
    verbose(space * 15 + "--train_or_test " + str(paras.train_or_test) + " \\")
    verbose(space * 15 + "--train_vali_test_load " + paras.train_vali_test_load + " \\")
    verbose(space * 15 + "--trained_model " + paras.trained_model + " \\")
    verbose(space * 15 + "--seed " + str(paras.seed))
    return paras


def meta_testing(config, paras):
    """Start meta-testing"""
    tester = Tester(config, paras)
    (
        Average_loss,
        Average_accuracy,
        Average_recall,
        Average_precision,
        Average_TNR,
        Average_FPR,
        Average_FNR,
        Average_f1,
    ) = tester.exec()
    verbose("Average performance of all tested CWEs: ")
    verbose("Average Loss: " + str(Average_loss))
    verbose("Average Accuracy: " + str(Average_accuracy))
    verbose("Average Recall: " + str(Average_recall))
    verbose("Average Precision: " + str(Average_precision))
    verbose("Average TNR: " + str(Average_TNR))
    verbose("Average FPR: " + str(Average_FPR))
    verbose("Average FNR: " + str(Average_FNR))
    verbose("Average f1: " + str(Average_f1))
    return


if __name__ == '__main__':
    faulthandler.enable()
    # GPU support is recommended.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set program arguments
    paras, config = set_args()

    # Redirect stdout and stderr to log files
    var_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = redir_log(var_time)
    LogOutputFile = open(log_path, 'w')
    sys.stdout = LogOutputFile
    sys.stderr = LogOutputFile

    # show current time
    verbose("===================================================================")
    verbose("Experiment time: {}".format(var_time))

    # Show the script execution mode
    show_automode(paras)

    # Step 1: Split training, validation, and test data
    if paras.sep_train_vali_test == True or paras.automode == 1:
        # Output command line parameters
        paras = paras_split_trai_vali_test(paras)
        # Get meta-training and meta-testing data based on the experiment scheme
        paras = get_and_store_trai_vali_test_data(config, paras)
        time.sleep(5)

    # Step 2: Meta-training
    if paras.train_or_test == 0 or paras.automode == 1:
        # Output command line parameters
        paras = paras_meta_training(paras)
        # Start meta-training
        paras = meta_training(config, paras)
        K.clear_session()
        time.sleep(5)

    # Step 3: Meta-testing
    if paras.train_or_test == 1 or paras.automode == 1:
        # Output command line parameters
        paras = paras_meta_testing(paras)
        # Start meta-testing
        meta_testing(config, paras)
        K.clear_session()

    # close log file
    LogOutputFile.close()
